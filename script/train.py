"""WV-LUT 基础网络和 LUT 微调的统一单卡训练入口。"""

import math
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

# 根据脚本自身位置定位项目根目录，避免依赖启动时的当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.architecture import WVLUT
from common.finetune import Finetune
from common.repro_options import build_finetune_parser, build_train_parser
from common.reproduction import (
    MetricEvaluator,
    create_logger,
    extract_state_dict,
    format_train_status,
    format_val_status,
    prepare_experiment_dirs,
    require_generator_checkpoint,
    resume_training,
    save_checkpoint,
    seed_everything,
)
from common.utils import net_loss
from script.repro_data import (
    PairedBenchmark,
    PairedTrainDataset,
    create_train_loader,
    resolve_dataset_paths,
)


def select_device(device_name: str) -> torch.device:
    """验证并返回当前可见的单张计算设备。"""
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 训练，但当前服务器未检测到可用 CUDA 设备。")
    return device


def build_scheduler(optimizer: Adam, total_iter: int, lr0: float, lr1: float) -> LambdaLR:
    """构造与原项目一致的余弦学习率调度器。"""
    if lr1 < 0:
        factor = lambda step: ((1 + math.cos(step * math.pi / total_iter)) / 2) * 0.8 + 0.2
    else:
        ratio = lr1 / lr0
        factor = lambda step: ((1 + math.cos(step * math.pi / total_iter)) / 2) * (1 - ratio) + ratio
    return LambdaLR(optimizer, lr_lambda=factor)


def initialise_finetune_model(args, device: torch.device, logger) -> Finetune:
    """由导出 LUT 和基础网络 checkpoint 构造 LUT 微调模型。"""
    lut_dir = Path(args.lutDir)
    if not lut_dir.is_dir():
        raise FileNotFoundError(f"LUT 目录不存在：{lut_dir}")
    model = Finetune(
        lut_folder=str(lut_dir),
        modes=["c", "d", "y"],
        interval=args.interval,
        model_path=None,
        model_type=args.modelType,
        freeze_non_lut=True,
    ).to(device)
    init_checkpoint = require_generator_checkpoint(Path(args.initCheckpoint))
    payload = torch.load(init_checkpoint, map_location=device, weights_only=False)
    source_state = extract_state_dict(payload)
    target_state = model.state_dict()
    transferable = {
        key: value
        for key, value in source_state.items()
        if key in target_state and key.startswith(("GAM.", "scale_s1.", "scale_s2."))
    }
    incompatibility = model.load_state_dict(transferable, strict=False)
    if not transferable:
        raise RuntimeError("基础网络 checkpoint 中未找到 GAM 或 scale_s1/scale_s2 参数，无法初始化 LUT 微调。")
    logger.info(
        "[finetune-init] 已从 %s 初始化 %d 个 GAM/scale 参数；missing=%d, unexpected=%d。",
        init_checkpoint,
        len(transferable),
        len(incompatibility.missing_keys),
        len(incompatibility.unexpected_keys),
    )
    return model


def build_model(stage: str, args, device: torch.device, logger) -> torch.nn.Module:
    """按当前训练阶段构造模型，不修改 WV-LUT 网络定义。"""
    if stage == "pretrain":
        return WVLUT(nf=args.nf, model_type=args.modelType).to(device)
    return initialise_finetune_model(args, device, logger)


def run_validation(
    model: torch.nn.Module,
    benchmark: PairedBenchmark,
    evaluator: MetricEvaluator,
    device: torch.device,
) -> Dict[str, float]:
    """在完整测试集原图上逐图推理并计算统一指标。"""
    values = {"psnr": [], "rgb_ssim": [], "lpips": []}
    model.eval()
    with torch.no_grad():
        for _, lq, gt in benchmark:
            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            prediction = model(lq).clamp(0, 1)
            metrics = evaluator.calculate(prediction, gt)
            for name, value in metrics.items():
                values[name].append(value)
    if not values["psnr"]:
        raise RuntimeError("验证集为空，无法计算平均指标。")
    return {name: float(np.mean(items)) for name, items in values.items()}


def next_batch(loader, iterator):
    """读取下一批数据；一个 epoch 结束后重新创建迭代器。"""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main(stage: str = "pretrain") -> None:
    """执行指定阶段的单卡训练、周期验证和自动断点续训。"""
    parser = build_train_parser("训练 WV-LUT 基础网络") if stage == "pretrain" else build_finetune_parser()
    args = parser.parse_args()
    if args.totalIter <= args.initialIter:
        raise ValueError("totalIter 必须大于 initialIter。")
    if args.printFreq != 20:
        raise ValueError("统一复现要求 printFreq 固定为 20。")
    if args.valFreq != 1000:
        raise ValueError("统一复现要求 valFreq 固定为 1000。")

    device = select_device(args.device)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    directories = prepare_experiment_dirs(Path(args.expDir))
    train_logger = create_logger(f"{args.expDir}.train", directories["logs"] / "train.log")
    val_logger = create_logger(f"{args.expDir}.val", directories["logs"] / "val.log")
    train_logger.info(
        "[startup] device=%s, cuda=%s, dataset=%s, stage=%s, seed=%d。",
        device,
        torch.version.cuda if device.type == "cuda" else "cpu",
        args.dataset,
        stage,
        args.seed,
    )

    paths = resolve_dataset_paths(args.dataRoot, args.dataset)
    train_dataset = PairedTrainDataset(paths.train_lq, paths.train_gt, args.patchSize)
    train_loader = create_train_loader(train_dataset, args.batchSize, args.workerNum, args.seed)
    benchmark = PairedBenchmark(paths.test_lq, paths.test_gt)
    steps_per_epoch = len(train_loader)
    phase_total_iter = args.totalIter - args.initialIter
    total_epochs = math.ceil(phase_total_iter / steps_per_epoch)
    train_logger.info(
        "[data] train=%s (%d pairs), val=%s (%d pairs), batch=%d, patch=%dx%d, steps_per_epoch=%d。",
        paths.train_split,
        len(train_dataset),
        paths.test_split,
        len(benchmark),
        args.batchSize,
        args.patchSize,
        args.patchSize,
        steps_per_epoch,
    )

    model = build_model(stage, args, device, train_logger)
    optimizer = Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weightDecay,
    )
    # 每个训练阶段都在自身有效迭代范围内完成声明的余弦退火。
    scheduler = build_scheduler(optimizer, phase_total_iter, args.lr0, args.lr1)
    default_state = {
        "epoch": 1,
        "iteration": args.initialIter,
        "elapsed_seconds": 0.0,
        "best_psnr": float("-inf"),
        "best_rgb_ssim": None,
    }
    state = resume_training(directories, model, optimizer, scheduler, device, train_logger) if args.autoResume else default_state
    global_iter = max(int(state["iteration"]), args.initialIter)
    elapsed_before_resume = float(state["elapsed_seconds"])
    best_psnr = float(state["best_psnr"])
    best_rgb_ssim = state["best_rgb_ssim"]
    if global_iter >= args.totalIter:
        train_logger.info("[finish] checkpoint 已达到 iter=%d/%d，无需继续训练。", global_iter, args.totalIter)
        return

    evaluator = MetricEvaluator(device)
    writer = SummaryWriter(log_dir=str(directories["tensorboard"]))
    iterator = iter(train_loader)
    phase_iter = global_iter - args.initialIter
    running_losses = {"smooth_l1": 0.0, "vgg_no_grad": 0.0, "total": 0.0}
    interval_start = time.perf_counter()
    train_start = time.perf_counter()
    last_validation_iter = global_iter

    try:
        while global_iter < args.totalIter:
            (lq, gt), iterator = next_batch(train_loader, iterator)
            global_iter += 1
            phase_iter += 1
            epoch = (phase_iter - 1) // steps_per_epoch + 1
            step = (phase_iter - 1) % steps_per_epoch + 1
            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(lq)
            smooth_l1 = F.smooth_l1_loss(prediction, gt)
            # 原仓库的 VGG 项位于 no_grad 中：保留报告总 loss 的公式，但只由 Smooth L1 提供梯度。
            vgg_no_grad = args.perceptualWeight * net_loss(prediction, gt)
            total_loss = smooth_l1 + vgg_no_grad
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            running_losses["smooth_l1"] += float(smooth_l1.detach())
            running_losses["vgg_no_grad"] += float(vgg_no_grad)
            running_losses["total"] += float(total_loss.detach())
            writer.add_scalar("train/smooth_l1", float(smooth_l1.detach()), global_iter)
            writer.add_scalar("train/vgg_no_grad", float(vgg_no_grad), global_iter)
            writer.add_scalar("train/total", float(total_loss.detach()), global_iter)

            if global_iter % args.printFreq == 0:
                averages = {name: value / args.printFreq for name, value in running_losses.items()}
                elapsed_seconds = elapsed_before_resume + (time.perf_counter() - train_start)
                seconds_per_iter = (time.perf_counter() - interval_start) / args.printFreq
                eta_seconds = seconds_per_iter * (args.totalIter - global_iter)
                train_logger.info(
                    format_train_status(
                        Path(args.expDir).name,
                        epoch,
                        total_epochs,
                        global_iter,
                        args.totalIter,
                        step,
                        steps_per_epoch,
                        elapsed_seconds,
                        eta_seconds,
                        scheduler.get_last_lr()[0],
                        averages["total"],
                        {"smooth_l1": averages["smooth_l1"], "vgg_no_grad": averages["vgg_no_grad"]},
                    )
                )
                running_losses = {"smooth_l1": 0.0, "vgg_no_grad": 0.0, "total": 0.0}
                interval_start = time.perf_counter()

            should_validate = global_iter % args.valFreq == 0 or global_iter == args.totalIter
            if should_validate:
                metrics = run_validation(model, benchmark, evaluator, device)
                updated = metrics["psnr"] > best_psnr
                if updated:
                    best_psnr = metrics["psnr"]
                    best_rgb_ssim = metrics["rgb_ssim"]
                elapsed_seconds = elapsed_before_resume + (time.perf_counter() - train_start)
                val_logger.info(
                    format_val_status(
                        Path(args.expDir).name,
                        epoch,
                        total_epochs,
                        global_iter,
                        args.totalIter,
                        f"{paths.dataset}/{paths.test_split}",
                        metrics["psnr"],
                        metrics["rgb_ssim"],
                        best_psnr,
                        best_rgb_ssim,
                        updated,
                    )
                )
                val_logger.info("[metric-detail] lpips_alex_v0.1=%.4f", metrics["lpips"])
                writer.add_scalar("val/psnr", metrics["psnr"], global_iter)
                writer.add_scalar("val/rgb_ssim", metrics["rgb_ssim"], global_iter)
                writer.add_scalar("val/lpips_alex_v0.1", metrics["lpips"], global_iter)
                state_path = save_checkpoint(
                    directories,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_iter,
                    elapsed_seconds,
                    best_psnr,
                    best_rgb_ssim,
                    is_best=updated,
                    save_snapshot=global_iter % args.saveFreq == 0 or global_iter == args.totalIter,
                )
                train_logger.info(
                    "[checkpoint] iter=%d，latest_G.pth、%s%s 已保存。",
                    global_iter,
                    state_path.name,
                    "，best_G.pth 已更新" if updated else "",
                )
                last_validation_iter = global_iter
            elif global_iter % args.saveFreq == 0:
                elapsed_seconds = elapsed_before_resume + (time.perf_counter() - train_start)
                state_path = save_checkpoint(
                    directories,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_iter,
                    elapsed_seconds,
                    best_psnr,
                    best_rgb_ssim,
                    is_best=False,
                    save_snapshot=True,
                )
                train_logger.info(
                    "[checkpoint] iter=%d，latest_G.pth、%s 与定时权重已保存。",
                    global_iter,
                    state_path.name,
                )
    finally:
        writer.close()

    if last_validation_iter != global_iter:
        raise RuntimeError("训练结束时未完成完整验证集验证。")
    train_logger.info(
        "[finish] 训练完成：iter=%d/%d，best_psnr=%.4f，best_rgb_ssim=%s。",
        global_iter,
        args.totalIter,
        best_psnr,
        "unknown" if best_rgb_ssim is None else f"{best_rgb_ssim:.4f}",
    )


if __name__ == "__main__":
    main()
