"""WV-LUT 三个 LOL 测试集的统一评估入口。"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 根据脚本自身位置定位项目根目录，避免依赖启动时的当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.architecture import WVLUT
from common.finetune import Finetune
from common.repro_options import build_test_parser
from common.reproduction import (
    MetricEvaluator,
    calculate_model_complexity,
    create_logger,
    load_generator_weights,
    require_generator_checkpoint,
    seed_everything,
    write_metric_csv,
)
from script.repro_data import PairedBenchmark, resolve_dataset_paths


def build_test_model(args, device: torch.device) -> torch.nn.Module:
    """构造基础网络或 LUT 微调网络，并加载用户显式指定的生成权重。"""
    if args.model == "WVLUT":
        model = WVLUT(nf=args.nf, model_type=args.modelType).to(device)
    else:
        if not args.lutDir:
            raise ValueError("测试 --model LUT 时必须显式传入 --lutDir。")
        model = Finetune(
            lut_folder=args.lutDir,
            modes=["c", "d", "y"],
            interval=args.interval,
            model_path=None,
            model_type=args.modelType,
            freeze_non_lut=True,
        ).to(device)
    checkpoint = require_generator_checkpoint(Path(args.checkpoint))
    missing_keys, unexpected_keys = load_generator_weights(model, checkpoint, device)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"checkpoint 与当前 {args.model} 结构不匹配：missing={list(missing_keys)}, "
            f"unexpected={list(unexpected_keys)}"
        )
    return model.eval()


def save_enhanced_image(prediction: torch.Tensor, target_path: Path) -> None:
    """按 LQ 的规范化相对路径保存一张 RGB 增强图。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image = prediction.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.round(image * 255.0).astype(np.uint8), mode="RGB").save(target_path)


def main() -> None:
    """在完整测试集上保存增强图、统一指标和复杂度 CSV。"""
    args = build_test_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 测试，但当前服务器未检测到可用 CUDA 设备。")
    seed_everything(args.seed)

    paths = resolve_dataset_paths(args.dataRoot, args.dataset)
    result_dir = Path(args.resultRoot) / args.experimentName / args.dataset
    enhanced_dir = result_dir / "enhanced"
    logger = create_logger(f"{args.experimentName}.{args.dataset}.test", result_dir / "test.log")
    logger.info(
        "[startup] dataset=%s, split=%s, model=%s, checkpoint=%s, device=%s。",
        args.dataset,
        paths.test_split,
        args.model,
        args.checkpoint,
        device,
    )

    benchmark = PairedBenchmark(paths.test_lq, paths.test_gt)
    model = build_test_model(args, device)
    evaluator = MetricEvaluator(device)
    values = {"psnr": [], "rgb_ssim": [], "lpips": []}

    with torch.no_grad():
        for key, lq, gt in benchmark:
            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            prediction = model(lq).clamp(0, 1)
            metrics = evaluator.calculate(prediction, gt)
            for name, value in metrics.items():
                values[name].append(value)
            save_enhanced_image(prediction, enhanced_dir / key)
            logger.info(
                "[image] key=%s, psnr=%.4f, rgb_ssim=%.4f, lpips=%.4f。",
                key,
                metrics["psnr"],
                metrics["rgb_ssim"],
                metrics["lpips"],
            )

    if not values["psnr"]:
        raise RuntimeError("完整测试集为空，未生成指标。")
    means = {name: float(np.mean(items)) for name, items in values.items()}
    complexity = calculate_model_complexity(model, device)
    row = {
        "experiment": args.experimentName,
        "dataset": args.dataset,
        "train_split": paths.train_split,
        "test_split": paths.test_split,
        "psnr": f"{means['psnr']:.4f}",
        "psnr_mode": "BasicSR-RGB-crop0",
        "ssim": f"{means['rgb_ssim']:.4f}",
        "ssim_mode": "BasicSR-RGB-channel-mean-crop0",
        "lpips": f"{means['lpips']:.4f}",
        "lpips_backbone": "alex",
        "lpips_version": "0.1",
        "lpips_range": "[-1,1]",
        "resize": "false",
        "gt_mean": "false",
        "self_ensemble": "false",
        "params_m": f"{complexity['params_m']:.4f}",
        "gmacs_g": f"{complexity['gmacs_g']:.4f}",
        "gflops_g": f"{complexity['gflops_g']:.4f}",
        "input_size": complexity["input_size"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "enhanced_images": str(enhanced_dir.resolve()),
        "metric_source": "project_eval",
        "complexity_tool": complexity["complexity_tool"],
        "complexity_note": complexity["complexity_note"],
    }
    write_metric_csv(result_dir / "metric.csv", row)
    logger.info(
        "[summary] psnr=%.4f, rgb_ssim=%.4f, lpips_alex_v0.1=%.4f, params_m=%.4f, "
        "gmacs_g=%.4f, gflops_g=%.4f。",
        means["psnr"],
        means["rgb_ssim"],
        means["lpips"],
        complexity["params_m"],
        complexity["gmacs_g"],
        complexity["gflops_g"],
    )


if __name__ == "__main__":
    main()
