"""低照度复现实验的通用辅助函数。"""

import csv
import logging
import math
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch


BEIJING_TZ = timezone(timedelta(hours=8))
# 版本 2 恢复原仓库 VGG 无梯度语义；旧 state 不能安全续训。
TRAINING_SEMANTICS_VERSION = 2


class BeijingFormatter(logging.Formatter):
    """将所有日志时间固定转换为北京时间。"""

    def formatTime(self, record, datefmt=None):  # noqa: N802
        timestamp = datetime.fromtimestamp(record.created, timezone.utc)
        return timestamp.astimezone(BEIJING_TZ).strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def create_logger(name: str, log_path: Path) -> logging.Logger:
    """创建同时写入终端和文件的北京时间日志器。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 同一实验重复启动时先关闭旧 handler，避免日志重复输出。
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = BeijingFormatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def format_duration(seconds: float) -> str:
    """将秒数格式化为可超过 24 小时的 HH:MM:SS。"""
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_int = int(round(seconds))
    hours, remain = divmod(seconds_int, 3600)
    minutes, seconds = divmod(remain, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_train_status(
    project: str,
    epoch: int,
    total_epochs: int,
    global_iter: int,
    total_iters: int,
    step: int,
    steps_per_epoch: int,
    elapsed_seconds: float,
    eta_seconds: float,
    learning_rate: float,
    total_loss: float,
    losses: Dict[str, float],
) -> str:
    """生成固定字段顺序的周期训练状态行。"""
    loss_text = ", ".join(f"{name}={value:.4f}" for name, value in losses.items())
    return (
        f"[{project}][TRAIN] "
        f"[progress: epoch={epoch:,}/{total_epochs:,}, iter={global_iter:,}/{total_iters:,}, "
        f"step={step:,}/{steps_per_epoch:,}] "
        f"[time: elapsed={format_duration(elapsed_seconds)}, eta={format_duration(eta_seconds)}] "
        f"[optim: lr={learning_rate:.3e}] "
        f"[total_loss: {total_loss:.4f}] [loss: {loss_text}]"
    )


def format_val_status(
    project: str,
    epoch: int,
    total_epochs: int,
    global_iter: int,
    total_iters: int,
    dataset_name: str,
    psnr: float,
    rgb_ssim: float,
    best_psnr: float,
    best_rgb_ssim: Optional[float],
    updated: bool,
) -> str:
    """生成固定字段顺序的完整验证状态行。"""
    best_ssim_text = "unknown" if best_rgb_ssim is None else f"{best_rgb_ssim:.4f}"
    return (
        f"[{project}][VAL] "
        f"[progress: epoch={epoch:,}/{total_epochs:,}, iter={global_iter:,}/{total_iters:,}] "
        f"[data: name={dataset_name}] "
        f"[metric: psnr={psnr:.4f}, rgb_ssim={rgb_ssim:.4f}] "
        f"[best: key=psnr, value={best_psnr:.4f}, rgb_ssim={best_ssim_text}, "
        f"updated={'yes' if updated else 'no'}]"
    )


def prepare_experiment_dirs(exp_dir: Path) -> Dict[str, Path]:
    """创建统一的实验产物目录。"""
    directories = {
        "root": exp_dir,
        "models": exp_dir / "models",
        "states": exp_dir / "training_state",
        "logs": exp_dir / "logs",
        "tensorboard": exp_dir / "tb_looger",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """返回 DataParallel 包装前的实际网络。"""
    return model.module if hasattr(model, "module") else model


def _normalise_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """移除 DataParallel 产生的 module. 前缀。"""
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def extract_state_dict(payload) -> Dict[str, torch.Tensor]:
    """兼容新旧 checkpoint，提取生成网络 state_dict。"""
    if isinstance(payload, torch.nn.Module):
        return _normalise_state_dict(payload.state_dict())
    if isinstance(payload, dict):
        state_dict = payload.get("state_dict", payload.get("model", payload))
        if isinstance(state_dict, torch.nn.Module):
            state_dict = state_dict.state_dict()
        if isinstance(state_dict, dict):
            return _normalise_state_dict(state_dict)
    raise TypeError("checkpoint 中未找到可加载的生成网络 state_dict")


def require_generator_checkpoint(checkpoint_path: Path) -> Path:
    """验证显式传入的是标准生成网络 checkpoint，而不是 training state。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"生成网络 checkpoint 不存在：{checkpoint_path}")
    if not checkpoint_path.name.endswith("_G.pth"):
        raise ValueError(
            f"只接受标准 *_G.pth 生成网络 checkpoint，当前为：{checkpoint_path.name}。"
        )
    return checkpoint_path


def load_generator_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = True,
) -> Tuple[Tuple[Iterable[str], Iterable[str]], Dict[str, torch.Tensor]]:
    """从仅权重或旧式完整模型 checkpoint 加载生成网络。"""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(payload)
    incompatibility = unwrap_model(model).load_state_dict(state_dict, strict=strict)
    return (incompatibility.missing_keys, incompatibility.unexpected_keys), state_dict


def _iteration_from_path(path: Path) -> int:
    """从定时权重或 state 文件名中读取迭代次数。"""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def find_latest_state(states_dir: Path) -> Optional[Path]:
    """按保存的 global iteration 找到最新 training state。"""
    candidates = list(states_dir.glob("*.state"))
    return max(candidates, key=_iteration_from_path) if candidates else None


def save_checkpoint(
    directories: Dict[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    iteration: int,
    elapsed_seconds: float,
    best_psnr: float,
    best_rgb_ssim: Optional[float],
    is_best: bool,
    save_snapshot: bool,
) -> Path:
    """保存 latest、best、定时生成权重及完整训练状态，并返回 state 路径。"""
    model_state = unwrap_model(model).state_dict()
    latest_path = directories["models"] / "latest_G.pth"
    torch.save(model_state, latest_path)

    if save_snapshot:
        snapshot_path = directories["models"] / f"{iteration:06d}_G.pth"
        torch.save(model_state, snapshot_path)

    if is_best:
        torch.save(model_state, directories["models"] / "best_G.pth")

    state = {
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "epoch": epoch,
        "iteration": iteration,
        "elapsed_seconds": elapsed_seconds,
        "best_psnr": best_psnr,
        "best_rgb_ssim": best_rgb_ssim,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        # state 可能从不同工作目录恢复，因此保存绝对权重路径。
        "model_path": str(latest_path.resolve()),
    }
    state_path = directories["states"] / f"{iteration:06d}.state"
    torch.save(state, state_path)
    return state_path


def resume_training(
    directories: Dict[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, object]:
    """自动恢复最新完整 state；缺少 state 时仅恢复 latest_G 权重。"""
    default = {
        "epoch": 1,
        "iteration": 0,
        "elapsed_seconds": 0.0,
        "best_psnr": float("-inf"),
        "best_rgb_ssim": None,
    }
    state_path = find_latest_state(directories["states"])
    if state_path is not None:
        state = torch.load(state_path, map_location=device, weights_only=False)
        state_version = state.get("training_semantics_version")
        if state_version != TRAINING_SEMANTICS_VERSION:
            raise RuntimeError(
                f"{state_path} 来自修复前的训练语义（version={state_version}），"
                "继续恢复可能保持全黑塌缩。请保留旧实验目录，并使用新实验目录从头训练。"
            )
        checkpoint_path = Path(state.get("model_path", directories["models"] / "latest_G.pth"))
        if not checkpoint_path.exists():
            checkpoint_path = directories["models"] / "latest_G.pth"
        load_generator_weights(model, checkpoint_path, device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        default.update({key: state[key] for key in default if key in state})
        logger.info(
            "[resume] 从 %s 恢复：epoch=%s, iter=%s, elapsed=%s",
            state_path,
            default["epoch"],
            default["iteration"],
            format_duration(float(default["elapsed_seconds"])),
        )
        return default

    latest_path = directories["models"] / "latest_G.pth"
    if latest_path.exists():
        load_generator_weights(model, latest_path, device)
        logger.info("[resume] 未发现 training state，仅恢复 %s；optimizer 和 scheduler 从头开始。", latest_path)
    return default


def seed_everything(seed: int) -> None:
    """设置单卡训练需要的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_psnr(pred_rgb_255: np.ndarray, gt_rgb_255: np.ndarray) -> float:
    """按 BasicSR 等价的 RGB 联合 MSE 计算 crop_border=0 的 PSNR。"""
    _validate_rgb_pair(pred_rgb_255, gt_rgb_255)
    mse = np.mean((pred_rgb_255.astype(np.float64) - gt_rgb_255.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10((255.0**2) / mse))


def _ssim_single_channel(pred: np.ndarray, gt: np.ndarray) -> float:
    """使用 11x11、sigma=1.5 Gaussian 窗口计算单通道 SSIM。"""
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mu_pred = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
    mu_gt = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]
    sigma_pred = cv2.filter2D(pred**2, -1, window)[5:-5, 5:-5] - mu_pred**2
    sigma_gt = cv2.filter2D(gt**2, -1, window)[5:-5, 5:-5] - mu_gt**2
    sigma_cross = cv2.filter2D(pred * gt, -1, window)[5:-5, 5:-5] - mu_pred * mu_gt
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu_pred * mu_gt + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred**2 + mu_gt**2 + c1) * (sigma_pred + sigma_gt + c2)
    return float((numerator / denominator).mean())


def calculate_rgb_ssim(pred_rgb_255: np.ndarray, gt_rgb_255: np.ndarray) -> float:
    """按 BasicSR 等价口径计算 RGB 三通道平均 SSIM。"""
    _validate_rgb_pair(pred_rgb_255, gt_rgb_255)
    if pred_rgb_255.shape[0] < 11 or pred_rgb_255.shape[1] < 11:
        raise ValueError("RGB SSIM 至少需要 11x11 图像，当前图像过小。")
    return float(np.mean([_ssim_single_channel(pred_rgb_255[..., channel], gt_rgb_255[..., channel]) for channel in range(3)]))


def _validate_rgb_pair(pred: np.ndarray, gt: np.ndarray) -> None:
    """验证统一指标所需的 HWC RGB 同尺寸输入。"""
    if pred.shape != gt.shape:
        raise ValueError(f"预测图与 GT 尺寸不一致：{pred.shape} vs {gt.shape}")
    if pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError(f"统一指标只接受 HWC RGB 图像，当前形状为 {pred.shape}")


class MetricEvaluator:
    """在一个进程内复用 LPIPS-Alex-v0.1，避免逐图重复加载。"""

    def __init__(self, device: torch.device):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("缺少 lpips，请先按 README 安装 requirements.txt。") from exc
        self.device = device
        self.lpips = lpips.LPIPS(net="alex", version="0.1").to(device).eval()

    def calculate(self, pred_rgb_01: torch.Tensor, gt_rgb_01: torch.Tensor) -> Dict[str, float]:
        """对单张 RGB [0,1] NCHW 图像计算固定的三项质量指标。"""
        if pred_rgb_01.shape != gt_rgb_01.shape:
            raise ValueError(f"预测 Tensor 与 GT Tensor 尺寸不一致：{pred_rgb_01.shape} vs {gt_rgb_01.shape}")
        pred_hwc = pred_rgb_01.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        gt_hwc = gt_rgb_01.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        pred_m11 = pred_rgb_01.detach().clamp(0, 1) * 2.0 - 1.0
        gt_m11 = gt_rgb_01.detach().clamp(0, 1) * 2.0 - 1.0
        with torch.no_grad():
            lpips_value = float(self.lpips(pred_m11, gt_m11).item())
        return {
            "psnr": calculate_psnr(pred_hwc, gt_hwc),
            "rgb_ssim": calculate_rgb_ssim(pred_hwc, gt_hwc),
            "lpips": lpips_value,
        }


def calculate_model_complexity(model: torch.nn.Module, device: torch.device) -> Dict[str, object]:
    """按固定 1x3x256x256 和 THOP 口径统计参数量与计算量。"""
    model = unwrap_model(model).to(device).eval()
    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    try:
        from thop import profile

        dummy = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception as exc:
        raise RuntimeError(f"THOP 复杂度统计失败：{exc}") from exc
    return {
        "params_m": float(params_m),
        "gmacs_g": float(macs / 1e9),
        "gflops_g": float(2 * macs / 1e9),
        "input_size": "1x3x256x256",
        "complexity_tool": "thop.profile",
        "complexity_note": "GMACs=THOP返回值/1e9；GFLOPs=2*MACs/1e9；查表和索引算子可能未被 THOP 完整覆盖。",
    }


def write_metric_csv(csv_path: Path, row: Dict[str, object]) -> None:
    """以单行 CSV 保存一个完整测试集的平均指标与复杂度。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def copy_best_or_latest(exp_dir: Path) -> Path:
    """返回优先级为 best、latest 的默认生成权重，供文档和脚本检查使用。"""
    for name in ("best_G.pth", "latest_G.pth"):
        candidate = exp_dir / "models" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"未在 {exp_dir / 'models'} 找到 best_G.pth 或 latest_G.pth")
