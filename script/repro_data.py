"""LOL-v1、LOL-v2-syn 与 LOL-v2-real 的统一成对数据读取。"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetPaths:
    """一个 LOL 数据集的训练和完整测试集路径。"""

    dataset: str
    train_lq: Path
    train_gt: Path
    test_lq: Path
    test_gt: Path
    train_split: str
    test_split: str


def resolve_dataset_paths(data_root: str, dataset: str) -> DatasetPaths:
    """将可变数据根目录解析为固定 LOL 相对结构。"""
    root = Path(data_root)
    definitions = {
        "LOL-v1": {
            "train_lq": root / "LOL-v1" / "our485" / "low",
            "train_gt": root / "LOL-v1" / "our485" / "high",
            "test_lq": root / "LOL-v1" / "eval15" / "low",
            "test_gt": root / "LOL-v1" / "eval15" / "high",
            "train_split": "our485",
            "test_split": "eval15",
        },
        "LOL-v2-syn": {
            "train_lq": root / "LOL-v2" / "Synthetic" / "Train" / "Low",
            "train_gt": root / "LOL-v2" / "Synthetic" / "Train" / "Normal",
            "test_lq": root / "LOL-v2" / "Synthetic" / "Test" / "Low",
            "test_gt": root / "LOL-v2" / "Synthetic" / "Test" / "Normal",
            "train_split": "Synthetic/Train",
            "test_split": "Synthetic/Test",
        },
        "LOL-v2-real": {
            "train_lq": root / "LOL-v2" / "Real_captured" / "Train" / "Low",
            "train_gt": root / "LOL-v2" / "Real_captured" / "Train" / "Normal",
            "test_lq": root / "LOL-v2" / "Real_captured" / "Test" / "Low",
            "test_gt": root / "LOL-v2" / "Real_captured" / "Test" / "Normal",
            "train_split": "Real_captured/Train",
            "test_split": "Real_captured/Test",
        },
    }
    if dataset not in definitions:
        raise ValueError(f"不支持的数据集 {dataset}，可选值为 {', '.join(definitions)}")
    return DatasetPaths(dataset=dataset, **definitions[dataset])


def _normalised_relative_key(path: Path, root: Path) -> str:
    """将相对路径规范化为跨平台稳定的 LQ/GT 配对键。"""
    return path.relative_to(root).as_posix()


def build_pairs(lq_root: Path, gt_root: Path) -> List[Tuple[str, Path, Path]]:
    """按规范化相对路径构造并严格验证 LQ/GT 配对。"""
    for root, role in ((lq_root, "LQ"), (gt_root, "GT")):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} 目录不存在：{root}")

    def index_images(root: Path) -> Dict[str, Path]:
        indexed: Dict[str, Path] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = _normalised_relative_key(path, root)
            if key in indexed:
                raise RuntimeError(f"目录 {root} 存在重复相对路径键：{key}")
            indexed[key] = path
        if not indexed:
            raise RuntimeError(f"目录 {root} 中没有找到支持的图像文件。")
        return indexed

    lq_index = index_images(lq_root)
    gt_index = index_images(gt_root)
    missing_gt = sorted(set(lq_index) - set(gt_index))
    missing_lq = sorted(set(gt_index) - set(lq_index))
    if missing_gt or missing_lq:
        examples = []
        if missing_gt:
            examples.append(f"缺少 GT，例如：{missing_gt[:3]}")
        if missing_lq:
            examples.append(f"缺少 LQ，例如：{missing_lq[:3]}")
        raise RuntimeError("LQ/GT 规范化相对路径无法一一配对；" + "；".join(examples))
    return [(key, lq_index[key], gt_index[key]) for key in sorted(lq_index)]


def load_rgb(path: Path) -> np.ndarray:
    """读取一张图像并统一转换为 RGB HWC。"""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def rgb_to_tensor(image: np.ndarray) -> torch.Tensor:
    """将 RGB HWC uint8 图像转换为 [0,1] CHW Tensor。"""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)


class PairedTrainDataset(Dataset):
    """执行同位置随机裁剪和几何增强的低照度训练集。"""

    def __init__(self, lq_root: Path, gt_root: Path, patch_size: int):
        self.pairs = build_pairs(lq_root, gt_root)
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, lq_path, gt_path = self.pairs[index]
        lq = load_rgb(lq_path)
        gt = load_rgb(gt_path)
        if lq.shape != gt.shape:
            raise RuntimeError(f"成对图像尺寸不一致：{lq_path} {lq.shape}，{gt_path} {gt.shape}")
        height, width = lq.shape[:2]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"训练图像 {lq_path} 的尺寸 {height}x{width} 小于 patchSize={self.patch_size}。"
            )

        # 对 LQ 和 GT 使用同一裁剪坐标，保证监督空间对齐。
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        lq = lq[top : top + self.patch_size, left : left + self.patch_size]
        gt = gt[top : top + self.patch_size, left : left + self.patch_size]

        if random.random() < 0.5:
            lq, gt = np.fliplr(lq), np.fliplr(gt)
        if random.random() < 0.5:
            lq, gt = np.flipud(lq), np.flipud(gt)
        rotations = random.randint(0, 3)
        if rotations:
            lq, gt = np.rot90(lq, rotations), np.rot90(gt, rotations)
        return rgb_to_tensor(lq), rgb_to_tensor(gt)


class PairedBenchmark:
    """按完整原图顺序提供验证或测试所需的成对样本。"""

    def __init__(self, lq_root: Path, gt_root: Path):
        self.pairs = build_pairs(lq_root, gt_root)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        for key, lq_path, gt_path in self.pairs:
            lq = load_rgb(lq_path)
            gt = load_rgb(gt_path)
            if lq.shape != gt.shape:
                raise RuntimeError(f"成对图像尺寸不一致：{lq_path} {lq.shape}，{gt_path} {gt.shape}")
            yield key, rgb_to_tensor(lq).unsqueeze(0), rgb_to_tensor(gt).unsqueeze(0)


def _seed_worker(worker_id: int) -> None:
    """让 DataLoader worker 使用与主进程一致的可复现随机序列。"""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_train_loader(
    dataset: PairedTrainDataset,
    batch_size: int,
    worker_num: int,
    seed: int,
) -> DataLoader:
    """创建单卡训练 DataLoader，并保留最后一个非满 batch。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=worker_num,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=worker_num > 0,
    )
