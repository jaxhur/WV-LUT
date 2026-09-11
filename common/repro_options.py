"""统一复现实验的命令行参数定义。"""

import argparse


DATASET_CHOICES = ("LOL-v1", "LOL-v2-syn", "LOL-v2-real")


def add_shared_model_arguments(parser: argparse.ArgumentParser) -> None:
    """添加不改变 WV-LUT 结构的模型构造参数。"""
    parser.add_argument("--modelType", choices=("shared", "wo_shared"), default="shared")
    parser.add_argument("--nf", type=int, default=64)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--device", default="cuda", help="默认使用当前可见的单张 CUDA 设备。")


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """添加三个固定 LOL 数据集的数据根目录与数据集选择参数。"""
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--dataRoot", required=True, help="包含 LOL-v1 与 LOL-v2 的可变顶层数据目录。")


def build_train_parser(description: str) -> argparse.ArgumentParser:
    """构造基础网络和 LUT 微调共用的单卡训练参数解析器。"""
    parser = argparse.ArgumentParser(description=description)
    add_shared_model_arguments(parser)
    add_dataset_arguments(parser)
    parser.add_argument("--expDir", required=True, help="实验目录，例如 experiments/wvlut_lolv1。")
    # WV-LUT 在训练时会展开较多局部中间特征，单卡默认使用保守 micro-batch。
    parser.add_argument("--batchSize", type=int, default=4)
    parser.add_argument("--patchSize", type=int, default=96)
    parser.add_argument("--totalIter", type=int, default=150000)
    parser.add_argument("--initialIter", type=int, default=0, help="当前阶段开始前已完成的 global iteration。")
    parser.add_argument("--printFreq", type=int, default=20)
    parser.add_argument("--valFreq", type=int, default=1000)
    parser.add_argument("--saveFreq", type=int, default=1000)
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--lr1", type=float, default=1e-5)
    parser.add_argument("--weightDecay", type=float, default=0.0)
    parser.add_argument("--workerNum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--perceptualWeight", type=float, default=0.04)
    parser.add_argument("--noResume", dest="autoResume", action="store_false")
    parser.set_defaults(autoResume=True)
    return parser


def build_finetune_parser() -> argparse.ArgumentParser:
    """构造 LUT 微调参数解析器。"""
    parser = build_train_parser("使用导出的 LUT 进行单卡微调")
    parser.set_defaults(batchSize=2, patchSize=256, initialIter=150000, totalIter=160000, lr0=1e-4, lr1=1e-6)
    parser.add_argument("--lutDir", required=True, help="基础网络导出的 luts 目录。")
    parser.add_argument("--initCheckpoint", required=True, help="基础网络的 *_G.pth，用于初始化 GAM 与缩放参数。")
    return parser


def build_transfer_parser() -> argparse.ArgumentParser:
    """构造从基础网络权重导出 LUT 的参数解析器。"""
    parser = argparse.ArgumentParser(description="从 WV-LUT 生成网络导出查找表")
    add_shared_model_arguments(parser)
    parser.add_argument("--expDir", required=True, help="基础网络实验目录，LUT 将写入该目录的 luts/。")
    parser.add_argument("--checkpoint", required=True, help="显式传入基础网络 *_G.pth。")
    parser.add_argument("--modes", default="cdy")
    return parser


def build_test_parser() -> argparse.ArgumentParser:
    """构造完整测试集评估参数解析器。"""
    parser = argparse.ArgumentParser(description="在完整 LOL 测试集上评估 WV-LUT")
    add_shared_model_arguments(parser)
    add_dataset_arguments(parser)
    parser.add_argument("--model", choices=("WVLUT", "LUT"), default="WVLUT")
    parser.add_argument("--checkpoint", required=True, help="显式传入待测试的 *_G.pth。")
    parser.add_argument("--lutDir", default="", help="测试 LUT 微调模型时必须传入基础网络 luts 目录。")
    parser.add_argument("--resultRoot", default="test_result")
    parser.add_argument("--experimentName", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    return parser
