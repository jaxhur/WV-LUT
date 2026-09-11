# WV-LUT：三 LOL 数据集统一复现

本项目复现 **WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement**。`common/architecture.py`、`common/GAM.py` 与 `common/finetune.py` 中的网络结构和 LUT 推理逻辑保持不变；本次仅补齐 LOL-v1、LOL-v2-syn、LOL-v2-real 的数据适配、训练/验证编排、checkpoint、统一评价和结果归档。

## 统一实验约定

- 三个数据集分别训练、分别在自身完整测试集验证和测试，不做混合训练。
- 主训练沿用原项目的 `BatchSize=8`、`PatchSize=96`、`totalIter=150000`、`lr=1e-3 -> 1e-5`。
- LUT 微调沿用原项目的第二阶段配置：`BatchSize=2`、`PatchSize=256`、global iteration 从 `150000` 继续到 `160000`，即额外训练 `10000` iter，学习率为 `1e-4 -> 1e-6`。
- 训练固定每 `20` 个 global iteration 输出一次 `TRAIN` 状态行；每 `1000` iter 在对应完整测试集验证一次，最后一个 iteration 必定再验证一次。
- 验证集 PSNR 严格提高时更新 `best_G.pth`。验证使用 `model.eval()` 和 `torch.no_grad()`，输入完整原图，不 resize、不 GT-Mean、不 self-ensemble。
- PSNR 是 RGB `[0,255]` 全通道联合 MSE、`crop_border=0`；SSIM 是 `11x11`、`sigma=1.5` 的 RGB 三通道平均、`crop_border=0`；LPIPS 为 RGB `[-1,1]` 的 AlexNet v0.1。
- 复杂度固定使用 THOP、`model.eval()` 和 `1x3x256x256` 输入，报告 `Params(M)`、`GMACs(G)` 和 `GFLOPs(G)=2*MACs/1e9`。

## 1. 在 RTX 4090 / RTX 5090 服务器创建环境

训练环境目标是 CUDA 12.8 PyTorch runtime，因此不要安装旧项目中的 PyTorch 1.11。下面命令在远程 Linux 服务器的项目根目录执行：

```bash
conda env create -f environment.yml
conda activate wvlut-cu128

pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

切换到每一台 4090 或 5090 服务器后，先执行下面的最小环境检查；系统 `nvcc` 版本不能替代 `torch.version.cuda`。

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA runtime:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')
assert torch.cuda.is_available()
assert tuple(map(int, torch.version.cuda.split('.'))) >= (12, 8)
PY
```

项目没有自定义 CUDA 扩展，也没有在代码、配置或脚本中写死显卡型号、显存或 device index。`--device cuda` 默认使用当前 `CUDA_VISIBLE_DEVICES` 可见的单张卡。每次换服务器仍应完成一次真实 batch 的前向、反向和 AMP/扩展（若后续启用）的冒烟检查后再开完整训练。

首次训练会由 torchvision 下载 ImageNet VGG16 感知损失权重；服务器不能联网时，请先按 torchvision 官方缓存机制准备该权重。LPIPS Alex 权重同样应在服务器缓存或可联网状态下首次初始化。

## 2. 下载并放置数据集

数据根目录可任意命名。以下示例使用项目根目录下的 `data/`，并保持数据集内部目录不变：

```bash
pip install -U gdown
mkdir -p data
mkdir -p data/downloads
# Retinexformer 作者维护的公开数据链接，LOLv2 压缩包同时包含 Synthetic 和 Real_captured。
gdown --fuzzy \
  "https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing" \
  -O data/downloads/LOLv1.zip
gdown --fuzzy \
  "https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing" \
  -O data/downloads/LOLv2.zip

unzip data/downloads/LOLv1.zip -d data/downloads/LOLv1
unzip data/downloads/LOLv2.zip -d data/downloads/LOLv2
```

如果 `gdown` 因 Google Drive 权限或限流失败，请在浏览器手动下载 [LOLv1](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing) 与 [LOLv2](https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U/view?usp=sharing)。解压后，把 `data/downloads/` 中的实际数据目录移动/重命名到下述 `<data_root>` 结构；如果压缩包额外套了一层同名目录，应先整理掉重复层级。`LOLv2.zip` 同时提供 Synthetic 和 Real_captured。最终 `<data_root>` 下必须是：

```text
<data_root>/LOL-v1/our485/{low,high}
<data_root>/LOL-v1/eval15/{low,high}

<data_root>/LOL-v2/Synthetic/Train/{Low,Normal}
<data_root>/LOL-v2/Synthetic/Test/{Low,Normal}

<data_root>/LOL-v2/Real_captured/Train/{Low,Normal}
<data_root>/LOL-v2/Real_captured/Test/{Low,Normal}
```

数据读取器递归扫描 LQ 和 GT，并按规范化相对路径一一配对；发现缺图、重复键、尺寸不一致或 patch 小于训练裁剪尺寸时会明确报错，不会按文件名排序后静默错配。

## 3. 三数据集训练

从项目根目录运行。`DATA_ROOT` 可替换为服务器上的任意绝对数据路径。每条训练命令会自动在同一 `experiments/<实验名>/` 中恢复最新 `training_state/*.state`；如果只剩 `latest_G.pth`，则仅恢复模型权重并在日志中明确说明 optimizer/scheduler 未精确恢复。

| Dataset | 训练集 | 完整验证/测试集 | BatchSize | Train PatchSize | 主训练总 iter |
|---|---|---|---:|---:|---:|
| LOL-v1 | `our485` | `eval15` | 8 | 96x96 | 150,000 |
| LOL-v2-syn | `Synthetic/Train` | `Synthetic/Test` | 8 | 96x96 | 150,000 |
| LOL-v2-real | `Real_captured/Train` | `Real_captured/Test` | 8 | 96x96 | 150,000 |

每个数据集真实的 `steps_per_epoch` 由服务器上实际配对成功的图像数与 `DataLoader(drop_last=false)` 自动计算并写入 `train.log`；不能在未读取服务器数据前硬编码。若图像数为 `N`，则 `steps_per_epoch=ceil(N/8)`。

```bash
export DATA_ROOT=/path/to/data

bash script/train.sh LOL-v1 experiments/wvlut_lolv1
bash script/train.sh LOL-v2-syn experiments/wvlut_lolv2_syn
bash script/train.sh LOL-v2-real experiments/wvlut_lolv2_real
```

每个实验目录统一为：

```text
experiments/<实验名>/
  models/
    latest_G.pth
    best_G.pth
    <iter>_G.pth
  training_state/
    <iter>.state
  logs/
    train.log
    val.log
  tb_looger/
```

`train.log` 和终端训练行使用同一条 logger 消息，固定为北京时间 `YYYY-MM-DD HH:MM:SS INFO:`，并包含 epoch、global iter、epoch 内 step、elapsed、ETA、学习率、`total_loss`、`smooth_l1` 和加权后的 `perceptual`。`val.log` 独立保存完整测试集的 PSNR、RGB SSIM、LPIPS，以及历史最佳 PSNR 与其对应 RGB SSIM。

## 4. 导出 LUT 与第二阶段微调

基础网络训练完成后，以显式 checkpoint 导出 LUT。优先使用 `best_G.pth`；若想复现实验中的其他定时权重，直接把第二个参数替换为对应的 `*_G.pth`。

```bash
bash script/transferLUT.sh experiments/wvlut_lolv1
```

LUT 文件写入 `experiments/wvlut_lolv1/luts/`。随后开始 LUT 微调；它使用独立实验目录以避免覆盖基础网络 checkpoint。`initialIter=150000`、`totalIter=160000` 表示该阶段新增 10,000 次更新，日志中的 global iteration 延续原论文阶段编号。

```bash
export DATA_ROOT=/path/to/data

bash script/finetuneLUT.sh \
  LOL-v1 \
  experiments/wvlut_lolv1 \
  experiments/wvlut_lolv1_lut
```

对 LOL-v2-syn / LOL-v2-real 时只替换数据集名和对应基础/微调实验目录。微调同样自动续训，并生成自己的 `latest_G.pth`、`best_G.pth`、定时权重和 `training_state/*.state`。

## 5. 完整测试、增强图与 metric.csv

测试 checkpoint 必须显式传入，测试脚本不会猜测 latest 或 best。基础网络测试示例：

```bash
export DATA_ROOT=/path/to/data

bash script/eval.sh \
  LOL-v1 \
  wvlut_lolv1 \
  experiments/wvlut_lolv1/models/best_G.pth
```

LUT 微调模型测试时传入第四个参数，即基础实验生成的 LUT 目录：

```bash
bash script/eval.sh \
  LOL-v1 \
  wvlut_lolv1_lut \
  experiments/wvlut_lolv1_lut/models/best_G.pth \
  experiments/wvlut_lolv1/luts
```

同样分别对 `LOL-v2-syn` 和 `LOL-v2-real` 执行测试。每次测试会输出：

```text
test_result/<实验名>/<数据集名>/
  enhanced/                 # 与 LQ 相同的规范化相对路径和文件名
  metric.csv                # 整个测试集的平均质量指标与模型复杂度
  test.log
```

`metric.csv` 包含实验名、训练/测试 split、PSNR、RGB SSIM、LPIPS-Alex-v0.1、`Params(M)`、`GMACs(G)`、`GFLOPs(G)`、统一指标口径、输入尺寸、显式 checkpoint 路径和增强图目录。测试不提供 GT-Mean 开关，因此不会使用 GT 信息校正网络输出。

## 6. 重要边界

- 不要混用基础网络 `WVLUT` checkpoint 与 LUT 微调模型 checkpoint：前者用于 `--model WVLUT`，后者必须搭配对应的 `--model LUT --lutDir <基础实验>/luts`。
- 若只想从头开始某个实验，显式删除该实验目录中的 `models/` 和 `training_state/` 后再启动；普通重启不要传 `--noResume`。
- 训练、验证和测试默认单卡；通过 `CUDA_VISIBLE_DEVICES` 选择租用的 4090 或 5090，不需要改项目代码。
- THOP 对 LUT 查表、索引和部分 elementwise 操作可能统计不完整；`metric.csv` 会保留这一口径说明，不能把其数值混同于其他复杂度工具。

## 引用

```text
@ARTICLE{wvlut,
  author={Li, Canlin and Su, Haowen and Tan, Xin and Zhang, Xiangfei and Ma, Lizhuang},
  journal={IEEE Transactions on Multimedia},
  title={WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement},
  year={2025},
  doi={10.1109/TMM.2025.3535342}
}
```
