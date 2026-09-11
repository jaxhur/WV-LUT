# WV-LUT：三 LOL 数据集统一复现

本项目复现 **WV-LUT: Wide Vision Lookup Tables for Real-Time Low-Light Image Enhancement**。`common/architecture.py`、`common/GAM.py` 与 `common/finetune.py` 中的网络结构和 LUT 推理逻辑保持不变；本次仅补齐 LOL-v1、LOL-v2-syn、LOL-v2-real 的数据适配、训练/验证编排、checkpoint、统一评价和结果归档。

## 统一实验约定

- 三个数据集分别训练、分别在自身完整测试集验证和测试，不做混合训练。
- 论文原文使用两张 Tesla V100、总 `BatchSize=16`，基础训练为 `96×96` patch、`150000` iter；其显存版本未公开。为降低单卡显存占用，本仓库默认基础训练使用 `BatchSize=4`、`PatchSize=96`、`totalIter=150000`、`lr=1e-3 -> 1e-5`。这是单卡显存兼容配置，不能称为严格复刻论文的 batch 设置。
- LUT 微调保持原项目的 `PatchSize=256` 与 global iteration `150000 -> 160000`，即额外训练 `10000` iter、学习率 `1e-4 -> 1e-6`；单卡默认 `BatchSize=1`。论文原文的 LUT 微调是整图、batch 1，本仓库仍使用原项目已有的随机 patch 微调流程。
- 训练固定每 `20` 个 global iteration 输出一次 `TRAIN` 状态行；每 `1000` iter 在对应完整测试集验证一次，最后一个 iteration 必定再验证一次。
- 验证集 PSNR 严格提高时更新 `best_G.pth`。验证使用 `model.eval()` 和 `torch.no_grad()`，输入完整原图，不 resize、不 GT-Mean、不 self-ensemble。
- PSNR 是 RGB `[0,255]` 全通道联合 MSE、`crop_border=0`；SSIM 是 `11x11`、`sigma=1.5` 的 RGB 三通道平均、`crop_border=0`；LPIPS 为 RGB `[-1,1]` 的 AlexNet v0.1。
- 复杂度固定使用 THOP、`model.eval()` 和 `1x3x256x256` 输入，报告 `Params(M)`、`GMACs(G)` 和 `GFLOPs(G)=2*MACs/1e9`。

# 创建环境



```bash
git clone https://github.com/jaxhur/WV-LUT.git
git clone https://gitee.com/wallcaptain/WV-LUT.git

cd WV-LUT
conda env create -f environment.yml
conda activate wvlut-cu128

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```



# 数据集

```bash
pip install -U gdown
apt install -y unzip

cd ./data
# LOL-v1
gdown "https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z"
# LOL-v2
gdown "https://drive.google.com/uc?id=1L0UnJg6gZ4Eb7It2EuNxP0L3lQNmKMaP"


# 解压
unzip LOL-v1.zip -d LOL-v1
unzip LOL-v2-renamed.zip -d LOL-v2

rm LOL-v1.zip LOL-v2-renamed.zip
cd ../
```



# 训练



| Dataset | 训练集 | 完整验证/测试集 | BatchSize | Train PatchSize | 主训练总 iter |
|---|---|---|---:|---:|---:|
| LOL-v1 | `our485` | `eval15` | 4 | 96x96 | 150,000 |
| LOL-v2-syn | `Synthetic/Train` | `Synthetic/Test` | 4 | 96x96 | 150,000 |
| LOL-v2-real | `Real_captured/Train` | `Real_captured/Test` | 4 | 96x96 | 150,000 |



```bash
bash script/train.sh LOL-v1 experiments/wvlut_lolv1
# 把基础网络导出成真正的查找表。它不是训练命令
bash script/transferLUT.sh experiments/wvlut_lolv1
bash script/finetuneLUT.sh LOL-v1 experiments/wvlut_lolv1 experiments/wvlut_lolv1_lut

bash script/train.sh LOL-v2-real experiments/wvlut_lolv2_real
bash script/transferLUT.sh experiments/wvlut_lolv2_real
bash script/finetuneLUT.sh LOL-v2-real experiments/wvlut_lolv2_real experiments/wvlut_lolv2_real_lut


bash script/train.sh LOL-v2-syn experiments/wvlut_lolv2_syn
bash script/transferLUT.sh experiments/wvlut_lolv2_syn
bash script/finetuneLUT.sh LOL-v2-syn experiments/wvlut_lolv2_syn experiments/wvlut_lolv2_syn_lut

```

若单张 24GB 显卡使用默认值仍然 OOM，请保持模型和 `96×96` 训练 patch 不变，先把 micro-batch 降为 `2`；仍 OOM 再降为 `1`：

```bash
BATCH_SIZE=2 bash script/train.sh LOL-v1 experiments/wvlut_lolv1
# 若仍 OOM：BATCH_SIZE=1 bash script/train.sh LOL-v1 experiments/wvlut_lolv1
```

`BATCH_SIZE` 仅改变每次 optimizer update 的样本数；`totalIter`、loss、验证频率和统一评价口径均不变。请在实验记录中注明实际 batch，避免与论文的两卡总 batch 16 混淆。

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









# 测试

LUT 微调模型测试时传入第四个参数，即基础实验生成的 LUT 目录：

```bash
bash script/eval.sh LOL-v1 wvlut_lolv1_lut experiments/wvlut_lolv1_lut/models/best_G.pth experiments/wvlut_lolv1/luts
bash script/eval.sh LOL-v2-real wvlut_lolv2_real_lut experiments/wvlut_lolv2_real_lut/models/best_G.pth experiments/wvlut_lolv2_real/luts
bash script/eval.sh LOL-v2-syn wvlut_lolv2_syn_lut experiments/wvlut_lolv2_syn_lut/models/best_G.pth experiments/wvlut_lolv2_syn/luts
```

## 6. 重要边界

- 不要混用基础网络 `WVLUT` checkpoint 与 LUT 微调模型 checkpoint：前者用于 `--model WVLUT`，后者必须搭配对应的 `--model LUT --lutDir <基础实验>/luts`。
- 若只想从头开始某个实验，显式删除该实验目录中的 `models/` 和 `training_state/` 后再启动；普通重启不要传 `--noResume`。
