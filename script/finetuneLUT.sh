#!/usr/bin/env bash
set -euo pipefail

# 第一个参数为数据集名，第二个参数为基础网络实验目录，第三个参数为 LUT 微调实验目录。
DATASET="${1:?请传入 LOL-v1、LOL-v2-syn 或 LOL-v2-real}"
BASE_EXP_DIR="${2:?请传入基础网络实验目录}"
EXP_DIR="${3:?请传入 LUT 微调实验目录}"
DATA_ROOT="${DATA_ROOT:-./data}"
# LUT 微调保留原项目的 batch=2；显存不足时可通过 FINETUNE_BATCH_SIZE 覆盖。
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-2}"
FINETUNE_PATCH_SIZE="${FINETUNE_PATCH_SIZE:-256}"

python script/finetuneLUT.py \
  --dataset "${DATASET}" \
  --dataRoot "${DATA_ROOT}" \
  --expDir "${EXP_DIR}" \
  --lutDir "${BASE_EXP_DIR}/luts" \
  --initCheckpoint "${BASE_EXP_DIR}/models/150000_G.pth" \
  --batchSize "${FINETUNE_BATCH_SIZE}" \
  --patchSize "${FINETUNE_PATCH_SIZE}" \
  --initialIter 150000 \
  --totalIter 160000 \
  --printFreq 20 \
  --valFreq 1000 \
  --saveFreq 1000 \
  --lr0 1e-4 \
  --lr1 1e-6
