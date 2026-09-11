#!/usr/bin/env bash
set -euo pipefail

# 第一个参数是 LOL-v1、LOL-v2-syn 或 LOL-v2-real，数据根目录由 DATA_ROOT 覆盖。
DATASET="${1:?请传入 LOL-v1、LOL-v2-syn 或 LOL-v2-real}"
DATA_ROOT="${DATA_ROOT:-./data}"
EXP_DIR="${2:-experiments/wvlut_${DATASET}}"
# 单卡训练的降显存默认值；可通过 BATCH_SIZE 覆盖。
BATCH_SIZE="${BATCH_SIZE:-4}"
PATCH_SIZE="${PATCH_SIZE:-96}"

python script/train.py \
  --dataset "${DATASET}" \
  --dataRoot "${DATA_ROOT}" \
  --expDir "${EXP_DIR}" \
  --batchSize "${BATCH_SIZE}" \
  --patchSize "${PATCH_SIZE}" \
  --totalIter 150000 \
  --printFreq 20 \
  --valFreq 1000 \
  --saveFreq 1000 \
  --lr0 1e-3 \
  --lr1 1e-5
