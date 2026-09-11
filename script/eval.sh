#!/usr/bin/env bash
set -euo pipefail

# 依次传入数据集名、实验名、待测试的 *_G.pth；可选第四个参数为 LUT 目录。
DATASET="${1:?请传入 LOL-v1、LOL-v2-syn 或 LOL-v2-real}"
EXPERIMENT_NAME="${2:?请传入实验名}"
CHECKPOINT="${3:?请传入待测试的 *_G.pth}"
LUT_DIR="${4:-}"
DATA_ROOT="${DATA_ROOT:-./data}"

MODEL_ARGS=(--model WVLUT)
if [[ -n "${LUT_DIR}" ]]; then
  MODEL_ARGS=(--model LUT --lutDir "${LUT_DIR}")
fi

python script/eval.py \
  --dataset "${DATASET}" \
  --dataRoot "${DATA_ROOT}" \
  --experimentName "${EXPERIMENT_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --resultRoot test_result \
  "${MODEL_ARGS[@]}"
