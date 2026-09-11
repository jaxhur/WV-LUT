#!/usr/bin/env bash
set -euo pipefail

# 第一个参数是基础网络实验目录；第二个参数可覆盖原流程默认的 150000 iter 权重。
EXP_DIR="${1:?请传入基础网络实验目录}"
CHECKPOINT="${2:-${EXP_DIR}/models/150000_G.pth}"

python script/transferLUT.py \
  --expDir "${EXP_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --modelType shared \
  --nf 64 \
  --interval 4
