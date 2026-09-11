#!/usr/bin/env bash
set -euo pipefail

# 第一个参数是基础网络实验目录；第二个参数可覆盖默认的 best_G.pth。
EXP_DIR="${1:?请传入基础网络实验目录}"
CHECKPOINT="${2:-${EXP_DIR}/models/best_G.pth}"

python script/transferLUT.py \
  --expDir "${EXP_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --modelType shared \
  --nf 64 \
  --interval 4
