#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/hermionegrangerislet/translationdataset/data}"
OUT_DIR="${OUT_DIR:-/kaggle/working/prepared}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"

python -m data.prepare \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --vocab-size "$VOCAB_SIZE"

echo "prepared: $OUT_DIR/spm_zh_en.model"
