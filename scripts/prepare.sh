#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/hermionegrangerislet/translationdataset/data}"
OUT_DIR="${OUT_DIR:-/kaggle/working/prepared}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"

if [ -f "$OUT_DIR/spm_zh_en.model" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "tokenizer already exists: $OUT_DIR/spm_zh_en.model"
  echo "set FORCE=1 to retrain it"
  exit 0
fi

python -m data.prepare \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --vocab-size "$VOCAB_SIZE"

echo "prepared: $OUT_DIR/spm_zh_en.model"
