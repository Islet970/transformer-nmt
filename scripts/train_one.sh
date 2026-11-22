#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NAME="${1:-transformer_improved}"
DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/hermionegrangerislet/translationdataset/data}"
SP_MODEL="${SP_MODEL:-/kaggle/working/prepared/spm_zh_en.model}"
RUNS_DIR="${RUNS_DIR:-runs}"
EPOCHS="${EPOCHS:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ ! -f "configs/${NAME}.yaml" ]; then
  echo "config not found: configs/${NAME}.yaml"
  echo "available:"
  ls -1 configs
  exit 1
fi

if [ ! -f "$SP_MODEL" ]; then
  echo "tokenizer not found: $SP_MODEL"
  echo "run: bash scripts/prepare.sh"
  exit 1
fi

if [ "$SKIP_EXISTING" = "1" ] && [ -f "$RUNS_DIR/$NAME/summary.json" ]; then
  echo "skip $NAME: $RUNS_DIR/$NAME/summary.json already exists"
  exit 0
fi

python train.py \
  --config "configs/${NAME}.yaml" \
  --data-dir "$DATA_DIR" \
  --sp-model "$SP_MODEL" \
  --output-dir "$RUNS_DIR" \
  --run-name "$NAME" \
  --epochs "$EPOCHS" \
  $EXTRA_ARGS
