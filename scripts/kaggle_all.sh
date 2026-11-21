#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/hermionegrangerislet/translationdataset/data}"
SP_MODEL="${SP_MODEL:-/kaggle/working/prepared/spm_zh_en.model}"
RUNS_DIR="${RUNS_DIR:-runs}"
EPOCHS="${EPOCHS:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ ! -f "$SP_MODEL" ]; then
  echo "tokenizer not found, preparing..."
  DATA_DIR="$DATA_DIR" OUT_DIR="$(dirname "$SP_MODEL")" bash scripts/prepare.sh
fi

DATA_DIR="$DATA_DIR" SP_MODEL="$SP_MODEL" RUNS_DIR="$RUNS_DIR" EPOCHS="$EPOCHS" \
  SKIP_EXISTING="$SKIP_EXISTING" bash scripts/run_all.sh

DATA_DIR="$DATA_DIR" SP_MODEL="$SP_MODEL" RUNS_DIR="$RUNS_DIR" \
  bash scripts/evaluate_all.sh

python analyze.py --runs-dir "$RUNS_DIR" --out-dir figures

echo "pipeline finished."
echo "checkpoints: $RUNS_DIR"
echo "figures: figures"
