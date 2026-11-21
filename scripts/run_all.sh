#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/hermionegrangerislet/translationdataset/data}"
SP_MODEL="${SP_MODEL:-/kaggle/working/prepared/spm_zh_en.model}"
RUNS_DIR="${RUNS_DIR:-runs}"
EPOCHS="${EPOCHS:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CONFIGS=(
  "configs/lstm.yaml"
  "configs/transformer.yaml"
  "configs/transformer_improved.yaml"
  "configs/ablation_no_pe.yaml"
  "configs/ablation_1head.yaml"
  "configs/ablation_no_tie.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$RUNS_DIR/$name/summary.json" ]; then
    echo "skip $name: $RUNS_DIR/$name/summary.json already exists"
    continue
  fi
  echo "============================================================"
  echo "running: $name"
  echo "============================================================"
  python train.py \
    --config "$cfg" \
    --data-dir "$DATA_DIR" \
    --sp-model "$SP_MODEL" \
    --output-dir "$RUNS_DIR" \
    --run-name "$name" \
    --epochs "$EPOCHS" \
    $EXTRA_ARGS
done

echo "all 6 runs finished. results in $RUNS_DIR"
