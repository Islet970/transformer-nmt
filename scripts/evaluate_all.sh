#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/hermionegrangerislet/translationdataset/data}"
SP_MODEL="${SP_MODEL:-/kaggle/working/prepared/spm_zh_en.model}"
RUNS_DIR="${RUNS_DIR:-runs}"

RUNS=(
  "lstm"
  "transformer"
  "transformer_improved"
  "ablation_no_pe"
  "ablation_1head"
  "ablation_no_tie"
)

for name in "${RUNS[@]}"; do
  ckpt="$RUNS_DIR/$name/best.pt"
  if [ ! -f "$ckpt" ]; then
    ckpt="$RUNS_DIR/$name/last.pt"
  fi
  if [ ! -f "$ckpt" ]; then
    echo "skip $name: no checkpoint"
    continue
  fi

  echo "evaluating $name with greedy (full test)"
  python evaluate.py \
    --checkpoint "$ckpt" \
    --data-dir "$DATA_DIR" \
    --sp-model "$SP_MODEL" \
    --split test \
    --beam 1

  echo "evaluating $name with beam=4 (first 500 test sentences)"
  python evaluate.py \
    --checkpoint "$ckpt" \
    --data-dir "$DATA_DIR" \
    --sp-model "$SP_MODEL" \
    --split test \
    --beam 4 \
    --length-penalty 0.6 \
    --limit 500
done
