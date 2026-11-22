#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

A="${1:-}"
B="${2:-}"
if [ -z "$A" ] || [ -z "$B" ]; then
  echo "usage: bash scripts/run_pair.sh <model-a> <model-b>"
  echo "example: bash scripts/run_pair.sh transformer_improved lstm"
  exit 1
fi
if [ "$A" = "$B" ]; then
  echo "model-a and model-b must be different"
  exit 1
fi
for name in "$A" "$B"; do
  if [ ! -f "configs/${name}.yaml" ]; then
    echo "config not found: configs/${name}.yaml"
    exit 1
  fi
done

DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/hermionegrangerislet/translationdataset/data}"
SP_MODEL="${SP_MODEL:-/kaggle/working/prepared/spm_zh_en.model}"
RUNS_DIR="${RUNS_DIR:-runs}"
EXPORT_DIR="${EXPORT_DIR:-exports}"
EPOCHS="${EPOCHS:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RESUME_A="${RESUME_A:-0}"
SAVE="${SAVE:-1}"

mkdir -p "$RUNS_DIR"

if [ ! -f "$SP_MODEL" ]; then
  echo "tokenizer not found: $SP_MODEL"
  echo "run: bash scripts/prepare.sh"
  exit 1
fi

echo "GPU0: $A"
echo "GPU1: $B"
echo "logs: $RUNS_DIR/$A.console.log, $RUNS_DIR/$B.console.log"

if [ "$RESUME_A" = "1" ] && [ -f "$RUNS_DIR/$A/last.pt" ]; then
  CUDA_VISIBLE_DEVICES=0 python train.py \
    --config "configs/${A}.yaml" \
    --data-dir "$DATA_DIR" \
    --sp-model "$SP_MODEL" \
    --output-dir "$RUNS_DIR" \
    --run-name "$A" \
    --epochs "$EPOCHS" \
    --resume \
    $EXTRA_ARGS > "$RUNS_DIR/$A.console.log" 2>&1 &
else
  CUDA_VISIBLE_DEVICES=0 bash scripts/train_one.sh "$A" \
    > "$RUNS_DIR/$A.console.log" 2>&1 &
fi
P0=$!

CUDA_VISIBLE_DEVICES=1 bash scripts/train_one.sh "$B" \
  > "$RUNS_DIR/$B.console.log" 2>&1 &
P1=$!

S0=0
S1=0
wait "$P0" || S0=$?
wait "$P1" || S1=$?

echo "exit: $A=$S0, $B=$S1"

if [ "$SAVE" = "1" ]; then
  if [ -f "$RUNS_DIR/$A/best.pt" ] || [ -f "$RUNS_DIR/$A/last.pt" ]; then
    RUNS_DIR="$RUNS_DIR" EXPORT_DIR="$EXPORT_DIR" SP_MODEL="$SP_MODEL" \
      bash scripts/save_model.sh "$A" || true
  fi
  if [ -f "$RUNS_DIR/$B/best.pt" ] || [ -f "$RUNS_DIR/$B/last.pt" ]; then
    RUNS_DIR="$RUNS_DIR" EXPORT_DIR="$EXPORT_DIR" SP_MODEL="$SP_MODEL" \
      bash scripts/save_model.sh "$B" || true
  fi
fi

if [ "$S0" -ne 0 ] || [ "$S1" -ne 0 ]; then
  exit 1
fi
