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

# train_one.sh / save_model.sh 都通过环境变量读取这些路径。
export DATA_DIR SP_MODEL RUNS_DIR EXPORT_DIR EPOCHS EXTRA_ARGS SAVE

run_and_save() {
  local gpu="$1"
  local name="$2"
  local resume="$3"

  if [ "$resume" = "1" ] && [ -f "$RUNS_DIR/$name/last.pt" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" python train.py \
      --config "configs/${name}.yaml" \
      --data-dir "$DATA_DIR" \
      --sp-model "$SP_MODEL" \
      --output-dir "$RUNS_DIR" \
      --run-name "$name" \
      --epochs "$EPOCHS" \
      --resume \
      $EXTRA_ARGS
  else
    CUDA_VISIBLE_DEVICES="$gpu" bash scripts/train_one.sh "$name"
  fi
  local rc=$?

  # 这个模型一结束就立即导出，不等另一个模型。
  if [ "$SAVE" = "1" ] && { [ -f "$RUNS_DIR/$name/best.pt" ] || [ -f "$RUNS_DIR/$name/last.pt" ]; }; then
    bash scripts/save_model.sh "$name" || true
  fi
  return "$rc"
}

echo "GPU0: $A (resume=$RESUME_A)"
echo "GPU1: $B"
echo "logs: $RUNS_DIR/$A.console.log, $RUNS_DIR/$B.console.log"

run_and_save 0 "$A" "$RESUME_A" > "$RUNS_DIR/$A.console.log" 2>&1 &
P0=$!

run_and_save 1 "$B" 0 > "$RUNS_DIR/$B.console.log" 2>&1 &
P1=$!

S0=0
S1=0
wait "$P0" || S0=$?
wait "$P1" || S1=$?

echo "exit: $A=$S0, $B=$S1"
echo "exports: $EXPORT_DIR/$A.zip, $EXPORT_DIR/$B.zip"

if [ "$S0" -ne 0 ] || [ "$S1" -ne 0 ]; then
  exit 1
fi
