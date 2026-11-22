#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NAME="${1:-}"
if [ -z "$NAME" ]; then
  echo "usage: bash scripts/save_model.sh <run-name>"
  echo "example: bash scripts/save_model.sh transformer_improved"
  exit 1
fi

RUNS_DIR="${RUNS_DIR:-runs}"
EXPORT_DIR="${EXPORT_DIR:-exports}"
SP_MODEL="${SP_MODEL:-/kaggle/working/prepared/spm_zh_en.model}"
SRC="$RUNS_DIR/$NAME"
DST="$EXPORT_DIR/$NAME"

if [ ! -d "$SRC" ]; then
  echo "run not found: $SRC"
  exit 1
fi

if [ -f "$SRC/best.pt" ]; then
  CKPT="$SRC/best.pt"
elif [ -f "$SRC/last.pt" ]; then
  CKPT="$SRC/last.pt"
else
  echo "no checkpoint found in $SRC"
  exit 1
fi

rm -rf "$DST"
mkdir -p "$DST"

# 轻量模型文件：只保留 model + config，约 100MB 左右，便于下载。
python - "$CKPT" "$DST/model.pt" <<'PY'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
torch.save({"model": ckpt["model"], "config": ckpt["config"]}, dst)
print("saved", dst)
PY

for f in config.json train.log loss.csv summary.json val_last.json; do
  if [ -f "$SRC/$f" ]; then cp "$SRC/$f" "$DST/"; fi
done
for f in "$SRC"/test_beam*.json "$SRC"/test_beam*_predictions.txt; do
  if [ -f "$f" ]; then cp "$f" "$DST/"; fi
done

if [ -f "$SP_MODEL" ]; then
  cp "$SP_MODEL" "$DST/spm_zh_en.model"
  if [ -f "${SP_MODEL%.model}.vocab" ]; then
    cp "${SP_MODEL%.model}.vocab" "$DST/spm_zh_en.vocab"
  fi
fi

python - "$EXPORT_DIR" "$NAME" <<'PY'
import os, shutil, sys
root, name = sys.argv[1], sys.argv[2]
out = os.path.join(root, name)
shutil.make_archive(out, "zip", root_dir=root, base_dir=name)
print("archive:", out + ".zip")
PY

echo "exported: $DST"
echo "zip: $EXPORT_DIR/$NAME.zip"
