#!/usr/bin/env python
"""Validate a finished training run and archive its small artifacts.

Checks (all must pass):
  * summary.json / loss.csv / train.log / best.pt / last.pt exist
  * the run reached the expected number of epochs
  * no NaN/Inf in train loss or validation BLEU
  * best_bleu equals the best epoch and is above --min-bleu
  * validation BLEU does not collapse at the end and has no absurd jump
  * best.pt loads and carries model/config

On success the small files are copied to results/<run-name>/ and a check.json
record is written.  Checkpoints stay in runs/ (git-ignored).

Usage:
    python scripts/check_run.py --run-dir runs/transformer
    python scripts/check_run.py --run-dir runs/ablation_1head --min-bleu 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys


def read_loss(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = []
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_bleu": float(row["val_bleu"]),
                    "val_chrf": float(row["val_chrf"]),
                    "seconds": float(row["seconds"]),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--min-bleu", type=float, default=5.0)
    parser.add_argument("--expect-epochs", type=int, default=10)
    parser.add_argument("--max-final-drop", type=float, default=0.25,
                        help="allowed relative drop from best BLEU to the last epoch")
    parser.add_argument("--max-jump", type=float, default=10.0,
                        help="max BLEU gain between two adjacent epochs")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    name = os.path.basename(run_dir.rstrip("\\/"))
    problems: list[str] = []
    notes: list[str] = []

    def need(rel: str) -> str:
        path = os.path.join(run_dir, rel)
        if not os.path.exists(path):
            problems.append(f"missing {rel}")
        return path

    summary_path = need("summary.json")
    loss_path = need("loss.csv")
    log_path = need("train.log")
    best_path = need("best.pt")
    last_path = need("last.pt")
    val_path = os.path.join(run_dir, "val_last.json")
    if not os.path.exists(val_path):
        notes.append("val_last.json missing (best epoch validation record)")

    if problems:
        print("FAIL", problems)
        return 1

    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    with open(val_path, encoding="utf-8") as fh:
        val_last = json.load(fh)
    rows = read_loss(loss_path)

    if len(rows) != args.expect_epochs:
        problems.append(f"epochs recorded {len(rows)} != expected {args.expect_epochs}")

    bleus = [r["val_bleu"] for r in rows]
    losses = [r["train_loss"] for r in rows]
    if any(not math.isfinite(x) for x in bleus + losses):
        problems.append("NaN/Inf in loss.csv")

    best_bleu = max(bleus) if bleus else float("nan")
    best_epoch = bleus.index(best_bleu) if bleus else -1
    if abs(summary.get("best_bleu", -1) - best_bleu) > 1e-3:
        problems.append(f"summary best_bleu {summary.get('best_bleu')} != csv max {best_bleu:.4f}")
    if best_bleu < args.min_bleu:
        problems.append(f"best BLEU {best_bleu:.2f} < min {args.min_bleu}")

    if bleus:
        if bleus[-1] < best_bleu * (1 - args.max_final_drop):
            problems.append(
                f"final BLEU {bleus[-1]:.2f} collapsed from best {best_bleu:.2f}"
            )
        for i in range(1, len(bleus)):
            if bleus[i] - bleus[i - 1] > args.max_jump:
                problems.append(f"implausible BLEU jump {bleus[i-1]:.2f} -> {bleus[i]:.2f}")
        gains = sum(1 for i in range(1, len(bleus)) if bleus[i] > bleus[i - 1])
        notes.append(f"BLEU improved in {gains}/{len(bleus)-1} epoch transitions")

    if abs(len(rows) - summary.get("epochs_run", -1)) > 0:
        problems.append(f"summary epochs_run {summary.get('epochs_run')} != csv rows {len(rows)}")

    try:
        import torch

        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        if "model" not in ckpt or "config" not in ckpt:
            problems.append("best.pt missing model/config")
        params = sum(v.numel() for v in ckpt["model"].values())
        notes.append(f"best.pt loads, {params:,} params, epoch {ckpt.get('epoch')}")
        del ckpt
    except Exception as exc:  # noqa: BLE001
        problems.append(f"best.pt unreadable: {exc}")

    ok = not problems
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"  best_bleu {best_bleu:.2f} (epoch {best_epoch}) | final {bleus[-1]:.2f} | chrF {rows[-1]['val_chrf']:.2f}")
    print(f"  epochs {len(rows)} | sec/epoch {rows[-1]['seconds']:.0f}")
    for item in notes:
        print(f"  note: {item}")
    for item in problems:
        print(f"  PROBLEM: {item}")

    if ok:
        out = os.path.join(args.out_dir, name)
        os.makedirs(out, exist_ok=True)
        for src, dst in [
            (summary_path, "summary.json"),
            (loss_path, "loss.csv"),
            (log_path, "train.log"),
            (val_path, "val_last.json"),
        ]:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out, dst))
        record = {
            "run": name,
            "best_bleu": best_bleu,
            "best_epoch": best_epoch,
            "final_bleu": bleus[-1],
            "val_chrf": rows[-1]["val_chrf"],
            "seconds_per_epoch": rows[-1]["seconds"],
            "checkpoint": os.path.relpath(best_path, os.path.dirname(run_dir)),
            "notes": notes,
        }
        with open(os.path.join(out, "check.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        print(f"  archived to {out}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
