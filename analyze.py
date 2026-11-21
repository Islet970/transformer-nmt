from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sacrebleu

from utils import load_json, save_json


def read_loss_csv(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
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


def discover_runs(runs_dir: str) -> List[str]:
    if not os.path.isdir(runs_dir):
        return []
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "loss.csv")):
            runs.append(path)
    return runs


def plot_loss(runs: List[str], out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    for run in runs:
        rows = read_loss_csv(os.path.join(run, "loss.csv"))
        if not rows:
            continue
        plt.plot(
            [r["epoch"] for r in rows],
            [r["train_loss"] for r in rows],
            marker="o",
            label=os.path.basename(run),
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training loss")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bleu(runs: List[str], out_path: str) -> None:
    names, values = [], []
    for run in runs:
        rows = read_loss_csv(os.path.join(run, "loss.csv"))
        if not rows:
            continue
        names.append(os.path.basename(run))
        values.append(max(r["val_bleu"] for r in rows))

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    plt.ylabel("Best validation BLEU")
    plt.title("Validation BLEU by run")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_length_buckets(runs: List[str], out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    any_data = False
    for run in runs:
        path = os.path.join(run, "test_beam1.json")
        if not os.path.exists(path):
            continue
        data = load_json(path)
        buckets = data.get("length_buckets", {})
        if not buckets:
            continue
        names = list(buckets.keys())
        values = [buckets[name]["bleu"] for name in names]
        plt.plot(names, values, marker="o", label=os.path.basename(run))
        any_data = True

    if not any_data:
        plt.close()
        return

    plt.xlabel("Source length bucket (tokens)")
    plt.ylabel("Test BLEU")
    plt.title("BLEU by source length")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_table(runs: List[str]) -> List[Dict]:
    table = []
    for run in runs:
        rows = read_loss_csv(os.path.join(run, "loss.csv"))
        if not rows:
            continue
        best = max(rows, key=lambda r: r["val_bleu"])
        summary_path = os.path.join(run, "summary.json")
        config = {}
        params = ""
        if os.path.exists(summary_path):
            summary = load_json(summary_path)
            config = summary.get("config", {})
            params = summary.get("params", "")

        test_bleu = test_chrf = beam4_bleu = ""
        test_path = os.path.join(run, "test_beam1.json")
        if os.path.exists(test_path):
            test_data = load_json(test_path)
            test_bleu = test_data.get("bleu", "")
            test_chrf = test_data.get("chrf", "")
        beam4_path = os.path.join(run, "test_beam4.json")
        if os.path.exists(beam4_path):
            beam4_bleu = load_json(beam4_path).get("bleu", "")

        table.append(
            {
                "run": os.path.basename(run),
                "model": config.get("model", ""),
                "params": params,
                "d_model": config.get("d_model", ""),
                "nlayers": config.get("nlayers", ""),
                "nheads": config.get("nheads", ""),
                "use_pe": config.get("use_pe", ""),
                "tie_weights": config.get("tie_weights", ""),
                "label_smoothing": config.get("label_smoothing", ""),
                "best_epoch": best["epoch"],
                "best_val_bleu": best["val_bleu"],
                "best_val_chrf": best["val_chrf"],
                "test_bleu": test_bleu,
                "test_chrf": test_chrf,
                "test_beam4_bleu": beam4_bleu,
                "seconds": round(best["seconds"], 1),
            }
        )
    return table


def parse_predictions(path: str) -> List[Dict[str, str]]:
    items = []
    current = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("###"):
                if current:
                    items.append(current)
                current = {}
            elif line.startswith("SRC: "):
                current["src"] = line[5:]
            elif line.startswith("REF: "):
                current["ref"] = line[5:]
            elif line.startswith("HYP: "):
                current["hyp"] = line[5:]
    if current:
        items.append(current)
    return items


def error_analysis(predictions_path: str, top_k: int = 10) -> List[Dict]:
    items = parse_predictions(predictions_path)
    scored = []
    for item in items:
        score = sacrebleu.sentence_bleu(item["hyp"], [item["ref"]], tokenize="13a").score
        scored.append({**item, "sentence_bleu": round(float(score), 2)})
    scored.sort(key=lambda x: x["sentence_bleu"])
    return scored[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = discover_runs(args.runs_dir)
    if not runs:
        print(f"No runs found in {args.runs_dir}")
        return

    plot_loss(runs, os.path.join(args.out_dir, "loss_curves.png"))
    plot_bleu(runs, os.path.join(args.out_dir, "val_bleu.png"))
    plot_length_buckets(runs, os.path.join(args.out_dir, "length_bleu.png"))

    table = build_table(runs)
    table_path = os.path.join(args.out_dir, "results_table.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    print(f"runs: {len(runs)}")
    for row in table:
        print(
            f"{row['run']:<28} BLEU={row['best_val_bleu']:.2f} "
            f"chrF={row['best_val_chrf']:.2f} epoch={row['best_epoch']}"
        )
    print(f"saved: {args.out_dir}/loss_curves.png")
    print(f"saved: {args.out_dir}/val_bleu.png")
    print(f"saved: {table_path}")

    if args.predictions:
        errors = error_analysis(args.predictions)
        save_json(os.path.join(args.out_dir, "error_cases.json"), errors)
        print(f"saved: {args.out_dir}/error_cases.json")


if __name__ == "__main__":
    main()
