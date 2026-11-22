from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

from config import DEFAULT_DATA_DIR, validate_data_dir
from data.tokenizer import SentencePieceTokenizer, train_sentencepiece
from utils import read_lines, save_json


def length_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "max": int(arr.max()),
    }


def collect_stats(data_dir: str, limit: int = 0) -> Dict:
    files = {
        "train": ("train.zh", "train.en"),
        "valid": ("valid.zh", "valid.en"),
        "test": ("test.zh", "test.en"),
    }
    stats: Dict[str, Dict] = {}

    for split, (zh_name, en_name) in files.items():
        zh = read_lines(os.path.join(data_dir, zh_name))
        en = read_lines(os.path.join(data_dir, en_name))
        if len(zh) != len(en):
            raise ValueError(f"{split}: zh/en line count mismatch {len(zh)} vs {len(en)}")

        if limit and limit > 0:
            zh, en = zh[:limit], en[:limit]

        zh_len = [len(x) for x in zh]
        en_len = [len(x.split()) for x in en]
        pairs = list(zip(zh, en))
        unique_pairs = len(set(pairs))
        empty = sum((not a.strip()) or (not b.strip()) for a, b in pairs)
        identical = sum(a.strip() == b.strip() for a, b in pairs)

        stats[split] = {
            "pairs": len(pairs),
            "unique_pairs": unique_pairs,
            "duplicate_pairs": len(pairs) - unique_pairs,
            "empty_pairs": empty,
            "identical_pairs": identical,
            "zh_chars": length_stats(zh_len),
            "en_words": length_stats(en_len),
            "example": {"zh": zh[0] if zh else "", "en": en[0] if en else ""},
        }
        print(
            f"[{split}] pairs={len(pairs):,} unique={unique_pairs:,} "
            f"empty={empty} zh_p95={stats[split]['zh_chars'].get('p95')} "
            f"en_p95={stats[split]['en_words'].get('p95')}"
        )

    return stats


def unk_rate(tokenizer: SentencePieceTokenizer, path: str, limit: int = 2000) -> float:
    total = unk = 0
    for i, line in enumerate(open(path, "r", encoding="utf-8")):
        if limit and i >= limit:
            break
        ids = tokenizer.encode(line.strip())
        total += len(ids)
        unk += sum(1 for x in ids if x == tokenizer.unk_id)
    return unk / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default="/kaggle/working/prepared")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--max-lines", type=int, default=0, help="0 means use all lines")
    parser.add_argument("--skip-spm", action="store_true")
    args = parser.parse_args()

    validate_data_dir(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    stats = collect_stats(args.data_dir, limit=args.max_lines)
    stats["data_dir"] = args.data_dir
    stats["vocab_size"] = args.vocab_size
    save_json(os.path.join(args.out_dir, "data_stats.json"), stats)
    print(f"Saved stats to {args.out_dir}/data_stats.json")

    if args.skip_spm:
        return

    model_prefix = os.path.join(args.out_dir, "spm_zh_en")
    train_files = [
        os.path.join(args.data_dir, "train.zh"),
        os.path.join(args.data_dir, "train.en"),
    ]
    print("Training SentencePiece...")
    model_path = train_sentencepiece(
        input_files=train_files,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )
    tokenizer = SentencePieceTokenizer(model_path)
    print(f"SentencePiece model: {model_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    stats["sp_model"] = model_path
    stats["sp_vocab_size"] = tokenizer.vocab_size
    stats["unk_rate_valid_zh"] = round(
        unk_rate(tokenizer, os.path.join(args.data_dir, "valid.zh")), 6
    )
    stats["unk_rate_valid_en"] = round(
        unk_rate(tokenizer, os.path.join(args.data_dir, "valid.en")), 6
    )
    save_json(os.path.join(args.out_dir, "data_stats.json"), stats)
    print(
        f"valid unk rate: zh={stats['unk_rate_valid_zh']:.4%}, "
        f"en={stats['unk_rate_valid_en']:.4%}"
    )


if __name__ == "__main__":
    main()
