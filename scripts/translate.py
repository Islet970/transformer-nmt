#!/usr/bin/env python
"""Translate Chinese sentences with a trained checkpoint.

Examples
--------
    # 单句（默认 greedy，CPU）
    python scripts/translate.py --checkpoint runs/transformer/best.pt --text "今天天气很好。"

    # 多句 + beam search + GPU（训练跑完后再用 cuda）
    python scripts/translate.py --checkpoint runs/transformer/best.pt \
        --text "你好。" --text "我喜欢学习机器翻译。" --beam 4 --device cuda

    # 从文件读（一行一句）
    python scripts/translate.py --checkpoint runs/lstm/best.pt --file my_sentences.txt

    # 交互模式：输入中文回车即出英文，输入 q 退出
    python scripts/translate.py --checkpoint runs/transformer/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from data.tokenizer import load_tokenizer  # noqa: E402
from evaluate import beam_decode_single, greedy_decode_batch  # noqa: E402
from models.factory import build_model  # noqa: E402
from utils import get_device, make_pad_mask  # noqa: E402


def load_model(checkpoint: str, device: torch.device, sp_model: str):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise KeyError(f"{checkpoint} has no 'config' field")
    cfg = Config.from_dict(ckpt["config"])
    cfg.sp_model = sp_model or cfg.sp_model
    tokenizer = load_tokenizer(cfg.sp_model)
    model = build_model(cfg, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, tokenizer


@torch.no_grad()
def translate(model, cfg: Config, tokenizer, text: str, device: torch.device, beam: int, length_penalty: float) -> str:
    ids = tokenizer.encode(text.strip(), add_bos=True, add_eos=True, max_len=cfg.max_len)
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = make_pad_mask(src).to(device)
    if beam <= 1:
        out_ids = greedy_decode_batch(model, src, src_mask, tokenizer, cfg.max_len)[0]
    else:
        out_ids = beam_decode_single(
            model, src, src_mask, tokenizer, cfg.max_len, beam_size=beam, length_penalty=length_penalty
        )
    return tokenizer.decode(out_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="runs/<name>/best.pt")
    parser.add_argument("--sp-model", default="prepared/spm_zh_en.model")
    parser.add_argument("--device", default="cpu", help="cpu (safe while training) or cuda")
    parser.add_argument("--beam", type=int, default=1, help="1 = greedy, 4 = beam search")
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--text", action="append", default=[], help="Chinese sentence (repeatable)")
    parser.add_argument("--file", default="", help="file with one Chinese sentence per line")
    parser.add_argument("--quiet", action="store_true", help="only print the translation")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg, tokenizer = load_model(args.checkpoint, device, args.sp_model)
    if not args.quiet:
        print(f"checkpoint: {args.checkpoint}")
        print(f"model: {cfg.model} | device: {device} | beam: {args.beam} | vocab: {tokenizer.vocab_size}")
        print("-" * 60)

    sentences: list[str] = list(args.text)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            sentences += [line.strip() for line in fh if line.strip()]

    if sentences:
        for text in sentences:
            start = time.time()
            out = translate(model, cfg, tokenizer, text, device, args.beam, args.length_penalty)
            if args.quiet:
                print(out)
            else:
                print(f"中文: {text}")
                print(f"英文: {out}   ({time.time() - start:.2f}s)")
        return 0

    print("交互模式：输入中文回车翻译，输入 q 退出。")
    while True:
        try:
            text = input("中文> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            break
        print(f"英文: {translate(model, cfg, tokenizer, text, device, args.beam, args.length_penalty)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
