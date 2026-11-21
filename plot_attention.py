from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch

from data.tokenizer import load_tokenizer
from evaluate import load_checkpoint_model
from models.transformer import Transformer
from utils import causal_mask, get_device, make_pad_mask


def find_cjk_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def read_line(path: str, index: int) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return line.rstrip("\n")
    raise IndexError(f"index {index} out of range for {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sp-model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="figures/attention.png")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    tokenizer = load_tokenizer(args.sp_model)
    model, cfg, _ = load_checkpoint_model(args.checkpoint, device, tokenizer)
    if not isinstance(model, Transformer):
        raise TypeError("Attention plot currently supports Transformer checkpoints only.")

    src_text = read_line(os.path.join(args.data_dir, f"{args.split}.zh"), args.index)
    ref_text = read_line(os.path.join(args.data_dir, f"{args.split}.en"), args.index)

    src_ids = tokenizer.encode(src_text, add_bos=True, add_eos=True, max_len=cfg.max_len)
    tgt_ids = tokenizer.encode(ref_text, add_bos=True, add_eos=True, max_len=cfg.max_len)
    tgt_in = tgt_ids[:-1]

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    tgt = torch.tensor([tgt_in], dtype=torch.long, device=device)
    src_mask = make_pad_mask(src).to(device)
    tgt_mask = make_pad_mask(tgt).to(device) & causal_mask(tgt.size(1), device)

    model.eval()
    model.enable_attention_store(True)
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        model.decode(memory, tgt, tgt_mask, src_mask)
    attn = model.get_last_cross_attention()
    if attn is None:
        raise RuntimeError("No attention captured.")
    attn = attn[0].mean(dim=0).float().cpu().numpy()

    src_labels = [tokenizer.id_to_piece(i) for i in src_ids]
    tgt_labels = [tokenizer.id_to_piece(i) for i in tgt_in]

    plt.rcParams["axes.unicode_minus"] = False
    font_path = find_cjk_font()
    font_prop = font_manager.FontProperties(fname=font_path) if font_path else None
    if font_path:
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=font_path).get_name()]

    fig, ax = plt.subplots(figsize=(max(6, len(src_labels) * 0.45), max(4, len(tgt_labels) * 0.35)))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_labels)))
    ax.set_xticklabels(
        src_labels, rotation=45, ha="right", fontsize=8, fontproperties=font_prop
    )
    ax.set_yticks(range(len(tgt_labels)))
    ax.set_yticklabels(tgt_labels, fontsize=8, fontproperties=font_prop)
    ax.set_xlabel("Source tokens", fontproperties=font_prop)
    ax.set_ylabel("Target tokens (reference)", fontproperties=font_prop)
    ax.set_title("Transformer cross-attention (last layer, head-average)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"saved: {args.out}")
    print("SRC:", src_text)
    print("REF:", ref_text)


if __name__ == "__main__":
    main()
