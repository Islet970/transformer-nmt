from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import torch
import sacrebleu
from torch.utils.data import DataLoader

from config import Config, apply_overrides, resolve_sp_model, validate_data_dir
from data.dataset import TranslationDataset, collate_fn
from data.tokenizer import SentencePieceTokenizer, load_tokenizer
from models.factory import build_model
from utils import (
    causal_mask,
    get_device,
    make_pad_mask,
    repeat_memory,
    save_json,
)


def ids_to_text(tokenizer: SentencePieceTokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode([int(x) for x in ids])


def _select_memory(memory, idx: torch.Tensor, batch: int):
    """Index encoder memory along the batch dimension.

    Works for the Transformer (a tensor) and for the LSTM (a tuple whose
    hidden/cell states are laid out as [layers, batch, hidden]).
    """
    if torch.is_tensor(memory):
        return memory.index_select(0, idx)
    if isinstance(memory, (tuple, list)):
        out = []
        for item in memory:
            if torch.is_tensor(item):
                dim = 0 if item.size(0) == batch else 1
                out.append(item.index_select(dim, idx))
            else:
                out.append(_select_memory(item, idx, batch))
        return tuple(out) if isinstance(memory, tuple) else out
    raise TypeError(f"unsupported memory type: {type(memory)}")


@torch.no_grad()
def greedy_decode_batch(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    tokenizer: SentencePieceTokenizer,
    max_len: int,
) -> List[List[int]]:
    """Batched greedy decoding with early stopping.

    Sentences that emit EOS are dropped from the batch, so the loop stops as
    soon as every sentence is finished instead of always running ``max_len``
    steps for the whole batch.  Outputs are identical to the naive version.
    """
    model.eval()
    device = src.device
    batch = src.size(0)
    memory = model.encode(src, src_mask)
    ys = torch.full((batch, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    alive = torch.arange(batch, device=device)
    finished: Dict[int, List[int]] = {}

    for _ in range(max_len):
        mem = _select_memory(memory, alive, batch)
        mask = src_mask.index_select(0, alive)
        tgt_mask = make_pad_mask(ys) & causal_mask(ys.size(1), device)
        logits = model.decode(mem, ys, tgt_mask, mask)
        next_id = logits[:, -1].argmax(dim=-1)
        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)

        eos = next_id == tokenizer.eos_id
        if eos.any():
            rows = eos.nonzero(as_tuple=False).squeeze(1)
            for row, sent_id in zip(rows.tolist(), alive[rows].tolist()):
                finished[sent_id] = ys[row].tolist()
            keep = ~eos
            alive = alive[keep]
            ys = ys[keep]
            if alive.numel() == 0:
                break

    for row, sent_id in enumerate(alive.tolist()):
        finished[sent_id] = ys[row].tolist()

    results = []
    for i in range(batch):
        ids = finished[i][1:]
        if tokenizer.eos_id in ids:
            ids = ids[: ids.index(tokenizer.eos_id)]
        results.append([x for x in ids if x not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)])
    return results


@torch.no_grad()
def beam_decode_single(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    tokenizer: SentencePieceTokenizer,
    max_len: int,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> List[int]:
    """Beam search for one sentence. Slow but simple and model-agnostic."""
    model.eval()
    device = src.device

    memory = model.encode(src, src_mask)
    beams: List[Tuple[List[int], float]] = [([tokenizer.bos_id], 0.0)]
    completed: List[Tuple[List[int], float]] = []

    for _ in range(max_len):
        if not beams:
            break

        active = [ids for ids, _ in beams]
        ys = torch.tensor(active, dtype=torch.long, device=device)
        tgt_mask = make_pad_mask(ys).to(device) & causal_mask(ys.size(1), device)
        mem = repeat_memory(memory, len(beams))
        mask = src_mask.repeat_interleave(len(beams), dim=0)
        logits = model.decode(mem, ys, tgt_mask, mask)
        log_probs = torch.log_softmax(logits[:, -1], dim=-1)

        candidates: List[Tuple[List[int], float]] = []
        for i, (ids, score) in enumerate(beams):
            top_scores, top_ids = log_probs[i].topk(beam_size)
            for token_score, token in zip(top_scores.tolist(), top_ids.tolist()):
                new_ids = ids + [token]
                new_score = score + token_score
                if token == tokenizer.eos_id:
                    completed.append((new_ids, new_score))
                else:
                    candidates.append((new_ids, new_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]

    pool = completed if completed else beams
    if not pool:
        return []

    def normalized(item):
        ids, score = item
        length = max(1, len(ids) - 1)
        penalty = ((5.0 + length) / 6.0) ** length_penalty
        return score / penalty

    best_ids, _ = max(pool, key=normalized)
    best_ids = best_ids[1:]
    if tokenizer.eos_id in best_ids:
        best_ids = best_ids[: best_ids.index(tokenizer.eos_id)]
    return [x for x in best_ids if x not in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)]


def compute_length_buckets(
    src_lengths: Sequence[int],
    preds: Sequence[str],
    refs: Sequence[str],
) -> Dict[str, float]:
    edges = [0, 10, 20, 30, 40, 10**9]
    names = ["1-10", "11-20", "21-30", "31-40", "40+"]
    buckets: Dict[str, List[int]] = {name: [] for name in names}
    for i, length in enumerate(src_lengths):
        for j in range(len(names)):
            if edges[j] < length <= edges[j + 1]:
                buckets[names[j]].append(i)
                break

    result = {}
    for name, indices in buckets.items():
        if not indices:
            continue
        p = [preds[i] for i in indices]
        r = [refs[i] for i in indices]
        result[name] = {
            "n": len(indices),
            "bleu": round(sacrebleu.corpus_bleu(p, [r], tokenize="13a").score, 4),
        }
    return result


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
    max_len: int = 128,
    beam: int = 1,
    length_penalty: float = 0.6,
    max_batches: int = 0,
) -> Dict:
    model.eval()
    preds: List[str] = []
    refs: List[str] = []
    sources: List[str] = []
    src_lengths: List[int] = []

    for batch_idx, (src, tgt) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        src = src.to(device)
        src_mask = make_pad_mask(src).to(device)

        if beam <= 1:
            pred_ids = greedy_decode_batch(model, src, src_mask, tokenizer, max_len)
        else:
            pred_ids = []
            for i in range(src.size(0)):
                pred_ids.append(
                    beam_decode_single(
                        model,
                        src[i : i + 1],
                        src_mask[i : i + 1],
                        tokenizer,
                        max_len=max_len,
                        beam_size=beam,
                        length_penalty=length_penalty,
                    )
                )

        for i in range(src.size(0)):
            preds.append(ids_to_text(tokenizer, pred_ids[i]))
            refs.append(ids_to_text(tokenizer, tgt[i].tolist()))
            sources.append(ids_to_text(tokenizer, src[i].tolist()))
            src_lengths.append(int((src[i] != tokenizer.pad_id).sum().item()))

    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="13a")
    chrf = sacrebleu.corpus_chrf(preds, [refs])

    return {
        "n": len(preds),
        "bleu": round(float(bleu.score), 4),
        "chrf": round(float(chrf.score), 4),
        "bleu_signature": str(bleu),
        "beam": beam,
        "length_penalty": length_penalty,
        "length_buckets": compute_length_buckets(src_lengths, preds, refs),
        "predictions": [
            {"src": s, "ref": r, "hyp": p} for s, r, p in zip(sources, refs, preds)
        ],
    }


def write_predictions(path: str, predictions: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, item in enumerate(predictions):
            f.write(f"### {i}\n")
            f.write(f"SRC: {item['src']}\n")
            f.write(f"REF: {item['ref']}\n")
            f.write(f"HYP: {item['hyp']}\n\n")


def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device,
    tokenizer: SentencePieceTokenizer,
    cfg: Config | None = None,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if cfg is None:
        if "config" not in ckpt:
            raise KeyError("Checkpoint has no 'config' field.")
        cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg, ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--sp-model", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.config:
        cfg = Config.from_yaml(args.config)
    elif "config" in ckpt:
        cfg = Config.from_dict(ckpt["config"])
    else:
        raise ValueError("No config in checkpoint; pass --config")

    cfg = apply_overrides(
        cfg,
        argparse.Namespace(
            data_dir=args.data_dir,
            sp_model=args.sp_model,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=args.device,
            beam=args.beam,
            length_penalty=args.length_penalty,
            limit_test=args.limit,
        ),
    )

    validate_data_dir(cfg.data_dir)
    sp_model = resolve_sp_model(cfg, args.sp_model)
    tokenizer = load_tokenizer(sp_model)
    # Keep the runtime overrides (--data-dir/--sp-model/--batch-size/...): the
    # checkpoint's stored config is only the fallback for values not passed here.
    model, cfg, ckpt = load_checkpoint_model(args.checkpoint, device, tokenizer, cfg=cfg)

    split = "valid" if args.split == "validation" else args.split
    ds = TranslationDataset(
        f"{cfg.data_dir}/{split}.zh",
        f"{cfg.data_dir}/{split}.en",
        tokenizer,
        max_len=cfg.max_len,
        limit=args.limit,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    result = evaluate_loader(
        model,
        loader,
        tokenizer,
        device,
        max_len=cfg.max_len,
        beam=args.beam,
        length_penalty=args.length_penalty,
        max_batches=args.max_batches,
    )
    result["checkpoint"] = args.checkpoint
    result["split"] = split
    result["sp_model"] = sp_model

    output = args.output or os.path.join(os.path.dirname(args.checkpoint), f"{split}_beam{args.beam}.json")
    save_json(output, {k: v for k, v in result.items() if k != "predictions"})
    write_predictions(os.path.splitext(output)[0] + "_predictions.txt", result["predictions"])
    print(
        f"split={split} n={result['n']} BLEU={result['bleu']:.2f} "
        f"chrF={result['chrf']:.2f} beam={args.beam}"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
