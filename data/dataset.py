from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from data.tokenizer import SentencePieceTokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        tokenizer: SentencePieceTokenizer,
        max_len: int = 128,
        limit: int = 0,
    ):
        with open(src_path, "r", encoding="utf-8") as f:
            self.src = [line.rstrip("\n") for line in f]
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt = [line.rstrip("\n") for line in f]

        if len(self.src) != len(self.tgt):
            raise ValueError(
                f"Line count mismatch: {src_path}={len(self.src)}, "
                f"{tgt_path}={len(self.tgt)}"
            )

        if limit and limit > 0:
            self.src = self.src[:limit]
            self.tgt = self.tgt[:limit]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids = self.tokenizer.encode(
            self.src[idx], add_bos=True, add_eos=True, max_len=self.max_len
        )
        tgt_ids = self.tokenizer.encode(
            self.tgt[idx], add_bos=True, add_eos=True, max_len=self.max_len
        )
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=0)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=0)
    return src, tgt


def build_dataloaders(cfg, tokenizer: SentencePieceTokenizer):
    data_dir = cfg.data_dir

    train_ds = TranslationDataset(
        f"{data_dir}/train.zh",
        f"{data_dir}/train.en",
        tokenizer,
        max_len=cfg.max_len,
        limit=cfg.limit_train,
    )
    val_ds = TranslationDataset(
        f"{data_dir}/valid.zh",
        f"{data_dir}/valid.en",
        tokenizer,
        max_len=cfg.max_len,
        limit=cfg.limit_val,
    )
    test_ds = TranslationDataset(
        f"{data_dir}/test.zh",
        f"{data_dir}/test.en",
        tokenizer,
        max_len=cfg.max_len,
        limit=cfg.limit_test,
    )

    pin_memory = torch.cuda.is_available()
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader
