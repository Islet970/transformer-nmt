from __future__ import annotations

import os
from typing import Iterable, List, Sequence

import sentencepiece as spm


class SentencePieceTokenizer:
    """Thin wrapper around SentencePiece with project special-token semantics."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if max_len is not None:
            reserve = int(add_bos) + int(add_eos)
            ids = ids[: max(0, max_len - reserve)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        if skip_special:
            special = {self.pad_id, self.bos_id, self.eos_id}
            ids = [int(i) for i in ids if int(i) not in special]
        return self.sp.decode([int(i) for i in ids])

    def pieces(self, text: str) -> List[str]:
        return self.sp.encode(text, out_type=str)

    def id_to_piece(self, idx: int) -> str:
        return self.sp.id_to_piece(int(idx))


def train_sentencepiece(
    input_files: Iterable[str],
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    num_threads: int = 4,
) -> str:
    """Train a shared BPE model from raw text files.

    Returns the path to the generated .model file.
    """
    input_files = list(input_files)
    for path in input_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training text not found: {path}")

    os.makedirs(os.path.dirname(os.path.abspath(model_prefix)), exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=int(vocab_size),
        model_type="bpe",
        character_coverage=float(character_coverage),
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        num_threads=int(num_threads),
        hard_vocab_limit=False,
        minloglevel=1,
    )

    model_path = f"{model_prefix}.model"
    if not os.path.exists(model_path):
        raise RuntimeError(f"SentencePiece training finished but model is missing: {model_path}")
    return model_path


def load_tokenizer(model_path: str) -> SentencePieceTokenizer:
    return SentencePieceTokenizer(model_path)
