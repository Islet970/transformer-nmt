from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Iterable, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_pad_mask(seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """[B, L] -> [B, 1, 1, L] boolean mask. True means keep."""
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """[1, 1, size, size] boolean lower-triangular mask."""
    return torch.tril(
        torch.ones(size, size, dtype=torch.bool, device=device)
    ).unsqueeze(0).unsqueeze(0)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: str, lines: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).replace("\n", " ").rstrip() + "\n")


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


class Logger:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def repeat_memory(memory, beam_size: int):
    """Repeat encoder memory for beam search along batch dimension."""
    if isinstance(memory, torch.Tensor):
        return memory.repeat_interleave(beam_size, dim=0)
    if isinstance(memory, tuple):
        return tuple(repeat_memory(item, beam_size) for item in memory)
    if isinstance(memory, list):
        return [repeat_memory(item, beam_size) for item in memory]
    return memory
