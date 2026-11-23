from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_DATA_DIR = "/kaggle/input/datasets/hermionegrangerislet/translationdataset/data"
DEFAULT_SP_MODEL = "/kaggle/working/prepared/spm_zh_en.model"


@dataclass
class Config:
    # data
    data_dir: str = DEFAULT_DATA_DIR
    sp_model: str = DEFAULT_SP_MODEL
    max_len: int = 128
    batch_size: int = 32
    num_workers: int = 0
    limit_train: int = 0
    limit_val: int = 0
    limit_test: int = 0

    # model
    model: str = "transformer"          # transformer | lstm
    d_model: int = 256
    nlayers: int = 4
    nheads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    use_pe: bool = True
    tie_weights: bool = False
    label_smoothing: float = 0.0
    norm_first: bool = True            # pre-LN; post-LN (False) is unstable at depth 4

    # optimization
    epochs: int = 10
    lr: float = 1.0
    warmup: int = 4000
    clip: float = 1.0
    weight_decay: float = 0.0
    seed: int = 42
    amp: bool = True
    patience: int = 3

    # runtime
    output_dir: str = "runs"
    run_name: str = ""
    device: str = "auto"
    max_steps: int = 0
    log_every: int = 100
    val_max_batches: int = 0

    # decoding
    beam: int = 1
    length_penalty: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Config":
        allowed = {f.name for f in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configs. pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
        return cls.from_dict(values)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def validate_data_dir(data_dir: str) -> None:
    required = ["train.zh", "train.en", "valid.zh", "valid.en", "test.zh", "test.en"]
    missing = [name for name in required if not os.path.exists(os.path.join(data_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"Data directory is invalid: {data_dir}\n"
            f"Missing files: {missing}\n"
            "Expected: train.zh, train.en, valid.zh, valid.en, test.zh, test.en"
        )


def resolve_sp_model(config: Config, explicit: Optional[str] = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    if config.sp_model:
        candidates.append(config.sp_model)
    candidates.extend([
        os.path.join(config.data_dir, "spm_zh_en.model"),
        "/kaggle/working/prepared/spm_zh_en.model",
        "/kaggle/working/spm_zh_en.model",
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "SentencePiece model not found. Tried:\n  " + "\n  ".join(candidates) +
        "\nRun: python -m data.prepare --data-dir ... --out-dir /kaggle/working/prepared"
    )


def apply_overrides(config: Config, args: Any) -> Config:
    """Apply non-None argparse values to a Config."""
    for field in fields(Config):
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(config, field.name, value)
    return config
