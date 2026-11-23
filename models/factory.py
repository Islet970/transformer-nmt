from __future__ import annotations

import torch.nn as nn

from models.lstm import LSTMSeq2Seq, init_lstm_weights
from models.transformer import Transformer, init_weights


def build_model(cfg, vocab_size: int) -> nn.Module:
    if cfg.model == "transformer":
        model = Transformer(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            nlayers=cfg.nlayers,
            nheads=cfg.nheads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_len=max(512, cfg.max_len + 2),
            use_pe=cfg.use_pe,
            tie_weights=cfg.tie_weights,
            pad_id=0,
            norm_first=cfg.norm_first,
        )
        model.apply(init_weights)
        if cfg.tie_weights:
            # init_weights initializes generator and decoder separately; re-tie after init.
            model.generator.weight = model.decoder.embed.weight
        return model

    if cfg.model == "lstm":
        model = LSTMSeq2Seq(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            nlayers=cfg.nlayers,
            dropout=cfg.dropout,
            pad_id=0,
        )
        model.apply(init_lstm_weights)
        return model

    raise ValueError(f"Unknown model type: {cfg.model}")
