from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nheads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % nheads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nheads={nheads}")
        self.d_model = d_model
        self.nheads = nheads
        self.dk = d_model // nheads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.store_attn = False
        self.last_attn: Optional[torch.Tensor] = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, len_q, _ = q.shape
        len_k = k.size(1)

        q = self.w_q(q).view(batch, len_q, self.nheads, self.dk).transpose(1, 2)
        k = self.w_k(k).view(batch, len_k, self.nheads, self.dk).transpose(1, 2)
        v = self.w_v(v).view(batch, len_k, self.nheads, self.dk).transpose(1, 2)

        x, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        if self.store_attn:
            self.last_attn = attn.detach()

        x = x.transpose(1, 2).contiguous().view(batch, len_q, self.d_model)
        return self.w_o(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nheads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nheads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nheads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nheads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nlayers: int,
        nheads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        use_pe: bool = True,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(d_model, max_len, dropout) if use_pe else None
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, nheads, d_ff, dropout) for _ in range(nlayers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(src) * math.sqrt(self.d_model)
        if self.pe is not None:
            x = self.pe(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nlayers: int,
        nheads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        use_pe: bool = True,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(d_model, max_len, dropout) if use_pe else None
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nheads, d_ff, dropout) for _ in range(nlayers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(tgt) * math.sqrt(self.d_model)
        if self.pe is not None:
            x = self.pe(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nlayers: int = 4,
        nheads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        use_pe: bool = True,
        tie_weights: bool = False,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(
            vocab_size, d_model, nlayers, nheads, d_ff, dropout, max_len, use_pe, pad_id
        )
        self.decoder = Decoder(
            vocab_size, d_model, nlayers, nheads, d_ff, dropout, max_len, use_pe, pad_id
        )
        self.generator = nn.Linear(d_model, vocab_size)
        if tie_weights:
            self.generator.weight = self.decoder.embed.weight

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(src, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(self.decoder(tgt, memory, tgt_mask, src_mask))

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, tgt, tgt_mask, src_mask)

    def enable_attention_store(self, enabled: bool = True) -> None:
        for layer in self.decoder.layers:
            layer.cross_attn.store_attn = enabled

    def get_last_cross_attention(self) -> Optional[torch.Tensor]:
        return self.decoder.layers[-1].cross_attn.last_attn


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
