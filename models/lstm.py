from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int, enc_size: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(enc_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden: [B, H], enc_outputs: [B, S, 2H]
        scores = self.v(
            torch.tanh(self.W_h(hidden).unsqueeze(1) + self.W_s(enc_outputs))
        ).squeeze(-1)  # [B, S]

        keep = src_mask.squeeze(1).squeeze(1).bool()  # [B, S]
        scores = scores.masked_fill(~keep, -1e9)
        attn = torch.softmax(scores, dim=-1)  # [B, S]
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)  # [B, 2H]
        return context, attn


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nlayers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=nlayers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if nlayers > 1 else 0.0,
        )
        self.bridge_h = nn.Linear(2 * d_model, d_model)
        self.bridge_c = nn.Linear(2 * d_model, d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        lengths = src_mask.squeeze(1).squeeze(1).sum(dim=1).long().cpu()
        lengths = lengths.clamp(min=1)

        emb = self.embed(src)
        packed = pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed)
        enc_outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1)
        )

        # h: [2*num_layers, B, H]; last layer forward/backward
        h_last = torch.cat([h[-2], h[-1]], dim=-1)
        c_last = torch.cat([c[-2], c[-1]], dim=-1)
        h0 = torch.tanh(self.bridge_h(h_last))
        c0 = torch.tanh(self.bridge_c(c_last))
        return enc_outputs, (h0, c0)


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        enc_size: int,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.attention = BahdanauAttention(d_model, enc_size)
        self.cell = nn.LSTMCell(d_model + enc_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.generator = nn.Linear(d_model, vocab_size)
        self.store_attn = False
        self.last_attn: Optional[torch.Tensor] = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        enc_outputs, (h, c) = memory
        embedded = self.embed(tgt)

        logits = []
        for t in range(tgt.size(1)):
            context, attn = self.attention(h, enc_outputs, src_mask)
            if self.store_attn:
                self.last_attn = attn.detach()
            inp = torch.cat([embedded[:, t], context], dim=-1)
            h, c = self.cell(self.dropout(inp), (h, c))
            logits.append(self.generator(self.dropout(h)))

        return torch.stack(logits, dim=1)


class LSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nlayers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = LSTMEncoder(vocab_size, d_model, nlayers, dropout, pad_id)
        self.decoder = LSTMDecoder(vocab_size, d_model, 2 * d_model, dropout, pad_id)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(src, src_mask)

    def decode(
        self,
        memory,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(tgt, memory, src_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, tgt, tgt_mask, src_mask)

    def enable_attention_store(self, enabled: bool = True) -> None:
        self.decoder.store_attn = enabled

    def get_last_cross_attention(self) -> Optional[torch.Tensor]:
        return self.decoder.last_attn


def init_lstm_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
