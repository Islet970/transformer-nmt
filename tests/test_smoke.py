from __future__ import annotations

import os
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import TranslationDataset, collate_fn
from evaluate import greedy_decode_batch
from models.factory import build_model
from utils import causal_mask, make_pad_mask


class DummyTokenizer:
    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3

    def __init__(self, vocab_size: int = 50):
        self.vocab_size = vocab_size

    def encode(self, text, add_bos=False, add_eos=False, max_len=None):
        ids = [4 + (ord(c) % 40) for c in text if not c.isspace()]
        if max_len is not None:
            reserve = int(add_bos) + int(add_eos)
            ids = ids[: max(0, max_len - reserve)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return " ".join(str(int(x)) for x in ids if int(x) not in (0, 2, 3))


class SmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = self.tmp.name
        for split in ["train", "valid", "test"]:
            with open(os.path.join(self.data_dir, f"{split}.zh"), "w", encoding="utf-8") as f:
                f.write("我 爱 深度 学习\n今天 天气 很 好\n")
            with open(os.path.join(self.data_dir, f"{split}.en"), "w", encoding="utf-8") as f:
                f.write("I love deep learning\nIt is nice today\n")

        self.tok = DummyTokenizer(vocab_size=50)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dataset_and_collate(self):
        ds = TranslationDataset(
            os.path.join(self.data_dir, "train.zh"),
            os.path.join(self.data_dir, "train.en"),
            self.tok,
            max_len=32,
        )
        self.assertEqual(len(ds), 2)
        src, tgt = ds[0]
        self.assertEqual(src[0].item(), self.tok.bos_id)
        self.assertEqual(src[-1].item(), self.tok.eos_id)

        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        src_batch, tgt_batch = next(iter(loader))
        self.assertEqual(src_batch.dim(), 2)
        self.assertEqual(tgt_batch.dim(), 2)

    def _run_model(self, model_name: str):
        cfg = Config(
            data_dir=self.data_dir,
            model=model_name,
            d_model=32,
            nlayers=2,
            nheads=4,
            d_ff=64,
            dropout=0.1,
            max_len=32,
            tie_weights=(model_name == "transformer"),
        )
        model = build_model(cfg, self.tok.vocab_size)
        src = torch.randint(4, self.tok.vocab_size, (2, 8))
        tgt = torch.randint(4, self.tok.vocab_size, (2, 7))
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_mask = make_pad_mask(src)
        tgt_mask = make_pad_mask(tgt_in) & causal_mask(tgt_in.size(1), torch.device("cpu"))

        logits = model(src, tgt_in, src_mask, tgt_mask)
        self.assertEqual(logits.shape, (2, 6, self.tok.vocab_size))

        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(
            logits.reshape(-1, self.tok.vocab_size), tgt_out.reshape(-1)
        )
        loss.backward()
        if model_name == "transformer":
            grad = model.encoder.layers[0].self_attn.w_q.weight.grad
        else:
            grad = model.encoder.lstm.weight_ih_l0.grad
        self.assertIsNotNone(grad)
        return model

    def test_transformer_forward_backward(self):
        model = self._run_model("transformer")
        model.eval()
        src = torch.randint(4, self.tok.vocab_size, (2, 8))
        src_mask = make_pad_mask(src)
        preds = greedy_decode_batch(model, src, src_mask, self.tok, max_len=6)
        self.assertEqual(len(preds), 2)
        self.assertTrue(all(isinstance(x, list) for x in preds))

    def test_lstm_forward_backward(self):
        self._run_model("lstm")

    def test_transformer_variants(self):
        for use_pe, nheads, tie in [(False, 8, True), (True, 1, True), (True, 8, False)]:
            cfg = Config(
                data_dir=self.data_dir,
                model="transformer",
                d_model=32,
                nlayers=2,
                nheads=nheads,
                d_ff=64,
                max_len=32,
                use_pe=use_pe,
                tie_weights=tie,
            )
            model = build_model(cfg, self.tok.vocab_size)
            src = torch.randint(4, self.tok.vocab_size, (2, 8))
            tgt = torch.randint(4, self.tok.vocab_size, (2, 7))
            src_mask = make_pad_mask(src)
            tgt_mask = make_pad_mask(tgt[:, :-1]) & causal_mask(6, torch.device("cpu"))
            logits = model(src, tgt[:, :-1], src_mask, tgt_mask)
            self.assertEqual(logits.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
