"""Token files, deterministic batching, and the frozen-val manifest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
TOKENIZER_JSON = "tokenizer.json"
MANIFEST_JSON = "val_manifest.json"


class DataExhausted(RuntimeError):
    pass


class ManifestMismatch(RuntimeError):
    pass


def load_tokens(path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


class TokenStream:
    """Sequential reader: global micro-batch i covers tokens [offset + i*B*T, offset + (i+1)*B*T]."""

    def __init__(self, path, seq_len: int, micro_batch: int, offset: int = 0):
        self.tokens = load_tokens(path)
        self.seq_len, self.micro_batch, self.offset = seq_len, micro_batch, offset

    def span(self, index: int) -> tuple[int, int]:
        start = self.offset + index * self.micro_batch * self.seq_len
        return start, start + self.micro_batch * self.seq_len + 1

    def get(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.span(index)
        if end > len(self.tokens):
            raise DataExhausted(f"need token {end}, file has {len(self.tokens)}")
        buf = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        return buf[:-1].view(self.micro_batch, self.seq_len), buf[1:].view(self.micro_batch, self.seq_len)


def val_batches(path, seq_len: int, micro_batch: int, max_tokens: int | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Non-overlapping (x, y) batches from the start of the file; the last batch may have fewer rows."""
    tokens = load_tokens(path)
    limit = len(tokens) - 1 if max_tokens is None else min(len(tokens) - 1, max_tokens)
    start = 0
    while start + seq_len <= limit:
        rows = min(micro_batch, (limit - start) // seq_len)
        n = rows * seq_len
        buf = torch.from_numpy(tokens[start : start + n + 1].astype(np.int64))
        yield buf[:-1].view(rows, seq_len), buf[1:].view(rows, seq_len)
        start += n


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(data_dir, n_docs: int, n_tokens: int, n_bytes: int, extra: dict | None = None) -> dict:
    data_dir = Path(data_dir)
    return {
        "val_bin_sha256": sha256_file(data_dir / VAL_BIN),
        "tokenizer_sha256": sha256_file(data_dir / TOKENIZER_JSON),
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        **(extra or {}),
    }


def write_manifest(data_dir, manifest: dict) -> None:
    (Path(data_dir) / MANIFEST_JSON).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def verify_manifest(data_dir) -> dict:
    data_dir = Path(data_dir)
    m = json.loads((data_dir / MANIFEST_JSON).read_text())
    for key, name in (("val_bin_sha256", VAL_BIN), ("tokenizer_sha256", TOKENIZER_JSON)):
        actual = sha256_file(data_dir / name)
        if actual != m[key]:
            raise ManifestMismatch(f"{name}: manifest {m[key][:12]} != actual {actual[:12]}")
    return m


def bits_per_byte(loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    return loss_nats / math.log(2) * n_tokens / n_bytes
