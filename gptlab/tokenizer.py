"""Data prep: document-hash split, 8k byte-level BPE, encoding, shard assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from gptlab.flops import VOCAB_SIZE

EOT = "<|endoftext|>"
VAL_PER_MILLE = 5
ENCODE_BATCH = 1024

ValDoc = tuple[str, np.ndarray, int]  # (doc id, tokens ending in EOT, utf-8 bytes of the text)


class InsufficientData(RuntimeError):
    pass


def is_val_doc(doc_id: str) -> bool:
    return int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16) % 1000 < VAL_PER_MILLE


def shard_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


def train_tokenizer(texts: Iterator[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # GPT-2 regex pre-split
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def encode_file_docs(tok: Tokenizer, docs: Iterable[tuple[str, str]], seed: int) -> tuple[np.ndarray, list[ValDoc]]:
    """Encode (id, text) docs. Train docs are shuffled with `seed` and concatenated; val docs are returned sorted by id."""
    eot = tok.token_to_id(EOT)
    assert tok.get_vocab_size() <= 65536, "uint16 storage"
    train_docs: list[np.ndarray] = []
    val_docs: list[ValDoc] = []
    batch: list[tuple[str, str]] = []

    def flush() -> None:
        for (doc_id, text), enc in zip(batch, tok.encode_batch([t for _, t in batch])):
            arr = np.array(enc.ids + [eot], dtype=np.uint16)
            if is_val_doc(doc_id):
                val_docs.append((doc_id, arr, len(text.encode("utf-8"))))
            else:
                train_docs.append(arr)
        batch.clear()

    prev_special = tok.encode_special_tokens
    tok.encode_special_tokens = True  # a literal EOT string in text is plain bytes; only our appended EOT is special
    try:
        for doc in docs:
            batch.append(doc)
            if len(batch) >= ENCODE_BATCH:
                flush()
        flush()
    finally:
        tok.encode_special_tokens = prev_special
    order = np.random.default_rng(seed).permutation(len(train_docs))
    train = np.concatenate([train_docs[i] for i in order]) if train_docs else np.zeros(0, np.uint16)
    val_docs.sort(key=lambda v: v[0])
    return train, val_docs


def concat_shards(paths: list[Path], out_path: Path, seed: int, min_tokens: int) -> int:
    """Append whole shards in seeded order until at least min_tokens are written."""
    total = 0
    with open(out_path, "wb") as f:
        for i in np.random.default_rng(seed).permutation(len(paths)):
            arr = np.fromfile(paths[i], dtype=np.uint16)
            f.write(arr.tobytes())
            total += len(arr)
            if total >= min_tokens:
                return total
    raise InsufficientData(f"only {total} train tokens, need {min_tokens}; encode more files")


def write_val_shard(val_docs: list[ValDoc], bin_path: Path, index_path: Path) -> None:
    index, offset = [], 0
    for doc_id, arr, n_bytes in val_docs:
        index.append({"id": doc_id, "offset": offset, "n_tokens": int(len(arr)), "n_bytes": int(n_bytes)})
        offset += len(arr)
    (np.concatenate([a for _, a, _ in val_docs]) if val_docs else np.zeros(0, np.uint16)).tofile(bin_path)
    Path(index_path).write_text(json.dumps(index))


def merge_val_shards(shards: list[tuple[Path, Path]], out_path: Path, min_tokens: int) -> tuple[int, int, int]:
    """Take whole val docs in global id order until at least min_tokens; returns (docs, tokens, bytes)."""
    entries = []
    for bin_path, index_path in shards:
        tokens = np.fromfile(bin_path, dtype=np.uint16)
        for e in json.loads(Path(index_path).read_text()):
            entries.append((e["id"], tokens[e["offset"] : e["offset"] + e["n_tokens"]], e["n_bytes"]))
    entries.sort(key=lambda e: e[0])
    picked, n_tokens, n_bytes = [], 0, 0
    for _, arr, nb in entries:
        if n_tokens >= min_tokens:
            break
        picked.append(arr)
        n_tokens += len(arr)
        n_bytes += nb
    if n_tokens < min_tokens:
        raise InsufficientData(f"only {n_tokens} val tokens, need {min_tokens}; encode more files")
    np.concatenate(picked).tofile(out_path)
    return len(picked), n_tokens, n_bytes
