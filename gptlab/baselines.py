"""Baselines measured before training: uniform, unigram, interpolated bigram, GPT-2-small reference."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from gptlab import data
from gptlab.flops import VOCAB_SIZE
from gptlab.tokenizer import EOT

LAMBDA_GRID = np.linspace(0.05, 0.95, 19)


def uniform_loss(vocab_size: int) -> float:
    return math.log(vocab_size)


def unigram_probs(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.bincount(tokens, minlength=vocab_size).astype(np.float64)
    return (counts + 1) / (counts.sum() + vocab_size)  # add-one smoothing


def unigram_ce(p: np.ndarray, tokens: np.ndarray) -> float:
    return float(-np.mean(np.log(p[tokens])))


def bigram_counts(tokens: np.ndarray, vocab_size: int, chunk: int = 50_000_000) -> np.ndarray:
    counts = np.zeros(vocab_size * vocab_size, dtype=np.float64)
    for start in range(0, len(tokens) - 1, chunk):
        seg = np.asarray(tokens[start : start + chunk + 1]).astype(np.int64)
        counts += np.bincount(seg[:-1] * vocab_size + seg[1:], minlength=vocab_size * vocab_size)
    return counts.reshape(vocab_size, vocab_size)


def interpolated_bigram_ce(counts: np.ndarray, uni_p: np.ndarray, tokens: np.ndarray, lam: float) -> float:
    prev = np.asarray(tokens[:-1]).astype(np.int64)
    nxt = np.asarray(tokens[1:]).astype(np.int64)
    row = counts.sum(axis=1)[prev]
    p_bi = np.divide(counts[prev, nxt], row, out=np.zeros(len(prev)), where=row > 0)
    return float(-np.mean(np.log(lam * p_bi + (1 - lam) * uni_p[nxt])))


def tune_lambda(counts: np.ndarray, uni_p: np.ndarray, heldout: np.ndarray) -> float:
    ces = [interpolated_bigram_ce(counts, uni_p, heldout, lam) for lam in LAMBDA_GRID]
    return float(LAMBDA_GRID[int(np.argmin(ces))])


def decode_docs(tokens: np.ndarray, tok) -> list[str]:
    eot = tok.token_to_id(EOT)
    ends = np.flatnonzero(np.asarray(tokens) == eot)
    docs, start = [], 0
    for end in ends:
        docs.append(tok.decode(np.asarray(tokens[start:end]).tolist()))
        start = end + 1
    return docs


def gpt2_bits_per_byte(texts: list[str], device: str, window: int = 1024, batch: int = 16) -> float:
    """External reference: GPT-2 small on our val text with its own tokenizer, in non-overlapping windows."""
    import torch
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    ids: list[int] = []
    for t in texts:
        ids.extend(tok.encode(t) + [tok.eos_token_id])
    arr = torch.tensor(ids)
    starts = list(range(0, len(arr) - 1, window))
    nll = 0.0
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            chunks = [arr[s : s + window + 1] for s in starts[i : i + batch]]
            for c in chunks:  # windows differ in length only at the tail; run each to keep it simple
                x, y = c[:-1].unsqueeze(0).to(device), c[1:].unsqueeze(0).to(device)
                logits = model(x).logits.float()
                nll += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    return nll / math.log(2) / n_bytes


def compute_baselines(
    data_dir,
    device: str,
    n_count: int = 200_000_000,
    n_heldout: int = 10_000_000,
    include_gpt2: bool = True,
    vocab_size: int = VOCAB_SIZE,
) -> dict:
    data_dir = Path(data_dir)
    manifest = data.verify_manifest(data_dir)
    train = data.load_tokens(data_dir / data.TRAIN_BIN)
    val = np.asarray(data.load_tokens(data_dir / data.VAL_BIN))
    counted = np.asarray(train[:n_count])
    heldout = np.asarray(train[n_count : n_count + n_heldout])
    uni = unigram_probs(counted, vocab_size)
    bi = bigram_counts(counted, vocab_size)
    lam = tune_lambda(bi, uni, heldout)
    losses = {
        "uniform": uniform_loss(vocab_size),
        "unigram": unigram_ce(uni, val),
        "bigram": interpolated_bigram_ce(bi, uni, val, lam),
    }
    out: dict = {
        name: {"loss_nats": loss, "bpb": data.bits_per_byte(loss, manifest["n_tokens"], manifest["n_bytes"])}
        for name, loss in losses.items()
    }
    out["bigram"]["lambda"] = lam
    out["counted_train_tokens"] = int(len(counted))
    out["val_bin_sha256"] = manifest["val_bin_sha256"]
    if include_gpt2:
        from tokenizers import Tokenizer

        texts = decode_docs(val, Tokenizer.from_file(str(data_dir / data.TOKENIZER_JSON)))
        out["gpt2_small"] = {"bpb": gpt2_bits_per_byte(texts, device), "note": "GPT-2 tokenizer; reference only"}
    return out
