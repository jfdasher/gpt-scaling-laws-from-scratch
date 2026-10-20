import math

import numpy as np

from gptlab import baselines as bl
from gptlab import data
from gptlab import tokenizer as tk


def test_cyclic_sequence_ordering():
    V = 16
    toks = np.tile(np.arange(V, dtype=np.uint16), 500)
    uni = bl.unigram_probs(toks, V)
    counts = bl.bigram_counts(toks, V)
    u = bl.unigram_ce(uni, toks)
    b = bl.interpolated_bigram_ce(counts, uni, toks, lam=0.9)
    assert b < u <= bl.uniform_loss(V) + 1e-6


def test_bigram_counts_chunking():
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 7, size=1001).astype(np.uint16)
    naive = np.zeros((7, 7))
    for a, b in zip(toks[:-1], toks[1:]):
        naive[a, b] += 1
    assert np.array_equal(bl.bigram_counts(toks, 7, chunk=100), naive)


def test_tune_lambda_picks_minimum():
    V = 16
    toks = np.tile(np.arange(V, dtype=np.uint16), 200)
    uni = bl.unigram_probs(toks, V)
    counts = bl.bigram_counts(toks, V)
    lam = bl.tune_lambda(counts, uni, toks)
    assert math.isclose(lam, 0.95)


def test_decode_docs_roundtrip():
    tok = tk.train_tokenizer(iter(["hello world", "second doc here"] * 50), vocab_size=280)
    eot = tok.token_to_id(tk.EOT)
    ids = tok.encode("hello world").ids + [eot] + tok.encode("second doc here").ids + [eot]
    assert bl.decode_docs(np.array(ids, np.uint16), tok) == ["hello world", "second doc here"]


def test_compute_baselines_without_gpt2(tmp_path):
    tok = tk.train_tokenizer(iter(["alpha beta gamma delta"] * 100), vocab_size=270)
    tok.save(str(tmp_path / data.TOKENIZER_JSON))
    eot = tok.token_to_id(tk.EOT)
    doc = tok.encode("alpha beta gamma delta").ids + [eot]
    np.array(doc * 3000, np.uint16).tofile(tmp_path / data.TRAIN_BIN)
    np.array(doc * 50, np.uint16).tofile(tmp_path / data.VAL_BIN)
    data.write_manifest(tmp_path, data.build_manifest(tmp_path, 50, len(doc) * 50, 22 * 50))
    out = bl.compute_baselines(tmp_path, device="cpu", n_count=5000, n_heldout=1000, include_gpt2=False, vocab_size=270)
    assert set(out) >= {"uniform", "unigram", "bigram"}
    assert out["bigram"]["loss_nats"] < out["unigram"]["loss_nats"] < out["uniform"]["loss_nats"]
    assert "bpb" in out["bigram"]
