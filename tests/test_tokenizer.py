import numpy as np
import pytest

from gptlab import tokenizer as tk

CORPUS = [f"Document {i}: the quick brown fox jumps over {i % 7} lazy dogs." for i in range(400)] + [
    "naïve café — 東京 🚀",
    "def f(x):\n    return x  # code\n\n\n",
    "tabs\tand   spaces",
]


def _toy_tokenizer():
    return tk.train_tokenizer(iter(CORPUS), vocab_size=300)


def test_split_rule_rate_and_determinism():
    ids = [f"<urn:uuid:{i:08d}>" for i in range(200_000)]
    flags = [tk.is_val_doc(i) for i in ids]
    assert 0.004 < sum(flags) / len(ids) < 0.006
    assert flags == [tk.is_val_doc(i) for i in ids]


def test_tokenizer_eot_and_roundtrip():
    tok = _toy_tokenizer()
    assert tok.token_to_id(tk.EOT) == 0
    assert tok.get_vocab_size() <= 300
    for text in CORPUS[-3:]:
        assert tok.decode(tok.encode(text).ids) == text


def _docs(n):
    return [(f"id-{i}", f"text number {i} with some words") for i in range(n)]


def test_encode_file_docs_split_and_shuffle():
    tok = _toy_tokenizer()
    docs = _docs(3000)
    n_val = sum(tk.is_val_doc(i) for i, _ in docs)
    assert n_val > 0
    train, val = tk.encode_file_docs(tok, iter(docs), seed=1)
    assert train.dtype == np.uint16
    assert int((train == 0).sum()) == len(docs) - n_val
    assert [v[0] for v in val] == sorted(i for i, _ in docs if tk.is_val_doc(i))
    assert all(v[1][-1] == 0 for v in val)
    assert val[0][2] == len(dict(docs)[val[0][0]].encode("utf-8"))
    train2, _ = tk.encode_file_docs(tok, iter(docs), seed=1)
    train3, _ = tk.encode_file_docs(tok, iter(docs), seed=2)
    assert np.array_equal(train, train2)
    assert not np.array_equal(train, train3)


def test_concat_shards(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"s{i}.bin"
        np.full(10, i, dtype=np.uint16).tofile(p)
        paths.append(p)
    n = tk.concat_shards(paths, tmp_path / "train.bin", seed=0, min_tokens=25)
    assert n == 30
    out = np.fromfile(tmp_path / "train.bin", dtype=np.uint16)
    assert len(out) == 30 and len(set(out.tolist())) == 3
    with pytest.raises(tk.InsufficientData):
        tk.concat_shards(paths, tmp_path / "x.bin", seed=0, min_tokens=41)


def test_merge_val_shards(tmp_path):
    a = [("b", np.array([5, 6, 0], np.uint16), 7), ("d", np.array([7, 0], np.uint16), 3)]
    b = [("a", np.array([1, 0], np.uint16), 2), ("c", np.array([2, 3, 4, 0], np.uint16), 9)]
    shards = []
    for name, docs in (("x", a), ("y", b)):
        tk.write_val_shard(docs, tmp_path / f"val_{name}.bin", tmp_path / f"val_{name}.json")
        shards.append((tmp_path / f"val_{name}.bin", tmp_path / f"val_{name}.json"))
    n_docs, n_tokens, n_bytes = tk.merge_val_shards(shards, tmp_path / "val.bin", min_tokens=6)
    assert (n_docs, n_tokens, n_bytes) == (3, 9, 18)  # a, b, c in id order
    assert np.fromfile(tmp_path / "val.bin", dtype=np.uint16).tolist() == [1, 0, 5, 6, 0, 2, 3, 4, 0]
    with pytest.raises(tk.InsufficientData):
        tk.merge_val_shards(shards, tmp_path / "v2.bin", min_tokens=100)
