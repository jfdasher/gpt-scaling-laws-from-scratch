import math

import numpy as np
import pytest
import torch

from gptlab import data


def _write(path, arr):
    np.asarray(arr, dtype=np.uint16).tofile(path)


def test_token_stream_slices(tmp_path):
    _write(tmp_path / "t.bin", np.arange(1000))
    s = data.TokenStream(tmp_path / "t.bin", seq_len=4, micro_batch=2, offset=10)
    x, y = s.get(0)
    assert x.tolist() == [[10, 11, 12, 13], [14, 15, 16, 17]]
    assert y.tolist() == [[11, 12, 13, 14], [15, 16, 17, 18]]
    assert s.get(1)[0][0, 0].item() == 18
    assert x.dtype == torch.int64
    with pytest.raises(data.DataExhausted):
        s.get(200)


def test_step_slice_independent_of_micro_batch(tmp_path):
    _write(tmp_path / "t.bin", np.arange(1000))
    a = data.TokenStream(tmp_path / "t.bin", seq_len=4, micro_batch=2)
    b = data.TokenStream(tmp_path / "t.bin", seq_len=4, micro_batch=4)
    step = 3  # batch of 16 tokens: accum 2 for a, accum 1 for b
    xa = torch.cat([a.get(step * 2 + m)[0] for m in range(2)])
    xb = b.get(step)[0]
    assert torch.equal(xa, xb)


def test_val_batches_cover_without_overlap(tmp_path):
    _write(tmp_path / "v.bin", np.arange(103))
    xs, ys = [], []
    for x, y in data.val_batches(tmp_path / "v.bin", seq_len=5, micro_batch=4):
        xs.append(x.reshape(-1))
        ys.append(y.reshape(-1))
    x_all = torch.cat(xs)
    assert x_all.tolist() == list(range(100))
    assert torch.cat(ys).tolist() == list(range(1, 101))
    limited = list(data.val_batches(tmp_path / "v.bin", seq_len=5, micro_batch=4, max_tokens=40))
    assert sum(x.numel() for x, _ in limited) == 40


def test_manifest_roundtrip_and_mismatch(tmp_path):
    _write(tmp_path / data.VAL_BIN, np.arange(50))
    (tmp_path / data.TOKENIZER_JSON).write_text("{}")
    m = data.build_manifest(tmp_path, n_docs=3, n_tokens=50, n_bytes=200, extra={"note": "x"})
    data.write_manifest(tmp_path, m)
    assert data.verify_manifest(tmp_path)["n_tokens"] == 50
    _write(tmp_path / data.VAL_BIN, np.arange(51))
    with pytest.raises(data.ManifestMismatch):
        data.verify_manifest(tmp_path)


def test_bits_per_byte():
    assert math.isclose(data.bits_per_byte(2.0, 100, 400), 2.0 / math.log(2) * 0.25)
