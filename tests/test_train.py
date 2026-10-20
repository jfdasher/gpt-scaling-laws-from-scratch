import json
import math
from dataclasses import replace

import numpy as np
import pytest

from gptlab import data
from gptlab.train import (
    NonFiniteLoss,
    RunFailed,
    SimulatedPreemption,
    TrainConfig,
    run_id,
    train,
)

V = 64


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    np.tile(np.arange(V, dtype=np.uint16), 5000).tofile(d / data.TRAIN_BIN)
    np.tile(np.arange(V, dtype=np.uint16), 128).tofile(d / data.VAL_BIN)
    (d / data.TOKENIZER_JSON).write_text("{}")
    data.write_manifest(d, data.build_manifest(d, n_docs=1, n_tokens=V * 128, n_bytes=V * 128 * 4))
    return d


def cfg(**kw):
    base = TrainConfig(
        name="t", n_layer=2, d_model=64, total_steps=20, peak_lr=3e-3, batch_tokens=256, micro_batch=4,
        seq_len=32, vocab_size=V, compile=False, log_every=1, eval_every=10, eval_tokens=1024, ckpt_every_s=1e9,
    )
    return replace(base, **kw)


def losses(run_dir):
    rows = [json.loads(line) for line in (run_dir / "losses.jsonl").read_text().splitlines()]
    return {r["step"]: r["loss"] for r in rows}  # later duplicates win, as after a real preemption


def run(c, run_dir, data_dir, **kw):
    return train(c, run_dir, data_dir, device="cpu", sample_prompts=(), **kw)


def test_run_id_semantics():
    c = cfg()
    assert run_id(c) == run_id(replace(c, name="other", phase="grid", budget_flops=1.0, tokens_per_param=2.0))
    assert run_id(c) != run_id(replace(c, seed=1))


def test_resume_is_exact(tmp_path, data_dir):
    c = cfg()
    a = run(c, tmp_path / "a", data_dir)
    with pytest.raises(SimulatedPreemption):
        run(c, tmp_path / "b", data_dir, preempt_at_step=10)
    b = run(c, tmp_path / "b", data_dir, preempt_at_step=10)
    la, lb = losses(tmp_path / "a"), losses(tmp_path / "b")
    assert sorted(la) == sorted(lb) == list(range(20))
    assert [la[s] for s in range(20)] == [lb[s] for s in range(20)]
    assert a["val_loss"] == b["val_loss"]
    assert not (tmp_path / "b" / "ckpt.pt").exists()


def test_loss_decreases(tmp_path, data_dir):
    run(cfg(total_steps=80, peak_lr=1e-2), tmp_path / "r", data_dir)
    l = losses(tmp_path / "r")
    assert l[79] < l[0] - 1.0


def test_nonfinite_loss_fails_loudly(tmp_path, data_dir):
    c = cfg(peak_lr=math.inf, grad_clip=None)
    with pytest.raises(NonFiniteLoss):
        run(c, tmp_path / "n", data_dir)
    forensic = json.loads((tmp_path / "n" / "forensic.json").read_text())
    assert forensic["step"] == 1
    assert len(forensic["token_span"]) == 2
    assert (tmp_path / "n" / "FAILED").exists()
    with pytest.raises(RunFailed):
        run(c, tmp_path / "n", data_dir)


def test_existing_result_short_circuits(tmp_path, data_dir):
    first = run(cfg(), tmp_path / "s", data_dir)
    n_lines = len((tmp_path / "s" / "losses.jsonl").read_text().splitlines())
    again = run(cfg(), tmp_path / "s", data_dir)
    assert again == first
    assert len((tmp_path / "s" / "losses.jsonl").read_text().splitlines()) == n_lines


def test_manifest_checked(tmp_path, data_dir):
    np.arange(10, dtype=np.uint16).tofile(data_dir / data.VAL_BIN)
    with pytest.raises(data.ManifestMismatch):
        run(cfg(), tmp_path / "m", data_dir)


def test_result_fields(tmp_path, data_dir):
    r = run(cfg(), tmp_path / "f", data_dir, git_commit="abc123")
    for key in (
        "run_id", "name", "phase", "config", "n_layer", "d_model", "n_params", "n_params_nonembed", "tokens",
        "flops", "budget_flops", "nominal_tokens_per_param", "tokens_per_param", "val_loss", "val_bpb",
        "gpu_seconds", "gpu_hours", "cost_usd", "torch_version", "wandb_url", "git_commit",
    ):
        assert key in r
    assert r["tokens"] == 20 * 256
    assert r["flops"] == 6.0 * r["n_params"] * r["tokens"]
    assert r["git_commit"] == "abc123"
