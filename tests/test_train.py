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


def test_nonfinite_gradient_fails_before_step(tmp_path, data_dir, monkeypatch):
    from gptlab import train as T

    real = T.torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def fake(params, clip):
        calls["n"] += 1
        real(params, clip)
        return T.torch.tensor(float("nan")) if calls["n"] == 4 else T.torch.tensor(1.0)  # 4th call = step 3

    monkeypatch.setattr(T.torch.nn.utils, "clip_grad_norm_", fake)
    with pytest.raises(NonFiniteLoss, match="gradient"):
        run(cfg(), tmp_path / "g", data_dir)
    forensic = json.loads((tmp_path / "g" / "forensic.json").read_text())
    assert forensic["step"] == 3
    assert forensic["grad_norm"] == "nan"
    assert (tmp_path / "g" / "FAILED").exists()


class RecordingLogger:
    url = None

    def __init__(self):
        self.finished = 0
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((step, dict(metrics)))

    def log_samples(self, step, samples):
        pass

    def finish(self):
        self.finished += 1


def test_logger_finished_exactly_once_on_every_exit(tmp_path, data_dir):
    ok = RecordingLogger()
    run(cfg(), tmp_path / "ok", data_dir, logger=ok)
    assert ok.finished == 1 and ok.logged[-1][1].keys() >= {"final/val_loss"}  # final log precedes finish

    bad = RecordingLogger()
    with pytest.raises(NonFiniteLoss):
        run(cfg(peak_lr=math.inf, grad_clip=None), tmp_path / "nf", data_dir, logger=bad)
    assert bad.finished == 1

    pre = RecordingLogger()
    with pytest.raises(SimulatedPreemption):
        run(cfg(), tmp_path / "pre", data_dir, logger=pre, preempt_at_step=5)
    assert pre.finished == 1

    with pytest.raises(data.ManifestMismatch):
        np.arange(10, dtype=np.uint16).tofile(data_dir / data.VAL_BIN)
        mm = RecordingLogger()
        run(cfg(), tmp_path / "mm", data_dir, logger=mm)
    assert mm.finished == 1


def test_failure_text_names_the_cause(tmp_path, data_dir):
    with pytest.raises(NonFiniteLoss):
        run(cfg(peak_lr=math.inf, grad_clip=None), tmp_path / "l", data_dir)
    assert "loss" in (tmp_path / "l" / "FAILED").read_text()


def test_gradient_failure_text_names_gradient(tmp_path, data_dir, monkeypatch):
    from gptlab import train as T

    monkeypatch.setattr(T.torch.nn.utils, "clip_grad_norm_", lambda p, c: T.torch.tensor(float("inf")))
    with pytest.raises(NonFiniteLoss):
        run(cfg(), tmp_path / "g", data_dir)
    assert "gradient" in (tmp_path / "g" / "FAILED").read_text()


def diag_rows(run_dir):
    return [json.loads(line) for line in (run_dir / "diag.jsonl").read_text().splitlines()]


def test_diag_written_at_right_steps(tmp_path, data_dir):
    run(cfg(diag_every=5), tmp_path / "d", data_dir)
    rows = diag_rows(tmp_path / "d")
    assert [r["step"] for r in rows] == [0, 5, 10, 15]
    for r in rows:
        assert {"grad_norm/layer_0", "grad_norm/layer_1", "attn_max_logit/layer_0", "attn_max_logit/layer_1"} <= r.keys()
    assert not (tmp_path / "plain").exists()
    run(cfg(), tmp_path / "plain", data_dir)
    assert not (tmp_path / "plain" / "diag.jsonl").exists()


def test_diag_goes_to_logger(tmp_path, data_dir):
    lg = RecordingLogger()
    run(cfg(diag_every=10), tmp_path / "d", data_dir, logger=lg)
    steps = [s for s, m in lg.logged if "attn_max_logit/layer_0" in m and "grad_norm/layer_0" in m]
    assert 0 in steps and 10 in steps


def test_diag_does_not_change_training_or_run_id(tmp_path, data_dir):
    a = run(cfg(), tmp_path / "a", data_dir)
    b = run(cfg(diag_every=3), tmp_path / "b", data_dir)
    assert losses(tmp_path / "a") == losses(tmp_path / "b") and a["val_loss"] == b["val_loss"]
    assert run_id(cfg()) == run_id(cfg(diag_every=3))


def test_resume_is_exact_with_diag(tmp_path, data_dir):
    c = cfg(diag_every=4)
    a = run(c, tmp_path / "a", data_dir)
    with pytest.raises(SimulatedPreemption):
        run(c, tmp_path / "b", data_dir, preempt_at_step=10)
    b = run(c, tmp_path / "b", data_dir, preempt_at_step=10)
    la, lb = losses(tmp_path / "a"), losses(tmp_path / "b")
    assert [la[s] for s in range(20)] == [lb[s] for s in range(20)]
    assert a["val_loss"] == b["val_loss"]


def test_diag_final_row_on_nonfinite_paths(tmp_path, data_dir, monkeypatch):
    from gptlab import train as T

    # loss path: LR=inf poisons weights at step 0, so step 1 fails; grad norms recorded, attention skipped
    with pytest.raises(NonFiniteLoss):
        run(cfg(peak_lr=math.inf, grad_clip=None, diag_every=1000), tmp_path / "l", data_dir)
    rows = diag_rows(tmp_path / "l")
    assert rows[-1]["step"] == 1 and "grad_norm/layer_0" in rows[-1] and "attn_max_logit/layer_0" not in rows[-1]

    # gradient path at a non-diag step: weights are fine, so the row has both parts
    real = T.torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def fake(params, clip):
        calls["n"] += 1
        real(params, clip)
        return T.torch.tensor(float("nan")) if calls["n"] == 4 else T.torch.tensor(1.0)

    monkeypatch.setattr(T.torch.nn.utils, "clip_grad_norm_", fake)
    with pytest.raises(NonFiniteLoss):
        run(cfg(diag_every=1000), tmp_path / "g", data_dir)
    last = diag_rows(tmp_path / "g")[-1]
    assert last["step"] == 3 and "grad_norm/layer_0" in last and "attn_max_logit/layer_0" in last


def test_data_exhaustion_preflight(tmp_path, data_dir):
    lg = RecordingLogger()
    with pytest.raises(data.DataExhausted, match="needs"):
        run(cfg(total_steps=5000), tmp_path / "x", data_dir, logger=lg)  # 5000*256 > 320000 tokens
    assert not (tmp_path / "x" / "losses.jsonl").exists() and not (tmp_path / "x" / "FAILED").exists()
    assert lg.finished == 1
    with pytest.raises(data.DataExhausted):
        run(cfg(data_offset=319_900), tmp_path / "y", data_dir)
