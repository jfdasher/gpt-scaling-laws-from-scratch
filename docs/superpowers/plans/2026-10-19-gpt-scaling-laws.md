# Small GPT Scaling Laws on Modal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a plain-PyTorch GPT training stack. Use it on Modal to fit a compute-optimal scaling law on FineWeb-Edu, test that law with a pre-registered 3e17-FLOP prediction, and run a training-dynamics track. Publish the results on W&B and the HF Hub.

**Architecture:** `gptlab/` is a Modal-free package (sizing math, model, data, tokenizer, baselines, training loop) that runs and is tested on CPU. `sweep.py` generates every phase's run configs from code and estimates their cost. `modal_app.py` is the only file that imports `modal`: it holds volumes, the image, the GPU and CPU functions, and the phase entrypoints. `analysis/` fits and plots offline from the pulled `results/*.json`. Tasks 1–12 build and test the code for free on CPU. Tasks 13–18 spend money on Modal, so each one is a user-ordered gate.

**Tech Stack:** Python 3.12, uv, PyTorch (≥2.5), HF `tokenizers`, pyarrow, numpy, scipy, matplotlib, wandb, transformers (for export and the GPT-2 baseline only), Modal ≥1.0, pytest.

**Spec:** `docs/superpowers/specs/2026-10-19-gpt-scaling-laws-design.md`

## Global Constraints

- **Never import `modal` outside `modal_app.py`.** `gptlab/`, `sweep.py` and `analysis/` must import and run on a CPU-only machine.
- **Only `torch` is allowed in the training step.** No HF Trainer, Accelerate or Lightning. `transformers` is imported only in `gptlab/baselines.py` (GPT-2 reference) and `analysis/export_hf.py`.
- **Constants (in `gptlab/flops.py`):**
  - `VOCAB_SIZE = 8192`, `SEQ_LEN = 512`, `HEAD_DIM = 64`, `BATCH_TOKENS = 65536`
  - `H100_PEAK_FLOPS = 989e12`, `H100_DOLLARS_PER_HOUR = 3.95`
- **Compute accounting:** C = 6·N·D, where N = **total** parameters (tied embeddings counted once) and D = `total_steps × batch_tokens`.
- **Grid:** budgets `(3e15, 1e16, 3e16, 1e17)`, tokens/param ratios `(5, 10, 20, 40, 80)`, test budget `3e17`.
- **Val split:** `int(sha256(doc_id).hexdigest(), 16) % 1000 < VAL_PER_MILLE` with `VAL_PER_MILLE = 5`. Val text is never printed or opened by hand.
- **Data files** on the data volume and in test fixtures: `train.bin`, `val.bin` (uint16), `tokenizer.json`, `val_manifest.json`.
- **Manifest serialization:** `json.dumps(m, indent=2, sort_keys=True)`, everywhere.
- **Run identity:** `run_id(cfg)` = first 12 hex characters of sha256 over the config JSON, excluding `name`, `phase`, `budget_flops` and `tokens_per_param`. Run directory: `/vol/ckpt/runs/<run_id>/`.
- **Module names mirror HF `LlamaForCausalLM`** (`embed_tokens`, `layers.i.self_attn.{q,k,v,o}_proj`, `layers.i.mlp.{gate,up,down}_proj`, `layers.i.input_layernorm`, `layers.i.post_attention_layernorm`, `norm`). RoPE uses the HF `rotate_half` convention.
- **Commits:** do NOT add `Co-Authored-By` or any self-credit lines. This is the user's global instruction, and it overrides any harness attribution reminder.
- **Package installs:** use `uv` only, never the system pip.

**User decisions (already made):**
- "Use FLOPs as the budget": budgets 3e15, 1e16, 3e16, 1e17; test run at 3e17.
- Five sizes per budget at 5–80 tokens/param; second seed for all five at the smallest budget.
- Cosine schedule sized to each run (not WSD); μP deferred.
- FineWeb-Edu sample-10BT; own 8k byte-level BPE trained with HF `tokenizers`.
- N counts total parameters, with tied embeddings; non-embedding N is reported alongside. Report bits-per-byte.
- Standard parameterization, with an LR sweep at the small and large ends.
- Llama-style architecture (RMSNorm, RoPE, SwiGLU, no biases, tied embeddings).
- Code structure: a plain PyTorch package plus a thin Modal layer.
- Training-dynamics track included in full (instability, warmup, β₂, batch size).
- 64k-token batch for the grid and test runs.
- Modal: H100 for every GPU run; $60 workspace spend limit.

**Corrections to the spec made while planning (binding):**
1. **`VAL_PER_MILLE` is 5, not 2.** With 0.2%, the ~4 parquet files we encode would give only ~4–5M val tokens, below the 10M target.
2. **LR-sweep runs aren't expected to be reused as grid points.** Grid LRs come from the fitted rule, so they rarely equal a swept LR. Run-ID dedupe stays, and it still merges `dyn/baseline` with `grid/3e16/r20`.
3. **The isoflop exponents satisfy b = 1 − a exactly,** because D_opt = C/(6·N_opt). The analysis reports b and notes the identity.
4. **The exact parameter count is** N = L·(4d² + 3·d·h + 2d) + d + V·d, with h = `ffn_hidden(d)` = 64·round(8d/192). The 12·L·d² figure is only an approximation.
5. **The GPU smoke resume check compares per-step losses within a tolerance, not exactly.** GPU kernels aren't bitwise deterministic. The exact check is the CPU resume test (Task 8).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv project, deps, CPU-only torch index for local dev, pytest config |
| `gptlab/flops.py` | constants, `ffn_hidden`, `param_count`, `choose_shape`, `size_for_budget` |
| `gptlab/schedule.py` | `lr_at` (warmup + cosine to a floor) |
| `gptlab/model.py` | `GPTConfig`, `RMSNorm`, RoPE, `Attention`, `MLP`, `Block`, `GPT`, `loss_fn` |
| `gptlab/reference.py` | `RefAdamW`, `causal_attention` (hand-written, parity-tested) |
| `gptlab/data.py` | `TokenStream`, `val_batches`, manifest build/verify, `bits_per_byte` |
| `gptlab/tokenizer.py` | split rule, tokenizer training, document encoding, shard concat, val merge |
| `gptlab/baselines.py` | uniform/unigram/bigram cross-entropy, GPT-2 bpb, `compute_baselines` |
| `gptlab/train.py` | `TrainConfig`, `run_id`, loggers, checkpointing, `train`, `benchmark` |
| `sweep.py` | phase configs, LR rule, cost estimate, `plan` / `fit-lr-rule` CLI |
| `modal_app.py` | image, volumes, data-prep/baseline/bench/train/collect/push functions and entrypoints |
| `analysis/common.py` | load results, failed runs and loss curves from `results/` |
| `analysis/isoflop.py` | parabola fits, seed noise σ, bootstrap, power laws, frontier |
| `analysis/parametric.py` | L(N, D) Huber/L-BFGS fit, bootstrap, optimal allocation |
| `analysis/predict.py` | pre-registered prediction file; comparison against the actual result |
| `analysis/costs.py` | cost line, phase totals, `costs.csv` |
| `analysis/plots.py` | all figures |
| `analysis/export_hf.py` | GPT → `LlamaForCausalLM`, model card, `export_run` |
| `tests/…` | one test file per module |
| `NOTES.md` | dated log of failures and surprises (source for "What didn't work") |
| `README.md` | final write-up |

---

### Task 1: Project scaffold

**Goal:** A uv project with an importable `gptlab` package, empty `analysis` package, a running pytest suite, and git hygiene.

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `gptlab/__init__.py`, `analysis/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`, `NOTES.md`

**Acceptance Criteria:**
- [ ] `uv sync` succeeds; torch comes from the CPU wheel index locally
- [ ] `uv run pytest -q` reports `1 passed`
- [ ] `.gitignore` excludes `.venv/`, `__pycache__/`, `wandb/`, `*.bin`, `*.pt`, `/export/`

**Verify:** `uv sync && uv run pytest -q` → `1 passed`

**Steps:**

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "gptlab"
version = "0.1.0"
description = "Small GPT scaling laws, trained from scratch in plain PyTorch on Modal"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.5",
    "numpy>=1.26",
    "tokenizers>=0.20",
    "pyarrow>=16",
    "huggingface_hub>=0.25",
    "wandb>=0.18",
    "scipy>=1.13",
    "matplotlib>=3.9",
    "modal>=1.0",
    "transformers>=4.45",
    "safetensors>=0.4",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["gptlab"]

[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
wandb/
*.bin
*.pt
/export/
```

- [ ] **Step 3: Write the package markers, smoke test and notes file**

`gptlab/__init__.py`:
```python
"""Plain-PyTorch GPT training stack for the scaling-law study."""
```

`analysis/__init__.py`:
```python
"""Offline analysis of pulled run results."""
```

`tests/__init__.py`: empty file.

`tests/test_smoke.py`:
```python
import gptlab


def test_package_imports():
    assert gptlab.__doc__
```

`NOTES.md`:
```markdown
# Lab notes

Dated log of failures, surprises and fixes. This is the raw material for the README's
"What didn't work" section — write entries as they happen, not afterwards.

Format: `## YYYY-MM-DD — short title`, then what happened, how it was diagnosed, what fixed it.
```

- [ ] **Step 4: Sync and run**

Run: `uv sync && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore gptlab analysis tests NOTES.md
git commit -m "chore: scaffold uv project, gptlab package and test suite"
```

---

### Task 2: Sizing math and LR schedule

**Goal:** Pure functions that turn (budget, tokens/param) into an exact model shape and step count, plus the warmup + cosine LR schedule.

**Files:**
- Create: `gptlab/flops.py`, `gptlab/schedule.py`, `tests/test_flops.py`, `tests/test_schedule.py`

**Acceptance Criteria:**
- [ ] `param_count` implements L·(4d² + 3dh + 2d) + d + V·d (+2·HEAD_DIM per layer with qk_norm)
- [ ] `choose_shape` lands within 20% of the target N for targets from 2e6 to 1e8, with d/L in [16, 64]
- [ ] `size_for_budget` gives flops within 1% of the budget, and 5 distinct shapes per grid budget
- [ ] `lr_at` gives peak/warmup at step 0, peak at the end of warmup, and approaches `final_frac·peak` at the last step; with warmup_frac=0, step 0 = peak

**Verify:** `uv run pytest -q tests/test_flops.py tests/test_schedule.py` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_flops.py`:
```python
import math

import pytest

from gptlab.flops import (
    HEAD_DIM,
    MAX_ASPECT,
    MIN_ASPECT,
    choose_shape,
    ffn_hidden,
    param_count,
    size_for_budget,
)

GRID_BUDGETS = (3e15, 1e16, 3e16, 1e17)
GRID_RATIOS = (5.0, 10.0, 20.0, 40.0, 80.0)


def test_ffn_hidden_is_multiple_of_64_near_8_thirds():
    for d in (64, 128, 192, 256, 512, 768):
        h = ffn_hidden(d)
        assert h % 64 == 0
        assert abs(h - 8 * d / 3) <= 32


def test_param_count_formula():
    d, L, V = 128, 4, 8192
    h = ffn_hidden(d)
    assert param_count(L, d, V) == L * (4 * d * d + 3 * d * h + 2 * d) + d + V * d
    assert param_count(L, d, V, qk_norm=True) == param_count(L, d, V) + L * 2 * HEAD_DIM


@pytest.mark.parametrize("target", [2e6 * (1.5**k) for k in range(10)])
def test_choose_shape_close_to_target(target):
    s = choose_shape(target)
    n = param_count(s.n_layer, s.d_model)
    assert abs(math.log(n / target)) < math.log(1.2)
    assert MIN_ASPECT <= s.d_model / s.n_layer <= MAX_ASPECT
    assert s.d_model % HEAD_DIM == 0


def test_size_for_budget_hits_budget_and_distinct_shapes():
    for budget in GRID_BUDGETS:
        sizes = [size_for_budget(budget, r) for r in GRID_RATIOS]
        for s in sizes:
            assert abs(s.flops / budget - 1) < 0.01
            assert s.tokens == s.steps * 65536
        shapes = {(s.n_layer, s.d_model) for s in sizes}
        assert len(shapes) == 5
        ns = [s.n_params for s in sizes]
        assert ns == sorted(ns, reverse=True)  # more tokens/param -> smaller model
```

`tests/test_schedule.py`:
```python
import math

from gptlab.schedule import lr_at


def test_warmup_then_cosine_to_floor():
    total, peak = 1000, 1e-3
    warmup = 50
    assert math.isclose(lr_at(0, total, peak, 0.05, 0.1), peak / warmup)
    assert math.isclose(lr_at(warmup - 1, total, peak, 0.05, 0.1), peak)
    assert math.isclose(lr_at(warmup, total, peak, 0.05, 0.1), peak)
    assert lr_at(total - 1, total, peak, 0.05, 0.1) < 0.1 * peak + 1e-6
    assert lr_at(total - 1, total, peak, 0.05, 0.1) >= 0.1 * peak


def test_monotone_decay_after_warmup():
    lrs = [lr_at(s, 200, 1.0, 0.05, 0.1) for s in range(10, 200)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_no_warmup_starts_at_peak():
    assert math.isclose(lr_at(0, 100, 2e-3, 0.0, 0.1), 2e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_flops.py tests/test_schedule.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gptlab.flops'`

- [ ] **Step 3: Implement `gptlab/flops.py`**

```python
"""Compute accounting and model sizing. C = 6 * N * D with N = total parameters (tied embeddings)."""

from __future__ import annotations

import math
from dataclasses import dataclass

VOCAB_SIZE = 8192
SEQ_LEN = 512
HEAD_DIM = 64
BATCH_TOKENS = 65536
H100_PEAK_FLOPS = 989e12  # dense bf16
H100_DOLLARS_PER_HOUR = 3.95

MIN_ASPECT, MAX_ASPECT, TARGET_ASPECT = 16, 64, 32  # d_model / n_layer
ASPECT_WEIGHT = 0.1  # how much a 2x aspect deviation costs, relative to a size error in ln N


def ffn_hidden(d_model: int) -> int:
    """SwiGLU hidden width: ~8/3 * d, rounded to a multiple of 64."""
    return 64 * max(1, round(8 * d_model / (3 * 64)))


def param_count(n_layer: int, d_model: int, vocab_size: int = VOCAB_SIZE, qk_norm: bool = False) -> int:
    """Exact parameter count of gptlab.model.GPT (tied embeddings counted once)."""
    h = ffn_hidden(d_model)
    per_layer = 4 * d_model * d_model + 3 * d_model * h + 2 * d_model
    if qk_norm:
        per_layer += 2 * HEAD_DIM
    return n_layer * per_layer + d_model + vocab_size * d_model


@dataclass(frozen=True)
class Shape:
    n_layer: int
    d_model: int


def choose_shape(n_target: float, vocab_size: int = VOCAB_SIZE) -> Shape:
    """Shape whose total params are closest to n_target, preferring d/L near TARGET_ASPECT."""
    best: tuple[float, Shape] | None = None
    for d in range(HEAD_DIM, 2048 + 1, HEAD_DIM):
        for n_layer in range(2, 49):
            aspect = d / n_layer
            if not MIN_ASPECT <= aspect <= MAX_ASPECT:
                continue
            n = param_count(n_layer, d, vocab_size)
            cost = abs(math.log(n / n_target)) + ASPECT_WEIGHT * abs(math.log(aspect / TARGET_ASPECT)) / math.log(2)
            if best is None or cost < best[0]:
                best = (cost, Shape(n_layer, d))
    assert best is not None
    return best[1]


@dataclass(frozen=True)
class RunSize:
    n_layer: int
    d_model: int
    n_params: int
    steps: int
    tokens: int
    flops: float


def size_for_budget(
    budget: float, tokens_per_param: float, batch_tokens: int = BATCH_TOKENS, vocab_size: int = VOCAB_SIZE
) -> RunSize:
    """Pick N ~ sqrt(C / (6 r)), snap to a shape, then set D = C / (6 N_actual) in whole batches."""
    shape = choose_shape(math.sqrt(budget / (6 * tokens_per_param)), vocab_size)
    n = param_count(shape.n_layer, shape.d_model, vocab_size)
    steps = max(1, round(budget / (6 * n * batch_tokens)))
    tokens = steps * batch_tokens
    return RunSize(shape.n_layer, shape.d_model, n, steps, tokens, 6.0 * n * tokens)
```

- [ ] **Step 4: Implement `gptlab/schedule.py`**

```python
"""Linear warmup, then cosine decay to final_frac * peak, over a run's own length."""

import math


def lr_at(step: int, total_steps: int, peak_lr: float, warmup_frac: float = 0.05, final_frac: float = 0.1) -> float:
    warmup = int(round(warmup_frac * total_steps))
    if warmup > 0 and step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return peak_lr * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress)))
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest -q tests/test_flops.py tests/test_schedule.py`
Expected: all pass. If `test_choose_shape_close_to_target` fails for some target, print `choose_shape(target)` and its `param_count`. Then adjust `ASPECT_WEIGHT` (never the 20% tolerance), and note the change in `NOTES.md`.

- [ ] **Step 6: Commit**

```bash
git add gptlab/flops.py gptlab/schedule.py tests/test_flops.py tests/test_schedule.py
git commit -m "feat: compute accounting, shape selection and warmup-cosine schedule"
```

---

### Task 3: Llama-style GPT model

**Goal:** The model, with module names that mirror HF Llama, optional QK-norm and z-loss, per-layer attention max-logit stats, and sampling.

**Files:**
- Create: `gptlab/model.py`, `tests/test_model.py`

**Acceptance Criteria:**
- [ ] `GPT.num_params()` equals `param_count(...)` for several shapes, with and without qk_norm
- [ ] Causal: changing token j leaves logits at positions < j unchanged
- [ ] `rotate_half([1,2,3,4]) == [-3,-4,1,2]`
- [ ] A tiny model overfits one batch (loss < 0.05 within 300 Adam steps)
- [ ] `loss_fn` with z_loss > 0 returns total > ce; with stats, the list has one value per layer
- [ ] `generate` is deterministic given a seeded generator

**Verify:** `uv run pytest -q tests/test_model.py` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:
```python
import torch

from gptlab.flops import param_count
from gptlab.model import GPT, GPTConfig, loss_fn, rotate_half


def tiny(**kw):
    cfg = GPTConfig(n_layer=2, d_model=64, vocab_size=64, seq_len=32, **kw)
    torch.manual_seed(0)
    return GPT(cfg)


def test_param_count_matches_formula():
    for L, d, qk in [(2, 64, False), (4, 128, False), (3, 192, True)]:
        m = GPT(GPTConfig(n_layer=L, d_model=d, vocab_size=256, seq_len=32, qk_norm=qk))
        assert m.num_params() == param_count(L, d, 256, qk_norm=qk)
        assert m.num_params_nonembed() == m.num_params() - 256 * d


def test_rotate_half_convention():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_logits_shape_and_causality():
    m = tiny().eval()
    idx = torch.randint(0, 64, (2, 16))
    logits = m(idx)
    assert logits.shape == (2, 16, 64)
    idx2 = idx.clone()
    idx2[:, 10] = (idx2[:, 10] + 1) % 64
    logits2 = m(idx2)
    assert torch.allclose(logits[:, :10], logits2[:, :10], atol=1e-6)
    assert not torch.allclose(logits[:, 10:], logits2[:, 10:])


def test_overfits_single_batch():
    m = tiny()
    idx = torch.randint(0, 64, (2, 17))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(300):
        _, ce = loss_fn(m(x), y)
        opt.zero_grad()
        ce.backward()
        opt.step()
    assert ce.item() < 0.05


def test_z_loss_and_stats():
    m = tiny(qk_norm=True)
    idx = torch.randint(0, 64, (2, 16))
    stats: list[float] = []
    logits = m(idx, stats=stats)
    assert len(stats) == 2
    total, ce = loss_fn(logits, idx, z_loss=1e-2)
    assert total.item() > ce.item()
    total0, ce0 = loss_fn(logits, idx)
    assert total0.item() == ce0.item()


def test_generate_deterministic():
    m = tiny().eval()
    prompt = torch.tensor([[1, 2, 3]])
    a = m.generate(prompt, 8, generator=torch.Generator().manual_seed(7))
    b = m.generate(prompt, 8, generator=torch.Generator().manual_seed(7))
    assert a.shape == (1, 11)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_model.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gptlab.model'`

- [ ] **Step 3: Implement `gptlab/model.py`**

```python
"""Llama-style GPT. Module names mirror HF LlamaForCausalLM so export is a key rename."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gptlab.flops import HEAD_DIM, SEQ_LEN, VOCAB_SIZE, ffn_hidden


@dataclass(frozen=True)
class GPTConfig:
    n_layer: int
    d_model: int
    vocab_size: int = VOCAB_SIZE
    seq_len: int = SEQ_LEN
    qk_norm: bool = False
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def n_head(self) -> int:
        assert self.d_model % HEAD_DIM == 0, "d_model must be a multiple of HEAD_DIM"
        return self.d_model // HEAD_DIM


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * h).to(x.dtype)


def rope_cos_sin(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos + rotate_half(x) * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(HEAD_DIM, cfg.norm_eps) if cfg.qk_norm else None

    def forward(self, x, cos, sin, stats: list[float] | None = None):
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, HEAD_DIM).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos[:t], sin[:t]), apply_rope(k, cos[:t], sin[:t])
        if stats is not None:
            scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(HEAD_DIM)
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            stats.append(scores.masked_fill(~causal, float("-inf")).amax().item())
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, d))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        h = ffn_hidden(cfg.d_model)
        self.gate_proj = nn.Linear(cfg.d_model, h, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, h, bias=False)
        self.down_proj = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, stats=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, stats)
        return x + self.mlp(self.post_attention_layernorm(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        cos, sin = rope_cos_sin(cfg.seq_len, HEAD_DIM, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layer)  # residual-branch output projections
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, stats: list[float] | None = None) -> torch.Tensor:
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, stats)
        return F.linear(self.norm(x), self.embed_tokens.weight)  # tied output head

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_params_nonembed(self) -> int:
        return self.num_params() - self.embed_tokens.weight.numel()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, generator=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.seq_len :])[:, -1, :].float() / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def loss_fn(logits: torch.Tensor, targets: torch.Tensor, z_loss: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (total loss to backprop, plain cross-entropy for logging)."""
    logits = logits.float()
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
    if z_loss == 0.0:
        return ce, ce
    z = torch.logsumexp(logits, dim=-1)
    return ce + z_loss * z.pow(2).mean(), ce
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q tests/test_model.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add gptlab/model.py tests/test_model.py
git commit -m "feat: Llama-style GPT with RoPE, SwiGLU, tied embeddings, qk-norm and z-loss switches"
```

---

### Task 4: Hand-written AdamW and attention (reference implementations)

**Goal:** Readable reference versions of AdamW and causal attention, proven equal to PyTorch's to tolerance. These are interview artefacts, not used in real runs.

**Files:**
- Create: `gptlab/reference.py`, `tests/test_reference.py`

**Acceptance Criteria:**
- [ ] `RefAdamW` matches `torch.optim.AdamW` (decay and no-decay groups) over 50 float64 steps to atol 1e-10
- [ ] `causal_attention` matches `F.scaled_dot_product_attention(is_causal=True)` to atol 1e-5 in fp32

**Verify:** `uv run pytest -q tests/test_reference.py` → 2 passed

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_reference.py`:
```python
import copy

import torch
import torch.nn.functional as F

from gptlab.reference import RefAdamW, causal_attention


def _groups(model, wd):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def test_ref_adamw_matches_torch():
    torch.manual_seed(0)
    model_a = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)).double()
    model_b = copy.deepcopy(model_a)
    opt_a = torch.optim.AdamW(_groups(model_a, 0.1), lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    opt_b = RefAdamW(_groups(model_b, 0.1), lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    for step in range(50):
        x = torch.randn(32, 8, dtype=torch.float64)
        lr = 1e-2 * (0.5 + 0.5 * step / 50)  # changing LR, like a schedule
        for opt, model in ((opt_a, model_a), (opt_b, model_b)):
            for g in opt.param_groups:
                g["lr"] = lr
            loss = model(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-10, rtol=0)


def test_causal_attention_matches_sdpa():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 10, 16) for _ in range(3))
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(causal_attention(q, k, v), expected, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_reference.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gptlab.reference'`

- [ ] **Step 3: Implement `gptlab/reference.py`**

```python
"""Hand-written AdamW and causal attention, for understanding. Real runs use the fused/SDPA versions."""

from __future__ import annotations

import math

import torch


class RefAdamW:
    """AdamW (Loshchilov & Hutter): Adam moments with bias correction, plus decoupled weight decay."""

    def __init__(self, param_groups, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.param_groups = [dict(g, lr=g.get("lr", lr)) for g in param_groups]
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}

    def zero_grad(self) -> None:
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        bc1 = 1 - self.beta1**self.t
        bc2 = 1 - self.beta2**self.t
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                m = self.m.setdefault(id(p), torch.zeros_like(p))
                v = self.v.setdefault(id(p), torch.zeros_like(p))
                m.mul_(self.beta1).add_(p.grad, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)
                p.mul_(1 - lr * wd)  # decoupled decay: shrink weights, independent of the gradient
                m_hat = m / bc1
                v_hat = v / bc2
                p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-lr)


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """softmax(q k^T / sqrt(d) + causal mask) v, for tensors shaped (batch, heads, time, head_dim)."""
    t = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    future = torch.ones(t, t, dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q tests/test_reference.py`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add gptlab/reference.py tests/test_reference.py
git commit -m "feat: hand-written AdamW and causal attention with parity tests"
```

---

### Task 5: Data stream, val batches, manifest and bits-per-byte

**Goal:** Deterministic sequential reads of a uint16 token file (step k always maps to the same slice), val iteration, a val manifest that freezes the split, and the bpb conversion.

**Files:**
- Create: `gptlab/data.py`, `tests/test_data.py`

**Acceptance Criteria:**
- [ ] `TokenStream.get(i)` returns `(x, y)` with y = x shifted by one, from `offset + i·B·T`; going past the end raises `DataExhausted`
- [ ] Step k's slice is independent of micro-batch size (same tokens for micro_batch 2×accum 2 and micro_batch 4×accum 1)
- [ ] `val_batches` covers every val position except the last, with no overlap; it honours `max_tokens`
- [ ] `verify_manifest` passes on an untouched dir and raises `ManifestMismatch` after `val.bin` or `tokenizer.json` changes
- [ ] `bits_per_byte(loss, n_tokens, n_bytes) == loss/ln2 · n_tokens/n_bytes`

**Verify:** `uv run pytest -q tests/test_data.py` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_data.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gptlab.data'`

- [ ] **Step 3: Implement `gptlab/data.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q tests/test_data.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add gptlab/data.py tests/test_data.py
git commit -m "feat: deterministic token stream, val batches, frozen-val manifest, bpb"
```

---

### Task 6: Split rule, tokenizer training, encoding, shard assembly

**Goal:** The pure-Python data-prep logic that Modal containers call. It covers the document-hash split, 8k BPE training, per-file encoding with a seeded shuffle, train shard concatenation, and the merge of the sorted val docs.

**Files:**
- Create: `gptlab/tokenizer.py`, `tests/test_tokenizer.py`

**Acceptance Criteria:**
- [ ] `is_val_doc` is deterministic, and the val rate on 200k synthetic IDs is within 0.4–0.6%
- [ ] The tokenizer trained on a toy corpus has `<|endoftext|>` at id 0; encode → decode roundtrips Unicode, code and whitespace runs
- [ ] `encode_file_docs` puts no val doc into train; train EOT count == number of train docs; val docs are sorted by id; the same seed gives the same order
- [ ] `concat_shards` writes whole shards in seeded order until it reaches ≥ min_tokens, and raises `InsufficientData` otherwise
- [ ] `merge_val_shards` gathers whole docs in global id order until it reaches ≥ min_tokens, and returns (docs, tokens, bytes)

**Verify:** `uv run pytest -q tests/test_tokenizer.py` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_tokenizer.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_tokenizer.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gptlab.tokenizer'`

- [ ] **Step 3: Implement `gptlab/tokenizer.py`**

```python
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

    for doc in docs:
        batch.append(doc)
        if len(batch) >= ENCODE_BATCH:
            flush()
    flush()
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q tests/test_tokenizer.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add gptlab/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: doc-hash val split, 8k BPE training, seeded encoding and shard assembly"
```

---

### Task 7: Baselines

**Goal:** The numbers to beat, measured before any training: uniform, unigram, an interpolated bigram (fit on train, evaluated on val), and a GPT-2-small bits-per-byte reference.

**Files:**
- Create: `gptlab/baselines.py`, `tests/test_baselines.py`

**Acceptance Criteria:**
- [ ] On a deterministic cyclic sequence: bigram CE < unigram CE ≤ uniform + 1e-6
- [ ] Chunked `bigram_counts` equals the naive count
- [ ] `tune_lambda` picks the λ that minimizes held-out CE on the grid
- [ ] `decode_docs` splits on EOT and roundtrips the texts
- [ ] `compute_baselines(data_dir, device, include_gpt2=False)` returns uniform/unigram/bigram, each with `loss_nats` and `bpb`

**Verify:** `uv run pytest -q tests/test_baselines.py` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_baselines.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_baselines.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gptlab.baselines'`

- [ ] **Step 3: Implement `gptlab/baselines.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q tests/test_baselines.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add gptlab/baselines.py tests/test_baselines.py
git commit -m "feat: uniform/unigram/bigram baselines and GPT-2 bpb reference"
```

---

### Task 8: Training loop with checkpoint/resume

**Goal:** `train(cfg, run_dir, data_dir, ...)` is the whole training step: bf16 autocast, gradient accumulation, clipping, the schedule, and fused AdamW. Around it: evals, per-layer stats, samples, atomic checkpoints, exact resume, loud NaN failure with forensics, and `result.json`. Also `benchmark()` for throughput.

**Files:**
- Create: `gptlab/train.py`, `tests/test_train.py`

**Acceptance Criteria:**
- [ ] `run_id` ignores `name`/`phase`/`budget_flops`/`tokens_per_param` and changes with `seed`
- [ ] Resume is exact: 20 straight steps produce the same per-step losses (`losses.jsonl`) and final val loss as 10 steps + `SimulatedPreemption` + resume + 10 steps (CPU fp32, compile off)
- [ ] Loss falls by more than 1.0 nat over 80 steps on the learnable synthetic stream
- [ ] A NaN loss writes `forensic.json` (step, loss, token span, gpu_seconds) and `FAILED`, then raises `NonFiniteLoss`; calling again raises `RunFailed`
- [ ] An existing `result.json` short-circuits (returns it, adds no new loss lines)
- [ ] A tampered `val.bin` raises `ManifestMismatch` before training
- [ ] `result.json` has every field listed in spec §3 (run_id, n_params, n_params_nonembed, tokens, flops, val_loss, val_bpb, gpu_seconds, gpu_hours, cost_usd, wandb_url, git_commit, …)

**Verify:** `uv run pytest -q tests/test_train.py` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_train.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_train.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gptlab.train'`

- [ ] **Step 3: Implement `gptlab/train.py`**

```python
"""The training loop. train() is what a Modal container (or a CPU test) calls for one run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

import torch

from gptlab import data as gdata
from gptlab.flops import BATCH_TOKENS, H100_DOLLARS_PER_HOUR, H100_PEAK_FLOPS, SEQ_LEN, VOCAB_SIZE
from gptlab.model import GPT, GPTConfig, loss_fn
from gptlab.schedule import lr_at

SAMPLE_PROMPTS = (
    "The water cycle is",
    "In 1492,",
    "To solve a quadratic equation,",
    "The most important idea in biology is",
)
HASH_EXCLUDE = frozenset({"name", "phase", "budget_flops", "tokens_per_param"})  # bookkeeping, not behaviour


@dataclass(frozen=True)
class TrainConfig:
    name: str
    n_layer: int
    d_model: int
    total_steps: int
    peak_lr: float
    phase: str = ""
    tag: str = ""  # distinguishes otherwise identical runs (e.g. the preempted smoke run)
    batch_tokens: int = BATCH_TOKENS
    micro_batch: int = 64  # sequences per forward/backward
    seq_len: int = SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    warmup_frac: float = 0.05
    final_lr_frac: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float | None = 1.0
    qk_norm: bool = False
    z_loss: float = 0.0
    seed: int = 0
    data_offset: int = 0
    log_every: int = 10
    eval_every: int | None = None  # default: total_steps // 10
    eval_tokens: int = 1_048_576
    ckpt_every_s: float = 600.0
    compile: bool = True
    budget_flops: float = 0.0
    tokens_per_param: float = 0.0


def run_id(cfg: TrainConfig) -> str:
    payload = {k: v for k, v in asdict(cfg).items() if k not in HASH_EXCLUDE}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


class SimulatedPreemption(RuntimeError):
    pass


class NonFiniteLoss(RuntimeError):
    pass


class RunFailed(RuntimeError):
    pass


class Logger(Protocol):
    url: str | None

    def log(self, metrics: dict, step: int) -> None: ...
    def log_samples(self, step: int, samples: list[tuple[str, str]]) -> None: ...
    def finish(self) -> None: ...


class NullLogger:
    url = None

    def log(self, metrics, step):
        pass

    def log_samples(self, step, samples):
        pass

    def finish(self):
        pass


class WandbLogger:
    """W&B run keyed by run_id, so a resumed container continues the same run."""

    def __init__(self, project: str, cfg: TrainConfig):
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project, id=run_id(cfg), name=cfg.name, group=cfg.phase, config=asdict(cfg), resume="allow"
        )
        self.url = self.run.url

    def log(self, metrics, step):
        self.run.log(metrics, step=step)

    def log_samples(self, step, samples):
        table = self._wandb.Table(columns=["step", "prompt", "completion"], data=[[step, p, c] for p, c in samples])
        self.run.log({"samples": table}, step=step)

    def finish(self):
        self.run.finish()


def _autocast(device: str):
    kind = torch.device(device).type
    return torch.autocast(device_type=kind, dtype=torch.bfloat16, enabled=kind == "cuda")


def make_optimizer(model: GPT, cfg: TrainConfig, fused: bool) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8, fused=fused)


def save_checkpoint(path: Path, model, optimizer, step: int, gpu_seconds: float) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "gpu_seconds": gpu_seconds,
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic: a crash leaves either the old or the new checkpoint, never half of one


def load_checkpoint(path: Path, model, optimizer) -> dict:
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["torch_rng"])
    return ck


def forward_backward(fwd, stream: gdata.TokenStream, step: int, accum: int, cfg: TrainConfig, device: str) -> float:
    """One optimizer step's worth of micro-batches; grads accumulate in .grad. Returns mean cross-entropy."""
    loss_sum = torch.zeros((), device=device)
    for micro in range(accum):
        x, y = stream.get(step * accum + micro)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _autocast(device):
            logits = fwd(x)
        total, ce = loss_fn(logits, y, cfg.z_loss)
        (total / accum).backward()
        loss_sum += ce.detach()
    return (loss_sum / accum).item()


@torch.no_grad()
def evaluate(model: GPT, val_path: Path, cfg: TrainConfig, device: str, max_tokens: int | None) -> float:
    total, count = 0.0, 0
    for x, y in gdata.val_batches(val_path, cfg.seq_len, cfg.micro_batch, max_tokens):
        x, y = x.to(device), y.to(device)
        with _autocast(device):
            logits = model(x)
        _, ce = loss_fn(logits, y)
        total += ce.item() * y.numel()
        count += y.numel()
    return total / count


def eval_metrics(model: GPT, val_path: Path, cfg: TrainConfig, device: str, manifest: dict) -> dict:
    loss = evaluate(model, val_path, cfg, device, cfg.eval_tokens)
    out = {"val/loss": loss, "val/bpb": gdata.bits_per_byte(loss, manifest["n_tokens"], manifest["n_bytes"])}
    x, _ = next(gdata.val_batches(val_path, cfg.seq_len, min(cfg.micro_batch, 8), cfg.eval_tokens))
    stats: list[float] = []
    with torch.no_grad(), _autocast(device):
        model(x.to(device), stats=stats)
    out.update({f"attn_max_logit/layer_{i}": v for i, v in enumerate(stats)})
    return out


def layer_grad_norms(model: GPT) -> dict:
    out = {}
    for i, layer in enumerate(model.layers):
        sq = sum(float(p.grad.float().pow(2).sum()) for p in layer.parameters() if p.grad is not None)
        out[f"grad_norm/layer_{i}"] = sq**0.5
    if model.embed_tokens.weight.grad is not None:
        out["grad_norm/embed"] = model.embed_tokens.weight.grad.float().norm().item()
    return out


def generate_samples(model: GPT, tokenizer_path: Path, prompts, device: str, max_new_tokens: int = 48):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    gen = torch.Generator(device=device).manual_seed(1234)
    out = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        with _autocast(device):
            seq = model.generate(ids, max_new_tokens, generator=gen)
        out.append((prompt, tok.decode(seq[0, ids.size(1) :].tolist())))
    return out


def _write_forensics(run_dir, model, optimizer, step, stream, accum, loss, lr, gpu_seconds) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, run_dir / "forensic.pt")
    span = [stream.span(step * accum)[0], stream.span(step * accum + accum - 1)[1]]
    info = {"step": step, "loss": repr(loss), "lr": lr, "token_span": span, "gpu_seconds": gpu_seconds}
    (run_dir / "forensic.json").write_text(json.dumps(info, indent=2))
    (run_dir / "FAILED").write_text(f"non-finite loss {loss!r} at step {step}")


def train(
    cfg: TrainConfig,
    run_dir,
    data_dir,
    logger: Logger | None = None,
    device: str | None = None,
    commit: Callable[[], None] = lambda: None,
    preempt_at_step: int | None = None,
    dollars_per_hour: float = H100_DOLLARS_PER_HOUR,
    sample_prompts=SAMPLE_PROMPTS,
    git_commit: str | None = None,
) -> dict:
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (run_dir / "FAILED").exists():
        raise RunFailed((run_dir / "FAILED").read_text())
    manifest = gdata.verify_manifest(data_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger = logger or NullLogger()
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed)
    model = GPT(
        GPTConfig(n_layer=cfg.n_layer, d_model=cfg.d_model, vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, qk_norm=cfg.qk_norm)
    ).to(device)
    optimizer = make_optimizer(model, cfg, fused=device.startswith("cuda"))
    ckpt_path = run_dir / "ckpt.pt"
    step, gpu_seconds = 0, 0.0
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, model, optimizer)
        step, gpu_seconds = ck["step"], ck["gpu_seconds"]
    fwd = torch.compile(model) if (cfg.compile and device.startswith("cuda")) else model

    accum = cfg.batch_tokens // (cfg.micro_batch * cfg.seq_len)
    assert accum * cfg.micro_batch * cfg.seq_len == cfg.batch_tokens, "batch_tokens must be a multiple of micro_batch*seq_len"
    stream = gdata.TokenStream(data_dir / gdata.TRAIN_BIN, cfg.seq_len, cfg.micro_batch, cfg.data_offset)
    val_path = data_dir / gdata.VAL_BIN
    eval_every = cfg.eval_every or max(1, cfg.total_steps // 10)
    n_params = model.num_params()

    seg_start = time.monotonic()
    last_ckpt = t_log = seg_start
    tokens_since_log = 0

    def elapsed() -> float:
        return gpu_seconds + (time.monotonic() - seg_start)

    with (run_dir / "losses.jsonl").open("a") as losses_file:

        def checkpoint(at_step: int) -> None:
            losses_file.flush()
            save_checkpoint(ckpt_path, model, optimizer, at_step, elapsed())
            commit()

        while step < cfg.total_steps:
            if preempt_at_step is not None and step == preempt_at_step and not (run_dir / "preempted").exists():
                checkpoint(step)
                (run_dir / "preempted").touch()
                commit()
                raise SimulatedPreemption(f"simulated preemption at step {step}")
            lr = lr_at(step, cfg.total_steps, cfg.peak_lr, cfg.warmup_frac, cfg.final_lr_frac)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = forward_backward(fwd, stream, step, accum, cfg, device)
            if not math.isfinite(loss):
                losses_file.flush()
                _write_forensics(run_dir, model, optimizer, step, stream, accum, loss, lr, elapsed())
                commit()
                raise NonFiniteLoss(f"loss {loss!r} at step {step}; see {run_dir / 'forensic.json'}")
            clip = cfg.grad_clip if cfg.grad_clip is not None else math.inf
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()  # pre-clip norm
            will_eval = (step + 1) % eval_every == 0 or step + 1 == cfg.total_steps
            layer_norms = layer_grad_norms(model) if will_eval else {}
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses_file.write(json.dumps({"step": step, "loss": loss, "lr": lr, "grad_norm": grad_norm}) + "\n")
            step += 1
            tokens_since_log += cfg.batch_tokens
            if step % cfg.log_every == 0:
                now = time.monotonic()
                tps = tokens_since_log / max(now - t_log, 1e-9)
                t_log, tokens_since_log = now, 0
                logger.log(
                    {
                        "train/loss": loss, "train/lr": lr, "train/grad_norm": grad_norm,
                        "perf/tokens_per_s": tps, "perf/mfu": 6 * n_params * tps / H100_PEAK_FLOPS,
                        "gpu_seconds": elapsed(),
                    },
                    step=step,
                )
            if will_eval:
                metrics = eval_metrics(model, val_path, cfg, device, manifest)
                metrics.update(layer_norms)
                logger.log(metrics, step=step)
                if sample_prompts:
                    logger.log_samples(step, generate_samples(model, data_dir / gdata.TOKENIZER_JSON, sample_prompts, device))
            if step < cfg.total_steps and time.monotonic() - last_ckpt >= cfg.ckpt_every_s:
                checkpoint(step)
                last_ckpt = time.monotonic()

    final_loss = evaluate(model, val_path, cfg, device, max_tokens=None)
    total_seconds = elapsed()
    torch.save(model.state_dict(), run_dir / "final.pt")
    tokens = cfg.total_steps * cfg.batch_tokens
    result = {
        "run_id": run_id(cfg),
        "name": cfg.name,
        "phase": cfg.phase,
        "config": asdict(cfg),
        "n_layer": cfg.n_layer,
        "d_model": cfg.d_model,
        "n_params": n_params,
        "n_params_nonembed": model.num_params_nonembed(),
        "tokens": tokens,
        "flops": 6.0 * n_params * tokens,
        "budget_flops": cfg.budget_flops,
        "nominal_tokens_per_param": cfg.tokens_per_param,
        "tokens_per_param": tokens / n_params,
        "val_loss": final_loss,
        "val_bpb": gdata.bits_per_byte(final_loss, manifest["n_tokens"], manifest["n_bytes"]),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600,
        "cost_usd": total_seconds / 3600 * dollars_per_hour,
        "torch_version": torch.__version__,
        "wandb_url": logger.url,
        "git_commit": git_commit,
    }
    logger.log({"final/val_loss": result["val_loss"], "final/val_bpb": result["val_bpb"]}, step=cfg.total_steps)
    tmp = result_path.with_name("result.json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, result_path)
    ckpt_path.unlink(missing_ok=True)
    commit()
    logger.finish()
    return result


def benchmark(n_layer: int, d_model: int, data_dir, device: str = "cuda", steps: int = 30, warmup: int = 10) -> dict:
    """Steady-state throughput for one shape with the real step (compile, bf16, fused AdamW)."""
    cfg = TrainConfig(name="bench", n_layer=n_layer, d_model=d_model, total_steps=steps, peak_lr=1e-3)
    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layer=n_layer, d_model=d_model)).to(device)
    optimizer = make_optimizer(model, cfg, fused=device.startswith("cuda"))
    fwd = torch.compile(model) if device.startswith("cuda") else model
    accum = cfg.batch_tokens // (cfg.micro_batch * cfg.seq_len)
    stream = gdata.TokenStream(Path(data_dir) / gdata.TRAIN_BIN, cfg.seq_len, cfg.micro_batch)
    t0 = 0.0
    for step in range(steps):
        if step == warmup:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.monotonic()
        forward_backward(fwd, stream, step, accum, cfg, device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    tps = (steps - warmup) * cfg.batch_tokens / (time.monotonic() - t0)
    n = model.num_params()
    return {"n_layer": n_layer, "d_model": d_model, "n_params": n, "tokens_per_s": tps,
            "flops_per_s": 6 * n * tps, "mfu": 6 * n * tps / H100_PEAK_FLOPS}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q tests/test_train.py`
Expected: all pass. If `test_resume_is_exact` fails, look for state that isn't restored (optimizer state, step, LR) before changing anything. Don't relax the equality to a tolerance.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add gptlab/train.py tests/test_train.py
git commit -m "feat: training loop with bf16, clipping, cosine LR, atomic checkpoints, exact resume, NaN forensics"
```

---

### Task 9: Sweep definitions, LR rule and cost dry-run

**Goal:** Every phase's run configs are generated from code. The LR rule is fitted from the sweep, and a dry-run planner prints runs, skips and the dollar estimate. Nothing in this module launches anything.

**Files:**
- Create: `sweep.py`, `tests/test_sweep.py`

**Acceptance Criteria:**
- [ ] `grid_configs(rule)` = 25 configs: 20 seed-0 plus 5 seed-1 at 3e15 with `data_offset=1_000_000_000`; unique run_ids
- [ ] `lr_configs()` = 13 configs with LRs ×2 apart around each centre
- [ ] `dynamics_configs(rule)` = 12 configs; `dyn/baseline` has the same run_id as `grid/3e16/r20`; batch variants keep tokens within one batch of each other
- [ ] `dynamics_fix_configs(rule, name)` gives `+qknorm` and `+zloss` variants of the named config
- [ ] `best_lrs` treats missing (failed) runs as +inf loss and flags edge optima; `fit_lr_rule` is the exact line through the two sweep ends, and it reports the check-point ratio
- [ ] `declared_test_config(n_opt, rule)` has flops within 1% of 3e17
- [ ] `python sweep.py plan grid --lr-rule <file>` prints the table and totals without importing modal

**Verify:** `uv run pytest -q tests/test_sweep.py` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_sweep.py`:
```python
import json
import math
import sys

import pytest

import sweep
from gptlab.train import run_id

RULE = (math.log(3e-3) + 0.3 * math.log(2.5e6), -0.3)  # lr = 3e-3 at 2.5M, falling as N^-0.3


def test_grid_configs():
    cfgs = sweep.grid_configs(RULE)
    assert len(cfgs) == 25
    assert len({run_id(c) for c in cfgs}) == 25
    seed1 = [c for c in cfgs if c.seed == 1]
    assert len(seed1) == 5
    assert all(c.data_offset == sweep.SEED1_OFFSET and c.budget_flops == 3e15 for c in seed1)


def test_lr_configs():
    cfgs = sweep.lr_configs()
    assert len(cfgs) == 13
    by_group = {}
    for c in cfgs:
        by_group.setdefault((c.budget_flops, c.tokens_per_param), []).append(c.peak_lr)
    for lrs in by_group.values():
        lrs.sort()
        assert all(math.isclose(b / a, 2.0) for a, b in zip(lrs, lrs[1:]))


def test_dynamics_configs():
    cfgs = sweep.dynamics_configs(RULE)
    assert len(cfgs) == 12
    by_name = {c.name: c for c in cfgs}
    grid = {c.name: c for c in sweep.grid_configs(RULE)}
    assert run_id(by_name["dyn/baseline"]) == run_id(grid["grid/3e+16/r20"])
    base_tokens = by_name["dyn/baseline"].total_steps * by_name["dyn/baseline"].batch_tokens
    for c in cfgs:
        if c.name.startswith("dyn/batch"):
            assert abs(c.total_steps * c.batch_tokens - base_tokens) <= c.batch_tokens
            assert c.batch_tokens % (c.micro_batch * c.seq_len) == 0
    assert by_name["dyn/lr10x-noclip"].grad_clip is None
    fixes = sweep.dynamics_fix_configs(RULE, "dyn/lr10x-noclip")
    assert [f.name for f in fixes] == ["dyn/lr10x-noclip+qknorm", "dyn/lr10x-noclip+zloss"]
    assert fixes[0].qk_norm and fixes[1].z_loss == 1e-4


def _result(cfg, loss):
    return {"run_id": run_id(cfg), "phase": "lr", "val_loss": loss, "n_params": 1, "config": {}}


def test_best_lrs_and_rule():
    cfgs = sweep.lr_configs()
    results = []
    for c in cfgs:
        if c.peak_lr == max(x.peak_lr for x in cfgs if x.budget_flops == c.budget_flops):
            continue  # highest LR "diverged": no result
        results.append(_result(c, abs(math.log(c.peak_lr / (c.peak_lr if c.tokens_per_param == 20 else 3e-3))) + 3))
    groups = sweep.best_lrs(cfgs, results)
    assert len(groups) == 3
    assert all(math.isinf(g["losses"][-1]) for g in groups)
    rule = sweep.fit_lr_rule(groups)
    assert {"c0", "c1", "check_ratio"} <= set(rule)


def test_declared_test_config():
    c = sweep.declared_test_config(5e7, RULE)
    from gptlab.flops import param_count

    n = param_count(c.n_layer, c.d_model)
    assert abs(6 * n * c.total_steps * c.batch_tokens / 3e17 - 1) < 0.01
    assert c.phase == "test"


def test_smoke_configs_distinct():
    a, b = sweep.smoke_configs()
    assert run_id(a) != run_id(b)


def test_plan_cli_does_not_import_modal(tmp_path, capsys):
    rule_path = tmp_path / "lr_rule.json"
    rule_path.write_text(json.dumps({"c0": RULE[0], "c1": RULE[1]}))
    sweep.main(["plan", "grid", "--lr-rule", str(rule_path), "--results", str(tmp_path / "results")])
    out = capsys.readouterr().out
    assert "25 runs" in out and "$" in out
    assert "modal" not in sys.modules
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_sweep.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sweep'`

- [ ] **Step 3: Implement `sweep.py`**

```python
"""Run configs for every phase, the LR rule, and a dry-run cost planner. Never launches anything.

    uv run python sweep.py plan <phase> [--lr-rule lr_rule.json] [--from-run NAME]
    uv run python sweep.py fit-lr-rule [--results results]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from gptlab.flops import BATCH_TOKENS, H100_DOLLARS_PER_HOUR, choose_shape, param_count, size_for_budget
from gptlab.train import TrainConfig, run_id

PHASES = ("smoke", "lr", "grid", "dynamics", "dynamics-fix", "test")
GRID_BUDGETS = (3e15, 1e16, 3e16, 1e17)
GRID_RATIOS = (5.0, 10.0, 20.0, 40.0, 80.0)
GRID_EXTRA_RATIOS: dict[float, tuple[float, ...]] = {}  # phase-3 gate, e.g. {1e17: (160.0,)}
SEED1_OFFSET = 1_000_000_000
TEST_BUDGET = 3e17
SMOKE_PREEMPT_STEP = 100
# (budget, tokens/param, centre LR, number of LRs); the first two are the rule's ends, the third is the check
LR_SWEEP = ((3e15, 80.0, 4e-3, 5), (3e16, 5.0, 2e-3, 5), (1e16, 20.0, 3e-3, 3))
DYNAMICS_BUDGET, DYNAMICS_RATIO = 3e16, 20.0
DEFAULT_FLOPS_PER_S = 7e13  # until bench.json exists
RUN_OVERHEAD_S = 90.0  # container start + compile + final eval


def make_config(phase: str, name: str, budget: float, ratio: float, lr: float, **overrides) -> TrainConfig:
    s = size_for_budget(budget, ratio, overrides.get("batch_tokens", BATCH_TOKENS))
    return TrainConfig(
        name=name, phase=phase, n_layer=s.n_layer, d_model=s.d_model, total_steps=s.steps, peak_lr=lr,
        budget_flops=budget, tokens_per_param=ratio, **overrides,
    )


def lr_values(center: float, n: int) -> tuple[float, ...]:
    return tuple(float(f"{center * 2.0**k:.4g}") for k in range(-(n // 2), n // 2 + 1))


def load_lr_rule(path="lr_rule.json") -> tuple[float, float]:
    d = json.loads(Path(path).read_text())
    return d["c0"], d["c1"]


def lr_for(n_params: int, rule: tuple[float, float]) -> float:
    c0, c1 = rule
    return float(f"{math.exp(c0 + c1 * math.log(n_params)):.4g}")


def smoke_configs() -> list[TrainConfig]:
    base = TrainConfig(name="smoke/a", phase="smoke", n_layer=4, d_model=256, total_steps=200, peak_lr=2e-3, eval_every=50)
    return [base, replace(base, name="smoke/preempt", tag="preempt")]


def lr_configs() -> list[TrainConfig]:
    return [
        make_config("lr", f"lr/{budget:.0e}/r{ratio:g}/lr{lr:.3g}", budget, ratio, lr)
        for budget, ratio, center, n in LR_SWEEP
        for lr in lr_values(center, n)
    ]


def grid_configs(rule) -> list[TrainConfig]:
    out = []
    for budget in GRID_BUDGETS:
        for ratio in GRID_RATIOS + GRID_EXTRA_RATIOS.get(budget, ()):
            s = size_for_budget(budget, ratio)
            lr = lr_for(s.n_params, rule)
            out.append(make_config("grid", f"grid/{budget:.0e}/r{ratio:g}", budget, ratio, lr))
            if budget == GRID_BUDGETS[0]:
                out.append(
                    make_config("grid", f"grid/{budget:.0e}/r{ratio:g}/seed1", budget, ratio, lr, seed=1, data_offset=SEED1_OFFSET)
                )
    return out


def dynamics_base(rule) -> TrainConfig:
    s = size_for_budget(DYNAMICS_BUDGET, DYNAMICS_RATIO)
    return make_config("dynamics", "dyn/baseline", DYNAMICS_BUDGET, DYNAMICS_RATIO, lr_for(s.n_params, rule))


def dynamics_configs(rule) -> list[TrainConfig]:
    base = dynamics_base(rule)
    lr = base.peak_lr
    out = [base]
    for mult in (3, 10):
        out.append(replace(base, name=f"dyn/lr{mult}x", peak_lr=lr * mult))
        out.append(replace(base, name=f"dyn/lr{mult}x-noclip", peak_lr=lr * mult, grad_clip=None))
    out += [
        replace(base, name="dyn/warmup0", warmup_frac=0.0),
        replace(base, name="dyn/warmup1", warmup_frac=0.01),
        replace(base, name="dyn/beta2-0.999", beta2=0.999),
    ]
    for bt in (32768, 131072, 262144, 524288):  # same LR at every batch size: noted as a limitation
        out.append(make_config("dynamics", f"dyn/batch{bt // 1024}k", DYNAMICS_BUDGET, DYNAMICS_RATIO, lr, batch_tokens=bt))
    return out


def dynamics_fix_configs(rule, from_run: str) -> list[TrainConfig]:
    src = {c.name: c for c in dynamics_configs(rule)}[from_run]
    return [replace(src, name=f"{from_run}+qknorm", qk_norm=True), replace(src, name=f"{from_run}+zloss", z_loss=1e-4)]


def declared_test_config(n_opt: float, rule, budget: float = TEST_BUDGET) -> TrainConfig:
    shape = choose_shape(n_opt)
    n = param_count(shape.n_layer, shape.d_model)
    steps = max(1, round(budget / (6 * n * BATCH_TOKENS)))
    return TrainConfig(
        name=f"test/{budget:.0e}", phase="test", n_layer=shape.n_layer, d_model=shape.d_model, total_steps=steps,
        peak_lr=lr_for(n, rule), budget_flops=budget, tokens_per_param=steps * BATCH_TOKENS / n,
    )


def configs_for_phase(phase, lr_rule="lr_rule.json", from_run="", prediction="predictions/3e17.json") -> list[TrainConfig]:
    if phase == "smoke":
        return smoke_configs()
    if phase == "lr":
        return lr_configs()
    if phase == "test":
        return [TrainConfig(**json.loads(Path(prediction).read_text())["declared_config"])]
    rule = load_lr_rule(lr_rule)
    if phase == "grid":
        return grid_configs(rule)
    if phase == "dynamics":
        return dynamics_configs(rule)
    if phase == "dynamics-fix":
        return dynamics_fix_configs(rule, from_run)
    raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")


def bench_shapes() -> list[tuple[int, int]]:
    """Smallest, largest and two middle grid shapes, for the throughput benchmark."""
    sizes = sorted({(s.n_params, s.n_layer, s.d_model) for b in GRID_BUDGETS for r in GRID_RATIOS for s in [size_for_budget(b, r)]})
    picks = [sizes[0], sizes[len(sizes) // 3], sizes[2 * len(sizes) // 3], sizes[-1]]
    return [(L, d) for _, L, d in picks]


def throughput_model(bench_path="bench.json"):
    """flops/s as a function of N: log-log interpolation of bench.json, else a constant."""
    p = Path(bench_path)
    if not p.exists():
        return lambda n: DEFAULT_FLOPS_PER_S
    rows = sorted(json.loads(p.read_text()), key=lambda r: r["n_params"])
    xs = np.log([r["n_params"] for r in rows])
    ys = np.log([r["flops_per_s"] for r in rows])
    return lambda n: float(np.exp(np.interp(math.log(n), xs, ys)))


def done_run_ids(results_dir="results") -> set[str]:
    d = Path(results_dir)
    ids = {json.loads(p.read_text())["run_id"] for p in d.glob("*.json")} if d.exists() else set()
    if (d / "failed").exists():
        ids |= {json.loads(p.read_text())["run_id"] for p in (d / "failed").glob("*.json")}
    return ids


def unique(cfgs: list[TrainConfig]) -> list[TrainConfig]:
    seen, out = set(), []
    for c in cfgs:
        if run_id(c) not in seen:
            seen.add(run_id(c))
            out.append(c)
    return out


def print_plan(cfgs: list[TrainConfig], done: set[str], bench_path="bench.json") -> None:
    tput = throughput_model(bench_path)
    cfgs = unique(cfgs)
    total_h = 0.0
    print(f"{'name':40} {'run_id':12} {'L':>3} {'d':>5} {'N':>10} {'tokens':>12} {'lr':>9} {'hours':>6} status")
    for c in cfgs:
        n = param_count(c.n_layer, c.d_model, c.vocab_size, c.qk_norm)
        tokens = c.total_steps * c.batch_tokens
        hours = (6 * n * tokens / tput(n) + RUN_OVERHEAD_S) / 3600
        status = "skip" if run_id(c) in done else "run"
        if status == "run":
            total_h += hours
        print(f"{c.name:40} {run_id(c):12} {c.n_layer:3d} {c.d_model:5d} {n:10,d} {tokens:12,d} {c.peak_lr:9.3g} {hours:6.2f} {status}")
    n_run = sum(run_id(c) not in done for c in cfgs)
    print(f"{len(cfgs)} runs ({n_run} to run): est. {total_h:.2f} GPU-hours, ${total_h * H100_DOLLARS_PER_HOUR:.2f}")


def _load_results(results_dir) -> list[dict]:
    return [json.loads(p.read_text()) for p in Path(results_dir).glob("*.json")]


def best_lrs(cfgs: list[TrainConfig], results: list[dict]) -> list[dict]:
    """Per LR-sweep group: losses aligned with LRs (missing/failed = +inf), best LR, interior flag."""
    by_id = {r["run_id"]: r for r in results}
    groups: dict[tuple[float, float], list[TrainConfig]] = {}
    for c in cfgs:
        groups.setdefault((c.budget_flops, c.tokens_per_param), []).append(c)
    out = []
    for (budget, ratio), members in groups.items():
        members.sort(key=lambda c: c.peak_lr)
        losses = [by_id[run_id(c)]["val_loss"] if run_id(c) in by_id else math.inf for c in members]
        i = int(np.argmin(losses))
        out.append({
            "budget": budget, "ratio": ratio, "n_params": param_count(members[0].n_layer, members[0].d_model),
            "lrs": [c.peak_lr for c in members], "losses": losses, "best_lr": members[i].peak_lr,
            "interior": 0 < i < len(members) - 1,
        })
    order = {(b, r): k for k, (b, r, _, _) in enumerate(LR_SWEEP)}
    return sorted(out, key=lambda g: order[(g["budget"], g["ratio"])])


def fit_lr_rule(groups: list[dict]) -> dict:
    """Line in (ln N, ln LR*) through the two sweep ends; the third group is the check."""
    (small, large, check) = groups[:3]
    x = np.log([small["n_params"], large["n_params"]])
    y = np.log([small["best_lr"], large["best_lr"]])
    c1, c0 = np.polyfit(x, y, 1)
    predicted = math.exp(c0 + c1 * math.log(check["n_params"]))
    return {"c0": float(c0), "c1": float(c1), "check_ratio": predicted / check["best_lr"], "groups": groups}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("phase", choices=PHASES)
    p.add_argument("--lr-rule", default="lr_rule.json")
    p.add_argument("--from-run", default="")
    p.add_argument("--prediction", default="predictions/3e17.json")
    p.add_argument("--results", default="results")
    p.add_argument("--bench", default="bench.json")
    f = sub.add_parser("fit-lr-rule")
    f.add_argument("--results", default="results")
    f.add_argument("--out", default="lr_rule.json")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        cfgs = configs_for_phase(args.phase, args.lr_rule, args.from_run, args.prediction)
        print_plan(cfgs, done_run_ids(args.results), args.bench)
    else:
        groups = best_lrs(lr_configs(), _load_results(args.results))
        for g in groups:
            flag = "ok" if g["interior"] else "EDGE: extend LR_SWEEP centre toward this end and rerun"
            print(f"{g['budget']:.0e} r{g['ratio']:g} N={g['n_params']:,} best_lr={g['best_lr']:.3g} {flag}")
        rule = fit_lr_rule(groups)
        print(f"rule: ln lr = {rule['c0']:.4f} + {rule['c1']:.4f} ln N; check ratio {rule['check_ratio']:.2f} (want 0.5-2)")
        Path(args.out).write_text(json.dumps(rule, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q tests/test_sweep.py`
Expected: all pass. Names format budgets with `:.0e`, so 3e16 renders as `3e+16`. The test uses `grid/3e+16/r20` on purpose.

- [ ] **Step 5: Commit**

```bash
git add sweep.py tests/test_sweep.py
git commit -m "feat: phase run grids, LR rule fit and dry-run cost planner"
```

---

### Task 10: Modal app

**Goal:** All the Modal wiring in one file: volumes, image and secrets; data prep (tokenizer, `.map()` encoding, finalize and manifest); baselines; throughput bench; `train_one`/`train_long` with retries, timeouts and a concurrency cap; result collection; and HF push. Phase entrypoints refuse to launch without `--yes`.

**Files:**
- Create: `modal_app.py`

**Acceptance Criteria:**
- [ ] `uv run python -c "import modal_app"` succeeds (app definition is valid)
- [ ] `grep -rl "^import modal\|^from modal" --include=*.py .` lists only `modal_app.py`
- [ ] Every GPU function sets `gpu="H100"` and an explicit `timeout`; train functions have `retries=modal.Retries(max_retries=3, ...)` and `max_containers=10`
- [ ] `launch` prints the plan and returns without spawning anything unless `--yes`
- [ ] The HF cache, data, checkpoint and inductor cache each live on a `modal.Volume`

**Verify:** `uv run python -c "import modal_app; print('ok')"` → `ok`; `grep -rl "^import modal\|^from modal" --include=*.py . | grep -v .venv` → `./modal_app.py`

**Steps:**

- [ ] **Step 1: Implement `modal_app.py`**

```python
"""Modal wiring. The only file that imports modal.

    uv run modal run modal_app.py::prep_data
    uv run modal run modal_app.py::baselines
    uv run modal run modal_app.py::bench
    uv run modal run --detach modal_app.py::launch --phase grid [--yes]
    uv run modal run modal_app.py::pull_results
    uv run modal run modal_app.py::publish --run-id <id> --repo-id <user/name> [--public]
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import modal

HF_DIR, DATA_DIR, CKPT_DIR, INDUCTOR_DIR = "/vol/hf", "/vol/data", "/vol/ckpt", "/vol/inductor"
DATASET, DATASET_PREFIX = "HuggingFaceFW/fineweb-edu", "sample/10BT/"
N_FILES = 4
TOKENIZER_TRAIN_BYTES = 1_000_000_000
MIN_TRAIN_TOKENS = 2_200_000_000
MIN_VAL_TOKENS = 10_000_000
WANDB_PROJECT = "gptlab-scaling"

app = modal.App("gptlab")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5", "numpy>=1.26", "tokenizers>=0.20", "pyarrow>=16", "huggingface_hub>=0.25", "wandb>=0.18",
        "transformers>=4.45", "safetensors>=0.4", "scipy>=1.13",
    )
    .env({"HF_HOME": HF_DIR, "TORCHINDUCTOR_CACHE_DIR": INDUCTOR_DIR})
    .add_local_python_source("gptlab", "sweep", "analysis")
)
hf_vol = modal.Volume.from_name("gptlab-hf", create_if_missing=True)
data_vol = modal.Volume.from_name("gptlab-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gptlab-ckpt", create_if_missing=True)
inductor_vol = modal.Volume.from_name("gptlab-inductor", create_if_missing=True)


# ---------- data prep (CPU) ----------

def _iter_docs(path: str):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(columns=["id", "text"], batch_size=8192):
        d = batch.to_pydict()
        yield from zip(d["id"], d["text"])


def _download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, filename, repo_type="dataset")
    hf_vol.commit()
    return path


@app.function(image=image, volumes={HF_DIR: hf_vol, DATA_DIR: data_vol}, cpu=16, memory=32768, timeout=3 * 3600)
def train_tokenizer_remote(files: list[str]) -> str:
    from gptlab.tokenizer import is_val_doc, train_tokenizer

    def texts():
        seen = 0
        for f in files:
            for doc_id, text in _iter_docs(_download(f)):
                if is_val_doc(doc_id):
                    continue  # tokenizer never sees val text
                yield text
                seen += len(text.encode("utf-8"))
                if seen >= TOKENIZER_TRAIN_BYTES:
                    return

    tok = train_tokenizer(texts())
    out = Path(DATA_DIR) / "tokenizer.json"
    tok.save(str(out))
    data_vol.commit()
    return str(out)


@app.function(image=image, volumes={HF_DIR: hf_vol, DATA_DIR: data_vol}, cpu=8, memory=65536, timeout=3 * 3600)
def encode_file(filename: str) -> dict:
    from tokenizers import Tokenizer

    from gptlab.tokenizer import encode_file_docs, shard_seed, write_val_shard

    data_vol.reload()
    tok = Tokenizer.from_file(f"{DATA_DIR}/tokenizer.json")
    train, val_docs = encode_file_docs(tok, _iter_docs(_download(filename)), seed=shard_seed(filename))
    shard_dir = Path(DATA_DIR) / "shards"
    shard_dir.mkdir(exist_ok=True)
    stem = Path(filename).stem
    train.tofile(shard_dir / f"train_{stem}.bin")
    write_val_shard(val_docs, shard_dir / f"val_{stem}.bin", shard_dir / f"val_{stem}.json")
    data_vol.commit()
    return {"file": filename, "train_tokens": int(train.size), "val_docs": len(val_docs),
            "val_tokens": int(sum(len(a) for _, a, _ in val_docs))}


@app.function(image=image, volumes={DATA_DIR: data_vol}, memory=32768, timeout=3600)
def finalize_data(files: list[str]) -> dict:
    from gptlab.data import build_manifest, write_manifest
    from gptlab.tokenizer import VAL_PER_MILLE, concat_shards, merge_val_shards

    data_vol.reload()
    shard_dir = Path(DATA_DIR) / "shards"
    stems = [Path(f).stem for f in files]
    n_train = concat_shards([shard_dir / f"train_{s}.bin" for s in stems], Path(DATA_DIR) / "train.bin", seed=0,
                            min_tokens=MIN_TRAIN_TOKENS)
    n_docs, n_tokens, n_bytes = merge_val_shards(
        [(shard_dir / f"val_{s}.bin", shard_dir / f"val_{s}.json") for s in stems], Path(DATA_DIR) / "val.bin", MIN_VAL_TOKENS
    )
    manifest = build_manifest(DATA_DIR, n_docs, n_tokens, n_bytes, extra={
        "dataset": DATASET, "files": files, "train_tokens": n_train, "val_per_mille": VAL_PER_MILLE,
    })
    write_manifest(DATA_DIR, manifest)
    data_vol.commit()
    return manifest


@app.local_entrypoint()
def prep_data(n_files: int = N_FILES):
    from huggingface_hub import list_repo_files

    files = sorted(f for f in list_repo_files(DATASET, repo_type="dataset")
                   if f.startswith(DATASET_PREFIX) and f.endswith(".parquet"))
    assert files, f"no parquet files under {DATASET}/{DATASET_PREFIX}"
    random.Random(0).shuffle(files)
    files = files[:n_files]
    print("files:", files)
    print("tokenizer:", train_tokenizer_remote.remote(files))
    for stats in encode_file.map(files):
        print(stats)
    manifest = finalize_data.remote(files)
    Path("data").mkdir(exist_ok=True)
    Path("data/val_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


# ---------- baselines and benchmark (GPU) ----------

@app.function(image=image, gpu="H100", volumes={DATA_DIR: data_vol, HF_DIR: hf_vol}, memory=32768, timeout=3600)
def run_baselines() -> dict:
    from gptlab.baselines import compute_baselines

    return compute_baselines(DATA_DIR, device="cuda")


@app.local_entrypoint()
def baselines():
    out = run_baselines.remote()
    Path("baselines.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


@app.function(image=image, gpu="H100", volumes={DATA_DIR: data_vol, INDUCTOR_DIR: inductor_vol}, timeout=3600)
def bench_remote(shapes: list[tuple[int, int]]) -> list[dict]:
    from gptlab.train import benchmark

    try:
        return [benchmark(L, d, DATA_DIR) for L, d in shapes]
    finally:
        inductor_vol.commit()


@app.local_entrypoint()
def bench():
    import sweep

    rows = bench_remote.remote(sweep.bench_shapes())
    Path("bench.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"L={r['n_layer']} d={r['d_model']} N={r['n_params']:,} {r['tokens_per_s']:,.0f} tok/s MFU {r['mfu']:.1%}")


# ---------- training (GPU) ----------

def _train_impl(cfg_dict: dict, preempt_at_step: int | None, git_commit: str | None) -> dict:
    from gptlab.train import TrainConfig, WandbLogger, run_id, train

    ckpt_vol.reload()
    cfg = TrainConfig(**cfg_dict)
    run_dir = Path(CKPT_DIR) / "runs" / run_id(cfg)
    try:
        return train(cfg, run_dir, DATA_DIR, logger=WandbLogger(WANDB_PROJECT, cfg), commit=ckpt_vol.commit,
                     preempt_at_step=preempt_at_step, git_commit=git_commit)
    finally:
        inductor_vol.commit()


TRAIN_KW = dict(
    image=image,
    gpu="H100",
    volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol, INDUCTOR_DIR: inductor_vol},
    secrets=[modal.Secret.from_name("wandb")],
    retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0),
    max_containers=10,
)


@app.function(timeout=2 * 3600, **TRAIN_KW)
def train_one(cfg_dict: dict, preempt_at_step: int | None = None, git_commit: str | None = None) -> dict:
    return _train_impl(cfg_dict, preempt_at_step, git_commit)


@app.function(timeout=6 * 3600, **TRAIN_KW)
def train_long(cfg_dict: dict, preempt_at_step: int | None = None, git_commit: str | None = None) -> dict:
    return _train_impl(cfg_dict, preempt_at_step, git_commit)


def _git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


@app.local_entrypoint()
def launch(phase: str, yes: bool = False, from_run: str = "", lr_rule: str = "lr_rule.json",
           prediction: str = "predictions/3e17.json"):
    import sweep
    from gptlab.train import run_id

    cfgs = sweep.unique(sweep.configs_for_phase(phase, lr_rule, from_run, prediction))
    done = sweep.done_run_ids("results")
    sweep.print_plan(cfgs, done)
    todo = [c for c in cfgs if run_id(c) not in done]
    if not yes:
        print("dry run: nothing launched. Re-run with --yes to launch.")
        return
    commit = _git_commit()
    fn = train_long if phase == "test" else train_one
    args = [(asdict(c), sweep.SMOKE_PREEMPT_STEP if c.tag == "preempt" else None, commit) for c in todo]
    for res in fn.starmap(args, return_exceptions=True, order_outputs=False):
        if isinstance(res, BaseException):
            print("FAILED:", repr(res))
        else:
            print(f"done {res['name']} {res['run_id']} val_loss={res['val_loss']:.4f} ${res['cost_usd']:.2f}")


# ---------- results and publishing ----------

@app.function(image=image, volumes={CKPT_DIR: ckpt_vol}, timeout=900)
def collect_results() -> dict:
    ckpt_vol.reload()
    out = {"results": [], "failed": [], "curves": {}}
    for d in sorted((Path(CKPT_DIR) / "runs").glob("*")):
        if (d / "result.json").exists():
            out["results"].append(json.loads((d / "result.json").read_text()))
        elif (d / "FAILED").exists():
            cfg = json.loads((d / "config.json").read_text())
            forensic = json.loads((d / "forensic.json").read_text()) if (d / "forensic.json").exists() else {}
            out["failed"].append({"run_id": d.name, "name": cfg["name"], "phase": cfg["phase"], "config": cfg,
                                  "forensic": forensic})
        if (d / "losses.jsonl").exists():
            out["curves"][d.name] = (d / "losses.jsonl").read_text()
    return out


@app.local_entrypoint()
def pull_results(out: str = "results"):
    root = Path(out)
    (root / "failed").mkdir(parents=True, exist_ok=True)
    (root / "curves").mkdir(exist_ok=True)
    got = collect_results.remote()
    for r in got["results"]:
        (root / f"{r['run_id']}.json").write_text(json.dumps(r, indent=2))
    for f in got["failed"]:
        (root / "failed" / f"{f['run_id']}.json").write_text(json.dumps(f, indent=2))
    for rid, text in got["curves"].items():
        (root / "curves" / f"{rid}.jsonl").write_text(text)
    print(f"{len(got['results'])} results, {len(got['failed'])} failed, {len(got['curves'])} curves -> {root}/")


@app.function(image=image, volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def push_model(run_id: str, repo_id: str, private: bool = True) -> str:
    from huggingface_hub import HfApi

    from analysis.export_hf import export_run

    out = export_run(Path(CKPT_DIR) / "runs" / run_id, Path(DATA_DIR) / "tokenizer.json", Path("/tmp/export") / run_id,
                     repo_id=repo_id)
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


@app.local_entrypoint()
def publish(run_id: str, repo_id: str, public: bool = False):
    print(push_model.remote(run_id, repo_id, private=not public))
```

- [ ] **Step 2: Verify that it imports and that modal stays isolated**

Run: `uv run python -c "import modal_app; print('ok')"`
Expected: `ok`

Run: `grep -rl "^import modal\|^from modal" --include=*.py . | grep -v .venv`
Expected: `./modal_app.py` only

- [ ] **Step 3: Run the full CPU suite (it must not regress)**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add modal_app.py
git commit -m "feat: Modal app with volumes, data prep, baselines, bench, retrying train functions and phase launcher"
```

---

### Task 11: Scaling-law analysis (isoflop, parametric, prediction, costs)

**Goal:** Offline fits from `results/`: isoflop parabolas with bootstrap, power laws, the parametric L(N, D) fit with bootstrap, the pre-registered prediction file with comparison, and the cost ledger. Each is validated on synthetic data with known exponents.

**Files:**
- Create: `analysis/common.py`, `analysis/isoflop.py`, `analysis/parametric.py`, `analysis/predict.py`, `analysis/costs.py`, `tests/test_fits.py`

**Acceptance Criteria:**
- [ ] On a synthetic Chinchilla law (E=1.69, A=406.4, B=410.7, α=0.34, β=0.28; N_opt ≈ 5.8M at 3e15, inside the grid) sampled on our grid, the isoflop `a` is within 0.05 of β/(α+β)=0.452, and every vertex is interior
- [ ] On the same synthetic data, the parametric fit recovers α and β within 0.03 and E within 0.05
- [ ] `seed_sigma` = sqrt(mean(diff²)/2) over the seed pairs
- [ ] `make_prediction` returns isoflop and parametric N_opt with intervals, `declared_config`, and predicted loss for the declared (N, D); the CLI refuses to overwrite an existing prediction file without `--force`
- [ ] `compare` reports the actual loss, the signed error and an `inside_interval` flag for each method
- [ ] `cost_line(r) == "this run: 2.10 GPU-hours, $8.30"` for gpu_hours=2.1, cost_usd=8.296; phase totals include failed runs' GPU time

**Verify:** `uv run pytest -q tests/test_fits.py` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_fits.py`:
```python
import json
import math

import numpy as np
import pytest

from analysis import costs, isoflop, parametric, predict

TRUE = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)  # Hoffmann et al.; optimum lies inside our grid
BUDGETS = (3e15, 1e16, 3e16, 1e17)
RATIOS = (5.0, 10.0, 20.0, 40.0, 80.0)


def law(n, d):
    return TRUE["E"] + TRUE["A"] / n ** TRUE["alpha"] + TRUE["B"] / d ** TRUE["beta"]


def synthetic_results(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for c in BUDGETS:
        for r in RATIOS:
            n = math.sqrt(c / (6 * r))
            d = c / (6 * n)
            for s in (0, 1) if c == BUDGETS[0] else (0,):
                name = f"grid/{c:.0e}/r{r:g}" + ("/seed1" if s else "")
                out.append({"run_id": name, "name": name, "phase": "grid", "budget_flops": c, "n_params": n,
                            "tokens": d, "val_loss": law(n, d) + rng.normal(0, noise), "gpu_hours": 0.1,
                            "cost_usd": 0.4})
    return out


def test_isoflop_recovers_exponent():
    res = synthetic_results(noise=1e-4)
    optima = isoflop.isoflop_optima(isoflop.grid_points(res), sigma=1e-3, n_boot=200)
    assert all(o["interior"] for o in optima)
    exps = isoflop.scaling_exponents(optima)
    target = TRUE["beta"] / (TRUE["alpha"] + TRUE["beta"])
    assert abs(exps["a"] - target) < 0.05
    assert math.isclose(exps["b"], 1 - exps["a"])
    assert exps["a_lo"] <= exps["a"] <= exps["a_hi"]


def test_seed_sigma():
    res = synthetic_results()
    res[1]["val_loss"] += 0.02  # seed1 of the first 3e15 point
    assert math.isclose(isoflop.seed_sigma(res), math.sqrt((0.02**2 / 5) / 2))


def test_parametric_recovers_law():
    res = synthetic_results(noise=1e-5)
    N, D, L = isoflop.grid_runs(res)
    fit = parametric.fit(N, D, L)
    assert abs(fit["alpha"] - TRUE["alpha"]) < 0.03
    assert abs(fit["beta"] - TRUE["beta"]) < 0.03
    assert abs(fit["E"] - TRUE["E"]) < 0.05


def test_prediction_and_compare(tmp_path):
    res = synthetic_results(noise=1e-4)
    rule = (math.log(3e-3) + 0.3 * math.log(2.5e6), -0.3)
    p = predict.make_prediction(res, rule, n_boot=50)
    assert {"isoflop", "parametric", "declared_config", "declared", "predicted_loss"} <= set(p)
    assert p["isoflop"]["n_opt_lo"] <= p["isoflop"]["n_opt"] <= p["isoflop"]["n_opt_hi"]
    out = tmp_path / "pred.json"
    predict.write_prediction(p, out)
    with pytest.raises(FileExistsError):
        predict.write_prediction(p, out)
    actual = {"val_loss": p["predicted_loss"]["parametric"]["value"] + 0.01, "name": "test/3e+17"}
    cmp = predict.compare(p, actual)
    assert math.isclose(cmp["parametric"]["error"], 0.01, abs_tol=1e-9)
    assert "inside_interval" in cmp["parametric"]


def test_costs():
    r = {"gpu_hours": 2.1, "cost_usd": 8.296, "phase": "grid", "name": "x", "run_id": "1"}
    assert costs.cost_line(r) == "this run: 2.10 GPU-hours, $8.30"
    failed = [{"phase": "dynamics", "name": "f", "run_id": "2", "forensic": {"gpu_seconds": 3600.0}}]
    totals = costs.phase_totals([r], failed, dollars_per_hour=4.0)
    assert totals["dynamics"]["gpu_hours"] == 1.0 and totals["dynamics"]["cost_usd"] == 4.0
    assert totals["grid"]["runs"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_fits.py`
Expected: FAIL with `ImportError` for the `analysis` modules

- [ ] **Step 3: Implement `analysis/common.py`**

```python
"""Loading pulled results (written by `modal run modal_app.py::pull_results`)."""

from __future__ import annotations

import json
from pathlib import Path


def load_results(results_dir="results") -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]


def load_failed(results_dir="results") -> list[dict]:
    d = Path(results_dir) / "failed"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def load_curve(results_dir, run_id: str) -> list[dict]:
    """Per-step rows; after a preemption the later copy of a step wins."""
    rows: dict[int, dict] = {}
    for line in (Path(results_dir) / "curves" / f"{run_id}.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["step"]] = r
    return [rows[s] for s in sorted(rows)]
```

- [ ] **Step 4: Implement `analysis/isoflop.py`**

```python
"""IsoFLOP method (Chinchilla approach 2): parabola in ln N per budget, then power laws in C.

    uv run python -m analysis.isoflop results
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.common import load_results


def grid_points(results) -> dict[float, list[tuple[float, float]]]:
    """budget -> sorted [(N, seed-averaged loss)]."""
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("phase") == "grid":
            groups[r["budget_flops"]][r["n_params"]].append(r["val_loss"])
    return {b: sorted((n, float(np.mean(ls))) for n, ls in by_n.items()) for b, by_n in groups.items()}


def grid_runs(results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, D, L) arrays, seed-averaged, for the parametric fit."""
    groups: dict = defaultdict(list)
    for r in results:
        if r.get("phase") == "grid":
            groups[(r["n_params"], r["tokens"])].append(r["val_loss"])
    keys = sorted(groups)
    return (np.array([k[0] for k in keys], float), np.array([k[1] for k in keys], float),
            np.array([np.mean(groups[k]) for k in keys]))


def seed_sigma(results) -> float:
    by_name = {r["name"]: r["val_loss"] for r in results if r.get("phase") == "grid"}
    diffs = [by_name[n] - by_name[n + "/seed1"] for n in by_name if n + "/seed1" in by_name]
    if not diffs:
        raise ValueError("no seed pairs found in grid results")
    return math.sqrt(float(np.mean(np.square(diffs))) / 2)


def _vertex(ln_n, loss):
    c2, c1, c0 = np.polyfit(ln_n, loss, 2)
    if c2 <= 0:
        return math.nan, math.nan, (c2, c1, c0)
    x = -c1 / (2 * c2)
    return x, c0 - c1**2 / (4 * c2), (c2, c1, c0)


def isoflop_optima(points_by_budget, sigma: float, n_boot: int = 1000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for budget in sorted(points_by_budget):
        pts = points_by_budget[budget]
        ln_n = np.log([p[0] for p in pts])
        loss = np.array([p[1] for p in pts])
        x, lmin, coeffs = _vertex(ln_n, loss)
        if math.isnan(x):
            raise ValueError(f"budget {budget:.0e}: loss vs ln N is not convex {coeffs}")
        boots = np.array([_vertex(ln_n, loss + rng.normal(0, sigma, loss.shape))[0] for _ in range(n_boot)])
        n_opt = math.exp(x)
        out.append({
            "budget": budget, "n_opt": n_opt,
            "n_opt_lo": math.exp(np.nanpercentile(boots, 2.5)), "n_opt_hi": math.exp(np.nanpercentile(boots, 97.5)),
            "d_opt": budget / (6 * n_opt), "tokens_per_param": budget / (6 * n_opt**2), "loss_min": lmin,
            "interior": bool(ln_n.min() < x < ln_n.max()), "frac_nonconvex_boot": float(np.isnan(boots).mean()),
            "points": pts, "parabola": [float(c) for c in coeffs], "_ln_n_boot": boots,
        })
    return out


def scaling_exponents(optima: list[dict]) -> dict:
    lnc = np.log([o["budget"] for o in optima])
    a, ln_k = np.polyfit(lnc, np.log([o["n_opt"] for o in optima]), 1)
    boots = np.stack([o["_ln_n_boot"] for o in optima])  # (budgets, n_boot)
    ok = np.all(np.isfinite(boots), axis=0)
    fits = np.array([np.polyfit(lnc, boots[:, j], 1) for j in np.flatnonzero(ok)])
    a_lo, a_hi = np.percentile(fits[:, 0], [2.5, 97.5])
    return {
        "a": float(a), "ln_k": float(ln_k), "a_lo": float(a_lo), "a_hi": float(a_hi),
        "b": float(1 - a), "b_lo": float(1 - a_hi), "b_hi": float(1 - a_lo),
        "note": "b = 1 - a exactly, because D_opt = C / (6 N_opt)",
        "_boot": fits,
    }


def predict_n_opt(exps: dict, budget: float) -> tuple[float, float, float]:
    lnc = math.log(budget)
    samples = np.exp(exps["_boot"][:, 1] + exps["_boot"][:, 0] * lnc)
    return math.exp(exps["ln_k"] + exps["a"] * lnc), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def frontier_predict(optima: list[dict], budget: float) -> float:
    """Straight line in (ln C, ln L_min) through the parabola minima; a point estimate with no floor term."""
    slope, icpt = np.polyfit(np.log([o["budget"] for o in optima]), np.log([o["loss_min"] for o in optima]), 1)
    return float(math.exp(icpt + slope * math.log(budget)))


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main(results_dir="results") -> None:
    res = load_results(results_dir)
    sigma = seed_sigma(res)
    optima = isoflop_optima(grid_points(res), sigma)
    exps = scaling_exponents(optima)
    print(f"seed noise sigma = {sigma:.4f} nats")
    for o in optima:
        print(f"C={o['budget']:.0e} N_opt={o['n_opt']:,.0f} [{o['n_opt_lo']:,.0f}, {o['n_opt_hi']:,.0f}] "
              f"tok/param={o['tokens_per_param']:.1f} L_min={o['loss_min']:.4f} {'ok' if o['interior'] else 'EDGE'}")
    print(f"a = {exps['a']:.3f} [{exps['a_lo']:.3f}, {exps['a_hi']:.3f}]  b = {exps['b']:.3f}  ({exps['note']})")
    out = {"sigma": sigma, "optima": [public(o) for o in optima], "exponents": public(exps)}
    Path(results_dir, "fits").mkdir(exist_ok=True)
    Path(results_dir, "fits", "isoflop.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
```

- [ ] **Step 5: Implement `analysis/parametric.py`**

```python
"""Parametric fit L(N, D) = E + A/N^alpha + B/D^beta (Hoffmann et al. approach 3), with run-level bootstrap.

    uv run python -m analysis.parametric results
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from analysis import isoflop
from analysis.common import load_results

HUBER_DELTA = 1e-3
INIT_GRID = {"a": (0, 5, 10, 15, 20), "b": (0, 5, 10, 15, 20), "e": (-1, 0, 1),
             "alpha": (0.25, 0.5, 1.0, 1.5), "beta": (0.25, 0.5, 1.0, 1.5)}


def _huber(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))


def _objective(theta, ln_n, ln_d, ln_l, delta):
    a, b, e, alpha, beta = theta
    pred = logsumexp(np.stack([a - alpha * ln_n, b - beta * ln_d, np.full_like(ln_n, e)]), axis=0)
    return float(_huber(pred - ln_l, delta).sum())


def _params(theta) -> dict:
    a, b, e, alpha, beta = (float(t) for t in theta)
    return {"E": math.exp(e), "A": math.exp(a), "B": math.exp(b), "alpha": alpha, "beta": beta,
            "a_implied": beta / (alpha + beta), "theta": [a, b, e, alpha, beta]}


def fit(N, D, L, inits=None, delta: float = HUBER_DELTA) -> dict:
    ln_n, ln_d, ln_l = np.log(N), np.log(D), np.log(L)
    inits = inits if inits is not None else list(itertools.product(*INIT_GRID.values()))
    best = None
    for x0 in inits:
        res = minimize(_objective, np.array(x0, float), args=(ln_n, ln_d, ln_l, delta), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    out = _params(best.x)
    out["objective"] = float(best.fun)
    return out


def bootstrap(N, D, L, full: dict, n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """Resample runs with replacement; start each fit from the full fit plus a small grid."""
    rng = np.random.default_rng(seed)
    inits = [tuple(full["theta"])] + list(itertools.product((5, 10), (5, 10), (0,), (0.3, 0.6), (0.3, 0.6)))
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N), len(N))
        out.append(fit(N[idx], D[idx], L[idx], inits=inits))
    return out


def optimal_allocation(p: dict, budget: float) -> tuple[float, float]:
    alpha, beta, A, B = p["alpha"], p["beta"], p["A"], p["B"]
    g = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    n = g * (budget / 6) ** (beta / (alpha + beta))
    return n, budget / (6 * n)


def predicted_loss(p: dict, n: float, d: float) -> float:
    return p["E"] + p["A"] / n ** p["alpha"] + p["B"] / d ** p["beta"]


def interval(values) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(values, float), [2.5, 97.5])
    return float(lo), float(hi)


def main(results_dir="results") -> None:
    N, D, L = isoflop.grid_runs(load_results(results_dir))
    full = fit(N, D, L)
    boots = bootstrap(N, D, L, full)
    summary = {"fit": full}
    for k in ("E", "A", "B", "alpha", "beta", "a_implied"):
        lo, hi = interval([b[k] for b in boots])
        summary[k] = {"value": full[k], "lo": lo, "hi": hi}
        print(f"{k:10} {full[k]:.4g} [{lo:.4g}, {hi:.4g}]")
    Path(results_dir, "fits").mkdir(exist_ok=True)
    Path(results_dir, "fits", "parametric.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
```

- [ ] **Step 6: Implement `analysis/predict.py`**

```python
"""Pre-registered prediction at 3e17, written and committed before the test run, then compared to it.

    uv run python -m analysis.predict make --results results --lr-rule lr_rule.json --out predictions/3e17.json
    uv run python -m analysis.predict compare --prediction predictions/3e17.json --results results
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from analysis import isoflop, parametric
from analysis.common import load_results
from gptlab.flops import param_count

TEST_BUDGET = 3e17


def make_prediction(results, rule, budget: float = TEST_BUDGET, n_boot: int = 1000) -> dict:
    import sweep

    optima = isoflop.isoflop_optima(isoflop.grid_points(results), isoflop.seed_sigma(results), n_boot)
    exps = isoflop.scaling_exponents(optima)
    n_iso, n_lo, n_hi = isoflop.predict_n_opt(exps, budget)
    cfg = sweep.declared_test_config(n_iso, rule, budget)
    n_decl = param_count(cfg.n_layer, cfg.d_model)
    d_decl = cfg.total_steps * cfg.batch_tokens

    N, D, L = isoflop.grid_runs(results)
    full = parametric.fit(N, D, L)
    boots = parametric.bootstrap(N, D, L, full, n_boot=n_boot)
    n_par, d_par = parametric.optimal_allocation(full, budget)
    n_par_lo, n_par_hi = parametric.interval([parametric.optimal_allocation(b, budget)[0] for b in boots])
    loss_lo, loss_hi = parametric.interval([parametric.predicted_loss(b, n_decl, d_decl) for b in boots])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "isoflop": {"n_opt": n_iso, "n_opt_lo": n_lo, "n_opt_hi": n_hi, "d_opt": budget / (6 * n_iso),
                    "a": exps["a"], "a_lo": exps["a_lo"], "a_hi": exps["a_hi"]},
        "parametric": {"n_opt": n_par, "n_opt_lo": n_par_lo, "n_opt_hi": n_par_hi, "d_opt": d_par,
                       "params": {k: full[k] for k in ("E", "A", "B", "alpha", "beta", "a_implied")}},
        "declared_config": asdict(cfg),
        "declared": {"n_params": n_decl, "tokens": d_decl, "rule": "isoflop N_opt snapped to a shape"},
        "predicted_loss": {
            "parametric": {"value": parametric.predicted_loss(full, n_decl, d_decl), "lo": loss_lo, "hi": loss_hi},
            "isoflop_frontier": {"value": isoflop.frontier_predict(optima, budget), "note": "point estimate, no interval"},
        },
    }


def write_prediction(p: dict, path, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; a pre-registered prediction is never overwritten silently")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p, indent=2))


def compare(p: dict, actual: dict) -> dict:
    out = {"actual_val_loss": actual["val_loss"], "run": actual.get("name")}
    for method, pred in p["predicted_loss"].items():
        entry = {"predicted": pred["value"], "error": actual["val_loss"] - pred["value"],
                 "rel_error": (actual["val_loss"] - pred["value"]) / pred["value"]}
        if "lo" in pred:
            entry["inside_interval"] = pred["lo"] <= actual["val_loss"] <= pred["hi"]
        out[method] = entry
    return out


def main(argv=None) -> None:
    import sweep

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--results", default="results")
    m.add_argument("--lr-rule", default="lr_rule.json")
    m.add_argument("--out", default="predictions/3e17.json")
    m.add_argument("--force", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--prediction", default="predictions/3e17.json")
    c.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        p = make_prediction(load_results(args.results), sweep.load_lr_rule(args.lr_rule))
        p["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        write_prediction(p, args.out, args.force)
        print(json.dumps({k: p[k] for k in ("isoflop", "declared", "predicted_loss")}, indent=2))
    else:
        p = json.loads(Path(args.prediction).read_text())
        actual = next(r for r in load_results(args.results) if r.get("phase") == "test")
        out = compare(p, actual)
        Path(args.prediction).with_name(Path(args.prediction).stem + "_outcome.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Implement `analysis/costs.py`**

```python
"""Cost ledger from result.json (and failed runs' forensic GPU time).

    uv run python -m analysis.costs results
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from analysis.common import load_failed, load_results
from gptlab.flops import H100_DOLLARS_PER_HOUR


def cost_line(r: dict) -> str:
    return f"this run: {r['gpu_hours']:.2f} GPU-hours, ${r['cost_usd']:.2f}"


def _rows(results, failed, dollars_per_hour):
    for r in results:
        yield {"run_id": r["run_id"], "name": r["name"], "phase": r["phase"], "status": "ok",
               "gpu_hours": r["gpu_hours"], "cost_usd": r["cost_usd"]}
    for f in failed:
        hours = f.get("forensic", {}).get("gpu_seconds", 0.0) / 3600
        yield {"run_id": f["run_id"], "name": f["name"], "phase": f["phase"], "status": "failed",
               "gpu_hours": hours, "cost_usd": hours * dollars_per_hour}


def phase_totals(results, failed, dollars_per_hour=H100_DOLLARS_PER_HOUR) -> dict:
    tot: dict = defaultdict(lambda: {"runs": 0, "gpu_hours": 0.0, "cost_usd": 0.0})
    for row in _rows(results, failed, dollars_per_hour):
        t = tot[row["phase"]]
        t["runs"] += 1
        t["gpu_hours"] += row["gpu_hours"]
        t["cost_usd"] += row["cost_usd"]
    return dict(tot)


def write_costs_csv(results, failed, path, dollars_per_hour=H100_DOLLARS_PER_HOUR) -> None:
    rows = list(_rows(results, failed, dollars_per_hour))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "name", "phase", "status", "gpu_hours", "cost_usd"])
        w.writeheader()
        w.writerows(rows)


def main(results_dir="results") -> None:
    res, failed = load_results(results_dir), load_failed(results_dir)
    write_costs_csv(res, failed, Path(results_dir) / "costs.csv")
    totals = phase_totals(res, failed)
    for phase, t in sorted(totals.items()):
        print(f"{phase:14} {t['runs']:3d} runs {t['gpu_hours']:7.2f} GPU-h ${t['cost_usd']:8.2f}")
    print(f"{'total':14} {sum(t['runs'] for t in totals.values()):3d} runs "
          f"{sum(t['gpu_hours'] for t in totals.values()):7.2f} GPU-h ${sum(t['cost_usd'] for t in totals.values()):8.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
```

- [ ] **Step 8: Run to verify pass**

Run: `uv run pytest -q tests/test_fits.py`
Expected: all pass. The parametric fit tries 1200 starts, so the test may take ~30–60 s; that's expected. If `test_parametric_recovers_law` misses the tolerance, check first that the synthetic N and D span enough range. Don't loosen the tolerance.

- [ ] **Step 9: Commit**

```bash
git add analysis/common.py analysis/isoflop.py analysis/parametric.py analysis/predict.py analysis/costs.py tests/test_fits.py
git commit -m "feat: isoflop and parametric scaling fits with bootstrap, pre-registered prediction, cost ledger"
```

---

### Task 12: Figures and HF export

**Goal:** Every figure in spec §7.1, and the Llama-format export with a model card, proven logit-equivalent to `gptlab.GPT`.

**Files:**
- Create: `analysis/plots.py`, `analysis/export_hf.py`, `tests/test_plots.py`, `tests/test_export.py`

**Acceptance Criteria:**
- [ ] `to_llama(GPT)` logits match `GPT` logits to atol 1e-4 (fp32, random init, 2 layers, d=128)
- [ ] `to_llama` refuses a model with `qk_norm=True`
- [ ] `model_card(...)` contains YAML front matter with `license: apache-2.0` and `HuggingFaceFW/fineweb-edu`, the cost line, N_total and N_nonembed, tokens, val loss and bpb, and the samples
- [ ] Every plot function writes a non-empty PNG from synthetic inputs (Agg backend)

**Verify:** `uv run pytest -q tests/test_plots.py tests/test_export.py` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/test_export.py`:
```python
import pytest
import torch

from analysis.export_hf import model_card, to_llama
from gptlab.model import GPT, GPTConfig


def test_llama_logits_parity():
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=2, d_model=128, vocab_size=256, seq_len=64)).eval()
    hf = to_llama(m)
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        ours = m(idx)
        theirs = hf(idx).logits
    assert torch.allclose(ours, theirs, atol=1e-4)


def test_refuses_qk_norm():
    m = GPT(GPTConfig(n_layer=1, d_model=64, vocab_size=64, seq_len=16, qk_norm=True))
    with pytest.raises(ValueError):
        to_llama(m)


def test_model_card_contents():
    result = {"name": "test/3e+17", "run_id": "abc", "n_layer": 8, "d_model": 512, "n_params": 50_000_000,
              "n_params_nonembed": 45_000_000, "tokens": 1_000_000_000, "flops": 3e17, "val_loss": 3.1,
              "val_bpb": 1.05, "gpu_hours": 2.1, "cost_usd": 8.296, "wandb_url": "https://wandb.ai/x",
              "git_commit": "deadbeef", "config": {"peak_lr": 1e-3}}
    card = model_card(result, [("The water cycle is", " a process")], repo_id="user/gptlab-50m")
    assert card.startswith("---\n")
    assert "license: apache-2.0" in card and "HuggingFaceFW/fineweb-edu" in card
    assert "this run: 2.10 GPU-hours, $8.30" in card
    assert "50,000,000" in card and "45,000,000" in card and "1.05" in card
    assert "The water cycle is" in card
```

`tests/test_plots.py`:
```python
import math

from analysis import plots


def _optima():
    return [{"budget": c, "n_opt": 1e6 * (c / 3e15) ** 0.5, "n_opt_lo": 0.9e6 * (c / 3e15) ** 0.5,
             "n_opt_hi": 1.1e6 * (c / 3e15) ** 0.5, "d_opt": c / (6e6 * (c / 3e15) ** 0.5),
             "tokens_per_param": 20.0, "loss_min": 4 - 0.1 * math.log10(c / 3e15),
             "points": [(n, 4 + (math.log(n) - math.log(1e6)) ** 2 * 0.05) for n in (5e5, 7e5, 1e6, 1.4e6, 2e6)],
             "parabola": [0.05, 0.0, 4.0], "interior": True} for c in (3e15, 1e16, 3e16, 1e17)]


def test_plots_write_files(tmp_path):
    optima = _optima()
    exps = {"a": 0.5, "ln_k": math.log(1e6) - 0.5 * math.log(3e15), "b": 0.5}
    paths = [
        plots.plot_isoflop(optima, tmp_path / "isoflop.png"),
        plots.plot_optima(optima, exps, tmp_path / "optima.png", predicted=(3e17, 1e7), actual=(3e17, 1.1e7)),
        plots.plot_frontier(optima, tmp_path / "frontier.png", predicted=(3e17, 3.7), actual=(3e17, 3.72)),
        plots.plot_tokens_per_param(optima, tmp_path / "tpp.png"),
        plots.plot_lr_sweep([{"budget": 3e15, "ratio": 80.0, "n_params": 2.5e6, "lrs": [1e-3, 2e-3, 4e-3],
                              "losses": [4.1, 4.0, float("inf")], "best_lr": 2e-3}], tmp_path / "lr.png"),
        plots.plot_curves({"a": [{"step": s, "loss": 5 - s / 100, "grad_norm": 1.0} for s in range(100)],
                           "b": [{"step": s, "loss": 5 - s / 120, "grad_norm": 2.0} for s in range(100)]},
                          "test", tmp_path / "curves.png"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_plots.py tests/test_export.py`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `analysis/export_hf.py`**

```python
"""Export gptlab.GPT checkpoints as LlamaForCausalLM (tied embeddings) with a model card."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from analysis.costs import cost_line
from gptlab.flops import ffn_hidden
from gptlab.model import GPT, GPTConfig
from gptlab.tokenizer import EOT
from gptlab.train import SAMPLE_PROMPTS


def llama_config(cfg: GPTConfig):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=ffn_hidden(cfg.d_model),
        num_hidden_layers=cfg.n_layer, num_attention_heads=cfg.n_head, num_key_value_heads=cfg.n_head,
        max_position_embeddings=cfg.seq_len, rms_norm_eps=cfg.norm_eps, rope_theta=cfg.rope_base,
        tie_word_embeddings=True, attention_bias=False, mlp_bias=False, hidden_act="silu",
        bos_token_id=0, eos_token_id=0,
    )


def to_llama(model: GPT):
    from transformers import LlamaForCausalLM

    if model.cfg.qk_norm:
        raise ValueError("qk_norm models have no LlamaForCausalLM equivalent; export grid/test models only")
    hf = LlamaForCausalLM(llama_config(model.cfg))
    sd = {f"model.{k}": v for k, v in model.state_dict().items()}
    sd["lm_head.weight"] = model.embed_tokens.weight
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not [m for m in missing if "rotary" not in m], missing
    hf.tie_weights()
    return hf.eval()


def model_card(result: dict, samples: list[tuple[str, str]], repo_id: str) -> str:
    sample_md = "\n".join(f"- **{p}**{c}" for p, c in samples)
    return f"""---
license: apache-2.0
language: [en]
library_name: transformers
pipeline_tag: text-generation
datasets: [HuggingFaceFW/fineweb-edu]
tags: [scaling-laws, from-scratch, pytorch, llama]
---

# {repo_id}

A {result['n_params']:,}-parameter Llama-style model trained from scratch in plain PyTorch as part of a
compute-optimal scaling-law study (run `{result['name']}`).

**Cost:** {cost_line(result)}

| | |
|---|---|
| Parameters (total, tied embeddings) | {result['n_params']:,} |
| Parameters (non-embedding) | {result['n_params_nonembed']:,} |
| Layers × width | {result['n_layer']} × {result['d_model']} |
| Training tokens | {result['tokens']:,} |
| Training compute | {result['flops']:.3g} FLOPs (6·N·D) |
| Val loss (nats/token) | {result['val_loss']:.4f} |
| Val bits per byte | {result['val_bpb']:.4f} |
| Tokenizer | own 8192-token byte-level BPE |
| Context | 512 tokens |

Training data: FineWeb-Edu `sample-10BT` (ODC-By), document-hash train/val split.
W&B run: {result.get('wandb_url')} · code commit: `{result.get('git_commit')}`

## Samples (fixed prompts, temperature 0.8, top-k 50)

{sample_md}

## Intended use and limitations

Research and teaching only. Small, English-only, trained on web text with FineWeb-Edu's quality filtering and
nothing else: no instruction tuning, no safety tuning. It will produce false and incoherent text.
"""


def export_run(run_dir: Path, tokenizer_path: Path, out_dir: Path, repo_id: str) -> Path:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    result = json.loads((run_dir / "result.json").read_text())
    c = result["config"]
    model = GPT(GPTConfig(n_layer=c["n_layer"], d_model=c["d_model"], vocab_size=c["vocab_size"], seq_len=c["seq_len"],
                          qk_norm=c["qk_norm"]))
    model.load_state_dict(torch.load(run_dir / "final.pt", map_location="cpu"))
    model.eval()
    tok = Tokenizer.from_file(str(tokenizer_path))
    gen = torch.Generator().manual_seed(1234)
    samples = []
    for prompt in SAMPLE_PROMPTS:
        ids = torch.tensor([tok.encode(prompt).ids])
        samples.append((prompt, tok.decode(model.generate(ids, 48, generator=gen)[0, ids.size(1):].tolist())))
    out_dir.mkdir(parents=True, exist_ok=True)
    to_llama(model).save_pretrained(out_dir, safe_serialization=True)
    PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), bos_token=EOT, eos_token=EOT,
                            model_max_length=c["seq_len"]).save_pretrained(out_dir)
    (out_dir / "README.md").write_text(model_card(result, samples, repo_id))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return out_dir
```

- [ ] **Step 4: Implement `analysis/plots.py`**

```python
"""All figures. Every function takes plain data and an output path, and returns the path.

    uv run python -m analysis.plots results figures
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_isoflop(optima, path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for o in optima:
        n = np.array([p[0] for p in o["points"]])
        line = ax.plot(n, [p[1] for p in o["points"]], "o", label=f"C = {o['budget']:.0e}")[0]
        xs = np.linspace(np.log(n.min()) - 0.2, np.log(n.max()) + 0.2, 100)
        ax.plot(np.exp(xs), np.polyval(o["parabola"], xs), "-", color=line.get_color(), alpha=0.6)
        ax.plot(o["n_opt"], o["loss_min"], "*", color=line.get_color(), markersize=12)
    ax.set(xscale="log", xlabel="parameters N (total)", ylabel="final val loss (nats/token)", title="IsoFLOP profiles")
    ax.legend()
    return _save(fig, path)


def plot_optima(optima, exps, path, predicted=None, actual=None) -> Path:
    c = np.array([o["budget"] for o in optima])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cs = np.logspace(np.log10(c.min()) - 0.2, np.log10(max(c.max(), predicted[0] if predicted else 0)) + 0.2, 50)
    n_fit = np.exp(exps["ln_k"] + exps["a"] * np.log(cs))
    for ax, key, fitted, label, exp in (
        (axes[0], "n_opt", n_fit, "N_opt", exps["a"]),
        (axes[1], "d_opt", cs / (6 * n_fit), "D_opt", exps["b"]),
    ):
        y = np.array([o[key] for o in optima])
        if key == "n_opt":
            ax.errorbar(c, y, yerr=[y - [o["n_opt_lo"] for o in optima], [o["n_opt_hi"] for o in optima] - y], fmt="o")
        else:
            ax.plot(c, y, "o")
        ax.plot(cs, fitted, "-", label=f"fit: exponent {exp:.3f}")
        ref = fitted[0] * (cs / cs[0]) ** 0.5
        ax.plot(cs, ref, ":", color="gray", label="Chinchilla slope 0.5")
        if predicted and key == "n_opt":
            ax.plot(*predicted, "s", mfc="none", markersize=10, label="predicted 3e17")
        if actual and key == "n_opt":
            ax.plot(*actual, "s", markersize=7, label="trained 3e17")
        ax.set(xscale="log", yscale="log", xlabel="compute C (FLOPs)", ylabel=label)
        ax.legend(fontsize=8)
    return _save(fig, path)


def plot_frontier(optima, path, predicted=None, actual=None) -> Path:
    c = np.array([o["budget"] for o in optima])
    lmin = np.array([o["loss_min"] for o in optima])
    slope, icpt = np.polyfit(np.log(c), np.log(lmin), 1)
    cs = np.logspace(np.log10(c.min()) - 0.2, np.log10(max(c.max(), predicted[0] if predicted else 0)) + 0.2, 50)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(c, lmin, "o", label="isoflop minima")
    ax.plot(cs, np.exp(icpt + slope * np.log(cs)), "-", label=f"L ∝ C^{slope:.3f}")
    if predicted:
        ax.plot(*predicted, "s", mfc="none", markersize=10, label="predicted 3e17")
    if actual:
        ax.plot(*actual, "s", markersize=7, label="trained 3e17")
    ax.set(xscale="log", yscale="log", xlabel="compute C (FLOPs)", ylabel="best val loss", title="Compute frontier")
    ax.legend()
    return _save(fig, path)


def plot_tokens_per_param(optima, path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([o["budget"] for o in optima], [o["tokens_per_param"] for o in optima], "o-", label="measured optimum")
    ax.axhline(20, color="gray", ls=":", label="Chinchilla ≈ 20")
    ax.set(xscale="log", xlabel="compute C (FLOPs)", ylabel="D_opt / N_opt", title="Tokens per parameter at optimum")
    ax.legend()
    return _save(fig, path)


def plot_lr_sweep(groups, path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for g in groups:
        losses = [l if math.isfinite(l) else np.nan for l in g["losses"]]
        line = ax.plot(g["lrs"], losses, "o-", label=f"N={g['n_params']:,.0f} ({g['budget']:.0e})")[0]
        for lr, l in zip(g["lrs"], g["losses"]):
            if not math.isfinite(l):
                ax.plot(lr, np.nanmax(losses) if np.isfinite(np.nanmax(losses)) else 0, "x", color=line.get_color())
        ax.axvline(g["best_lr"], color=line.get_color(), ls=":", alpha=0.6)
    ax.set(xscale="log", xlabel="peak LR", ylabel="final val loss", title="LR sweep (x = diverged)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_curves(curves: dict[str, list[dict]], title: str, path, metric: str = "loss", ylim=None) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name, rows in curves.items():
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r[metric] for r in rows], label=name, lw=1)
        axes[1].plot(steps, [r["grad_norm"] for r in rows], label=name, lw=1)
    axes[0].set(ylabel=metric, title=title)
    if ylim:
        axes[0].set_ylim(*ylim)
    axes[1].set(ylabel="grad norm (pre-clip)", xlabel="step", yscale="log")
    axes[0].legend(fontsize=8)
    return _save(fig, path)


def main(results_dir="results", out_dir="figures") -> None:
    fits = Path(results_dir, "fits", "isoflop.json")
    if fits.exists():
        iso = json.loads(fits.read_text())
        optima, exps = iso["optima"], iso["exponents"]
        pred_path, outcome_path = Path("predictions/3e17.json"), Path("predictions/3e17_outcome.json")
        pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
        outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else None
        p_n = (pred["budget"], pred["isoflop"]["n_opt"]) if pred else None
        a_n = (pred["budget"], pred["declared"]["n_params"]) if pred and outcome else None
        p_l = (pred["budget"], pred["predicted_loss"]["isoflop_frontier"]["value"]) if pred else None
        a_l = (pred["budget"], outcome["actual_val_loss"]) if outcome else None
        plot_isoflop(optima, Path(out_dir, "isoflop.png"))
        plot_optima(optima, exps, Path(out_dir, "optima.png"), predicted=p_n, actual=a_n)
        plot_frontier(optima, Path(out_dir, "frontier.png"), predicted=p_l, actual=a_l)
        plot_tokens_per_param(optima, Path(out_dir, "tokens_per_param.png"))
    rule = Path("lr_rule.json")
    if rule.exists():
        plot_lr_sweep(json.loads(rule.read_text())["groups"], Path(out_dir, "lr_sweep.png"))
    print(f"figures written to {out_dir}/")


if __name__ == "__main__":
    main(*sys.argv[1:])
```

Dynamics figures are built in Task 15 with `plot_curves`, since which runs go on which figure depends on what diverged.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest -q tests/test_plots.py tests/test_export.py`
Expected: all pass. If the parity test fails, compare layer by layer before changing anything. Likely causes are the RoPE convention, the RMSNorm eps, or `intermediate_size`. Fix `gptlab/model.py` only if it breaks the HF convention stated in Global Constraints.

- [ ] **Step 6: Full suite and commit**

Run: `uv run pytest -q`
Expected: all pass

```bash
git add analysis/plots.py analysis/export_hf.py tests/test_plots.py tests/test_export.py
git commit -m "feat: scaling-law figures and Llama-format HF export with model card"
```

---

### Task 13: Phase 0–1 on Modal — setup, data prep, baselines, smoke test, bench

**Goal:** Set the spend limit and secrets, freeze the val split, train the tokenizer, encode the data, commit the manifest and baselines, prove preemption-resume on an H100, and record throughput per shape.

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Create (outputs): `data/val_manifest.json`, `baselines.json`, `bench.json`
- Modify: `NOTES.md`

**Acceptance Criteria:**
- [ ] The user confirms in chat that the Modal workspace spend limit is set to $60 **before** any GPU function runs
- [ ] `modal secret list` shows `wandb` and `huggingface`
- [ ] `data/val_manifest.json` is committed with `n_tokens ≥ 10,000,000`, `train_tokens ≥ 2,200,000,000` and `val_per_mille: 5`, and it's identical to `/vol/data/val_manifest.json`
- [ ] `baselines.json` is committed before any training run, and bigram loss < unigram loss < uniform loss (9.011)
- [ ] Smoke: `results/` holds `smoke/a` and `smoke/preempt`; the preempt run's curve has every step 0–199 exactly once; max |loss_preempt − loss_a| over steps 100–199 is < 0.05; there's one W&B run per smoke config
- [ ] `bench.json` has 4 rows with `tokens_per_s` and `mfu`; `python sweep.py plan grid` uses it

**Verify:** `uv run modal run modal_app.py::pull_results` → then run the Step 6 check script → prints `SMOKE OK max diff …`

**Steps:**

- [ ] **Step 1: Have the user set the spend limit and secrets (user actions; ask and wait)**

Ask the user to:
1. Run `! uv run modal setup`, if this machine isn't authenticated yet.
2. Set the workspace spend limit to **$60** in the Modal dashboard (Settings → Usage & Billing), and confirm in chat.
3. Create the secrets, keeping the values out of the chat transcript:
   `! uv run modal secret create wandb WANDB_API_KEY=<key>`
   `! uv run modal secret create huggingface HF_TOKEN=<token with write scope>`

Then run: `uv run modal secret list`
Expected: rows for `wandb` and `huggingface`

- [ ] **Step 2: Prep the data (CPU only; this also freezes the split)**

Run: `uv run modal run modal_app.py::prep_data`
Expected: 4 `encode_file` stat lines, then a manifest with `n_tokens ≥ 10000000` and `train_tokens ≥ 2200000000`. If `InsufficientData` is raised, re-run with `--n-files 5` and log it in `NOTES.md`.
Don't print or inspect `val.bin` or val text at any point.

- [ ] **Step 3: Commit the manifest**

```bash
git add data/val_manifest.json
git commit -m "data: freeze FineWeb-Edu val split (manifest)"
```

- [ ] **Step 4: Baselines (first GPU use; the spend limit must already be confirmed)**

Run: `uv run modal run modal_app.py::baselines`
Expected: JSON with `uniform.loss_nats ≈ 9.011`, unigram and bigram below it in that order, and `gpt2_small.bpb` roughly in 0.9–1.3. Record the numbers in `NOTES.md`.

```bash
git add baselines.json NOTES.md
git commit -m "data: baselines measured before training"
```

- [ ] **Step 5: Smoke test with simulated preemption**

Run: `uv run python sweep.py plan smoke`, then `uv run modal run --detach modal_app.py::launch --phase smoke --yes`
Expected: `done smoke/a ...` and `done smoke/preempt ...`. The preempt run raises `SimulatedPreemption` once, and Modal retries it automatically. `FAILED:` lines aren't acceptable.

- [ ] **Step 6: Check the smoke result**

Run:
```bash
uv run modal run modal_app.py::pull_results
uv run python - <<'EOF'
import json, glob
from analysis.common import load_curve, load_results
res = {r["name"]: r for r in load_results("results")}
a, b = res["smoke/a"], res["smoke/preempt"]
ca, cb = load_curve("results", a["run_id"]), load_curve("results", b["run_id"])
assert [r["step"] for r in cb] == list(range(200)), "steps missing or duplicated"
lines = open(f"results/curves/{b['run_id']}.jsonl").read().splitlines()
assert len(lines) == 200, f"{len(lines)} lines: resume replayed or skipped steps"
diff = max(abs(x["loss"] - y["loss"]) for x, y in zip(ca[100:], cb[100:]))
assert diff < 0.05, diff
print("SMOKE OK", f"max diff {diff:.4f}", "costs", a["cost_usd"], b["cost_usd"])
EOF
```
Expected: `SMOKE OK …`. Also open the W&B project and confirm there's one run per smoke config, and that `smoke/preempt` has no gap at step 100.

- [ ] **Step 7: Throughput bench**

Run: `uv run modal run modal_app.py::bench`
Expected: 4 lines with tok/s and MFU. Then run `uv run python sweep.py plan grid --lr-rule <(echo '{"c0": -2.0, "c1": -0.25}')` and check that the cost estimate uses the measured throughput. If the grid estimate exceeds $25, stop and tell the user before Phase 2.

- [ ] **Step 8: Commit**

```bash
git add bench.json results/ NOTES.md
git commit -m "phase1: smoke test with preemption-resume, throughput bench"
```

---

### Task 14: Phase 2 on Modal — LR sweep and LR rule

**Goal:** Sweep peak LR at the small and large ends (plus a mid check), confirm each optimum is interior, and commit `lr_rule.json`.

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Create (outputs): `lr_rule.json`, `results/*.json`, `figures/lr_sweep.png`
- Modify: `sweep.py` (`LR_SWEEP` centres, only if an optimum lands on an edge), `NOTES.md`

**Acceptance Criteria:**
- [ ] The user approves the printed dry-run plan and cost before `--yes`
- [ ] `sweep.py fit-lr-rule` prints `ok` (interior) for all three groups
- [ ] `check_ratio` for the 9M group is within 0.5–2.0, or the deviation is written up in `NOTES.md` and the user accepts it
- [ ] `lr_rule.json` and `figures/lr_sweep.png` are committed

**Verify:** `uv run python sweep.py fit-lr-rule --results results` → three `ok` lines and `check ratio` within 0.5–2

**Steps:**

- [ ] **Step 1: Dry run, and get the user's approval**

Run: `uv run python sweep.py plan lr`
Expected: 13 runs and a dollar estimate. Show it to the user and wait for an explicit go.

- [ ] **Step 2: Launch**

Run: `uv run modal run --detach modal_app.py::launch --phase lr --yes`
Expected: 13 `done`/`FAILED` lines. A `FAILED` with `NonFiniteLoss` at the highest LR is a legitimate data point (it counts as +inf loss). Any other exception needs to be diagnosed before going further.

- [ ] **Step 3: Pull and fit**

Run: `uv run modal run modal_app.py::pull_results && uv run python sweep.py fit-lr-rule --results results`
Expected: three groups, each `ok`. If a group prints `EDGE`: move that group's centre in `LR_SWEEP` by ×4 toward the edge, then re-run Steps 1–3. Already-finished LRs are skipped by run_id. Log it in `NOTES.md`.

- [ ] **Step 4: Plot and commit**

Run: `uv run python -m analysis.plots results figures`
Expected: `figures/lr_sweep.png` exists

```bash
git add lr_rule.json figures/lr_sweep.png results/ sweep.py NOTES.md
git commit -m "phase2: LR sweep and fitted LR rule"
```

---

### Task 15: Phases 3–4 on Modal — IsoFLOP grid and training-dynamics track

**Goal:** Run the 25-run grid and the dynamics track. Confirm that every budget's minimum is interior, then provoke instability, diagnose it and fix it, with a figure and a written diagnosis for each experiment.

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Create (outputs): `results/…`, `results/fits/isoflop.json`, `figures/dyn_instability.png`, `figures/dyn_warmup.png`, `figures/dyn_beta2.png`, `figures/dyn_batch.png`
- Modify: `sweep.py` (`GRID_EXTRA_RATIOS`, only if a gate fails), `NOTES.md`

**Acceptance Criteria:**
- [ ] The user approves the grid and dynamics dry-run costs before `--yes`
- [ ] All 25 grid runs have results (no FAILED)
- [ ] `python -m analysis.isoflop results` prints `ok` (interior) for all 4 budgets, after any `GRID_EXTRA_RATIOS` additions
- [ ] All 12 dynamics runs are accounted for (result or FAILED with forensics); the diverging config is identified, and its `+qknorm` and `+zloss` fixes have been run
- [ ] Four dynamics figures are committed, and `NOTES.md` has a dated, one-paragraph diagnosis per experiment (instability: which layer's grad norm or attention max-logit blew up first, and at what step)

**Verify:** `uv run python -m analysis.isoflop results` → 4 lines ending `ok`; `ls figures/dyn_*.png | wc -l` → `4`

**Steps:**

- [ ] **Step 1: Grid dry run and approval**

Run: `uv run python sweep.py plan grid`
Expected: 25 runs. Show the cost to the user and wait for a go.

- [ ] **Step 2: Launch the grid**

Run: `uv run modal run --detach modal_app.py::launch --phase grid --yes`
Expected: 25 `done` lines

- [ ] **Step 3: Dynamics dry run, approval and launch**

Run: `uv run python sweep.py plan dynamics`
Expected: 12 runs; `dyn/baseline` shows `skip` once the grid's `grid/3e+16/r20` result has been pulled.
After approval: `uv run modal run --detach modal_app.py::launch --phase dynamics --yes`

- [ ] **Step 4: Pull and check the isoflop gate**

Run: `uv run modal run modal_app.py::pull_results && uv run python -m analysis.isoflop results`
Expected: 4 budgets, each `ok`. If a budget prints `EDGE`: add `{budget: (2.5,)}` (minimum at the large-N edge) or `{budget: (160.0,)}` (small-N edge) to `GRID_EXTRA_RATIOS`, then re-run Steps 1–2 and 4. Log it in `NOTES.md`.

- [ ] **Step 5: Diagnose the instability, then run the fixes**

Look at `results/failed/*.json` and the `dyn/lr*` curves. Find the config that diverged (NaN) or spiked worst. If none diverged, use `dyn/lr10x-noclip`.
Use the W&B `grad_norm/layer_*` and `attn_max_logit/layer_*` panels, and `forensic.json`, to work out which layer went first and when. Write the diagnosis in `NOTES.md`.
Then run: `uv run modal run --detach modal_app.py::launch --phase dynamics-fix --from-run <that config name> --yes`, after showing the plan to the user.

- [ ] **Step 6: Build the dynamics figures**

Run:
```bash
uv run modal run modal_app.py::pull_results
uv run python - <<'EOF'
import json
from analysis.common import load_curve, load_results, load_failed
from analysis.plots import plot_curves
rows = load_results("results") + load_failed("results")
ids = {r["name"]: r["run_id"] for r in rows}
ids.setdefault("dyn/baseline", ids["grid/3e+16/r20"])
def curves(names):
    return {n: load_curve("results", ids[n]) for n in names if n in ids}
inst = [n for n in ids if n.startswith("dyn/lr")] + ["dyn/baseline"]
plot_curves(curves(inst), "Instability: LR multiples, clipping, QK-norm, z-loss", "figures/dyn_instability.png", ylim=(2, 10))
plot_curves(curves(["dyn/baseline", "dyn/warmup0", "dyn/warmup1"]), "Warmup 5% / 0% / 1%", "figures/dyn_warmup.png")
plot_curves(curves(["dyn/baseline", "dyn/beta2-0.999"]), "beta2 0.95 vs 0.999", "figures/dyn_beta2.png")
plot_curves(curves(["dyn/batch32k", "dyn/baseline", "dyn/batch128k", "dyn/batch256k", "dyn/batch512k"]),
            "Batch size at equal FLOPs (x-axis: steps)", "figures/dyn_batch.png")
print("ok")
EOF
```
Expected: `ok`, and 4 PNGs. Add a final-loss-vs-batch-size note for each run to `NOTES.md`. Remember the step axis differs between batch sizes, so compare final loss, not curves.

- [ ] **Step 7: Commit**

```bash
git add results/ figures/ sweep.py NOTES.md
git commit -m "phase3-4: isoflop grid and training-dynamics track with diagnoses"
```

---

### Task 16: Phase 5 — fit, then commit and tag the prediction before the test run

**Goal:** Fit both scaling methods, write the 3e17 prediction (best size, tokens and loss, with intervals) to `predictions/3e17.json`, and commit and tag it **before** anything trains at 3e17.

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Create (outputs): `results/fits/isoflop.json`, `results/fits/parametric.json`, `predictions/3e17.json`, `figures/isoflop.png`, `figures/optima.png`, `figures/frontier.png`, `figures/tokens_per_param.png`, `results/costs.csv`
- Modify: `NOTES.md`

**Acceptance Criteria:**
- [ ] `predictions/3e17.json` contains `isoflop.n_opt` with an interval, `parametric.n_opt` with an interval, `declared_config`, and `predicted_loss.parametric` with lo/hi
- [ ] The prediction is committed and tagged `prediction-3e17`; `git log -1 --format=%cI prediction-3e17` predates any `test` phase result
- [ ] No `phase == "test"` result exists in `results/` at tag time
- [ ] `NOTES.md` records the isoflop `a` and the parametric `a_implied` side by side; if their intervals don't overlap, that's stated as a finding

**Verify:** `git tag -l prediction-3e17` → `prediction-3e17`; `uv run python -c "import json;p=json.load(open('predictions/3e17.json'));print(p['declared'],p['predicted_loss'])"` prints both

**Steps:**

- [ ] **Step 1: Fit both methods**

Run: `uv run python -m analysis.isoflop results && uv run python -m analysis.parametric results`
Expected: the isoflop table plus `a`, and parametric parameters with intervals, including `a_implied`. Record both values of `a` in `NOTES.md`.

- [ ] **Step 2: Confirm there's no test result yet**

Run: `uv run python -c "from analysis.common import load_results; assert not [r for r in load_results('results') if r['phase']=='test']; print('clean')"`
Expected: `clean`

- [ ] **Step 3: Write the prediction**

Run: `uv run python -m analysis.predict make --results results --lr-rule lr_rule.json --out predictions/3e17.json`
Expected: JSON showing the isoflop N_opt, the declared n_params/tokens, and predicted losses. Show it to the user.

- [ ] **Step 4: Figures and costs**

Run: `uv run python -m analysis.plots results figures && uv run python -m analysis.costs results`
Expected: 4 scaling figures plus the LR figure, and a per-phase cost table

- [ ] **Step 5: Commit and tag**

```bash
git add predictions/3e17.json results/ figures/ NOTES.md
git commit -m "phase5: pre-registered 3e17 prediction from isoflop and parametric fits"
git tag -a prediction-3e17 -m "Prediction committed before the 3e17 run"
```

---

### Task 17: Phase 6 on Modal — the 3e17 test run and the comparison

**Goal:** Train the declared 3e17 model and report how far the actual loss falls from each prediction.

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Create (outputs): `predictions/3e17_outcome.json`, `results/…`, updated figures
- Modify: `NOTES.md`

**Acceptance Criteria:**
- [ ] The tag `prediction-3e17` exists before launch, and the run's config equals `declared_config` (same run_id)
- [ ] The user approves the dry-run cost before `--yes`
- [ ] The test run finishes with a result (on `train_long`, 6h timeout)
- [ ] `predictions/3e17_outcome.json` reports the actual loss, the error against each method, and `inside_interval` for the parametric method
- [ ] `figures/optima.png` and `figures/frontier.png` show both the predicted and the actual point

**Verify:** `cat predictions/3e17_outcome.json` shows `actual_val_loss` and per-method `error`

**Steps:**

- [ ] **Step 1: Pre-launch checks and approval**

Run: `git tag -l prediction-3e17 && uv run python sweep.py plan test`
Expected: the tag is listed, and 1 run is shown with its estimated hours and $. Get the user's go.

- [ ] **Step 2: Launch**

Run: `uv run modal run --detach modal_app.py::launch --phase test --yes`
Expected: `done test/3e+17 …`

- [ ] **Step 3: Compare, plot and commit**

Run:
```bash
uv run modal run modal_app.py::pull_results
uv run python -m analysis.predict compare --prediction predictions/3e17.json --results results
uv run python -m analysis.plots results figures
uv run python -m analysis.costs results
```
Expected: the outcome JSON is printed, and the figures are regenerated with actual points

```bash
git add predictions/3e17_outcome.json results/ figures/ NOTES.md
git commit -m "phase6: 3e17 test run vs pre-registered prediction"
```

---

### Task 18: Phase 7 — publish models, W&B report and README

**Goal:** Push the models with model cards, get the W&B report public, and write the README with every section in spec §9.2, including the cost lines and "What didn't work".

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Create: `README.md`
- Modify: `NOTES.md`

**Acceptance Criteria:**
- [ ] The user picks the HF namespace/repo names and approves making each repo public (pushes are private until then)
- [ ] 5 models pushed: the 3e17 model, plus the grid model nearest N_opt at each of the 4 budgets. Each loads with `AutoModelForCausalLM.from_pretrained(repo)` and generates text
- [ ] Each model card contains its cost line
- [ ] The README has these sections, in order: TL;DR (a, b, tokens/param, prediction vs actual, total cost) · What happens in a training step (links to `gptlab/train.py` lines) · Data and evaluation protocol plus the baselines table · Method · Scaling results (both methods, disagreement, prediction test) · Training dynamics · What didn't work (from `NOTES.md`) · Cost (per-phase table plus per-run lines) · Reproduce · Limitations and next steps (WSD, μP, more budgets and seeds)
- [ ] The W&B report URL is in the README, and the user confirms the report is public

**Verify:** `grep -c "this run:" README.md` ≥ 5; `uv run python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('<repo>')"` succeeds for each repo

**Steps:**

- [ ] **Step 1: Choose models, and get repo names from the user**

Run:
```bash
uv run python - <<'EOF'
import json
iso = json.load(open("results/fits/isoflop.json"))
from analysis.common import load_results
res = [r for r in load_results("results") if r["phase"] == "grid" and r["config"]["seed"] == 0]
for o in iso["optima"]:
    best = min((r for r in res if r["budget_flops"] == o["budget"]), key=lambda r: abs(r["n_params"] - o["n_opt"]))
    print(o["budget"], best["name"], best["run_id"])
t = [r for r in load_results("results") if r["phase"] == "test"][0]
print("test", t["name"], t["run_id"])
EOF
```
Expected: 5 run_ids. Ask the user for the HF namespace and repo names.

- [ ] **Step 2: Push privately**

For each run_id, run: `uv run modal run modal_app.py::publish --run-id <id> --repo-id <namespace/name>`
Expected: a URL for each. Check the load: `uv run python -c "from transformers import AutoModelForCausalLM, AutoTokenizer as T; r='<repo>'; m=AutoModelForCausalLM.from_pretrained(r); t=T.from_pretrained(r); print(t.decode(m.generate(**t('The water cycle is', return_tensors='pt'), max_new_tokens=20)[0]))"`. For private repos this needs `HF_TOKEN` in the local env.

- [ ] **Step 3: W&B report (user action)**

Ask the user to create a W&B report in project `gptlab-scaling` (UI: Reports → Create). It should contain panels for the grid loss curves, the LR sweep and the dynamics experiments, and be set to public. Ask them to paste the URL.

- [ ] **Step 4: Write `README.md`**

Write the README with the sections listed in the acceptance criteria. Pull the numbers from `results/fits/*.json`, `predictions/3e17*.json`, `baselines.json` and `results/costs.csv`; never type them by hand. Give each table row for a run its `cost_line`. For the training-step walkthrough, link to specific line ranges in `gptlab/train.py` (`forward_backward`, the clip/step block, `save_checkpoint`) and in `gptlab/reference.py`. Build "What didn't work" from the `NOTES.md` entries.

- [ ] **Step 5: Make the repos public (after the user approves)**

Once the user approves: for each repo, run `uv run python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('<repo>', private=False)"`

- [ ] **Step 6: Commit**

```bash
git add README.md NOTES.md
git commit -m "docs: README with scaling results, dynamics findings, costs and what didn't work"
```

---

## Self-Review Notes

- **Spec coverage:**
  - §2 decisions → Global Constraints and Tasks 2–9.
  - §4 data protocol → Tasks 5, 6, 7, 13.
  - §5 model and loop → Tasks 3, 4, 8.
  - §6 phases → Tasks 13–17.
  - §7 analysis → Tasks 11, 12, 16.
  - §8 Modal → Task 10.
  - §9 publishing → Tasks 12, 18.
  - §10 risks → the gates in Tasks 14–16.
- **Deviations from the spec** are listed under "Corrections to the spec made while planning" at the top.
- **Name consistency:** `run_id`, `TrainConfig`, `size_for_budget`, `lr_for`, `declared_test_config`, `grid_points`/`grid_runs`, `load_curve` and `cost_line` are defined once and used with the same signatures throughout.
