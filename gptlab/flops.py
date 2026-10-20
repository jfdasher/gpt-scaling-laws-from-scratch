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
