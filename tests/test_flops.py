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
