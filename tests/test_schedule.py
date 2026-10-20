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
