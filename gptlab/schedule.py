"""Linear warmup, then cosine decay to final_frac * peak, over a run's own length."""

import math


def lr_at(step: int, total_steps: int, peak_lr: float, warmup_frac: float = 0.05, final_frac: float = 0.1) -> float:
    warmup = int(round(warmup_frac * total_steps))
    if warmup > 0 and step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return peak_lr * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress)))
