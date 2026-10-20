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
