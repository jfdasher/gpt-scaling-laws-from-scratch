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
