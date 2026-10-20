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
