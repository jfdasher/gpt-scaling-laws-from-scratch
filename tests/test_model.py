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
