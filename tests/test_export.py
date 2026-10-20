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
