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
