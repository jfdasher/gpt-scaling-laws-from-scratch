[Published W&B report](https://forge.coreweave.com/wandb/jfdasher-thermopylae-systems/gptlab-scaling/reports/Small-GPT-scaling-laws,-from-scratch--VmlldzoxODE1NDAzMg==)

# W&B report draft: "Small GPT scaling laws, from scratch"

Project: https://forge.coreweave.com/wandb/jfdasher-thermopylae-systems/gptlab-scaling

How the runs are organised:
- **Group** = phase: `smoke`, `lr`, `grid`, `dynamics`, `probe`, `test`.
- **Run name** = the config name, e.g. `grid/1e+17/r20`, `dyn/lr10x-noclip`.
- Grid v1 and grid v2 share the `grid` group. To tell them apart, filter on `config.batch_tokens`:
  - **v2:** 3e15 = 16384, 1e16 / 3e16 = 32768, 1e17 = 65536
  - **v1:** 65536 at every budget

---

## 1. TL;DR

> I trained 79 Llama-style GPTs (2.6M–57M parameters) from scratch on FineWeb-Edu using a hand-written PyTorch training loop on Modal H100s, for $18 total. They were used to fit a compute-optimal scaling law two ways.
> - **Isoflop fit:** N_opt ∝ C^0.545. Tokens per parameter at the optimum: 16–25.
> - **Parametric fit:** a = 0.565. It agrees with the isoflop fit, and both are close to Chinchilla's ~0.5.
> - **Prediction test:** before training, I committed a prediction for a 3e17-FLOP run, git-tagged `prediction-3e17`. The prediction came in within 2.1% (parametric) and 1.1% (frontier extrapolation) of the measured loss of 3.041.

## 2. Setup

> Llama-style architecture: RMSNorm, RoPE, SwiGLU, tied embeddings. Own 8k byte-level BPE tokenizer.
>
> The val split is frozen by document hash (0.5% of docs, 10M tokens), with its hash committed before training.
>
> Baselines on that split, in val loss (nats/token):
>
> | Baseline | Val loss |
> |---|---|
> | uniform | 9.01 |
> | unigram | 7.30 |
> | bigram | 5.19 |
>
> GPT-2 small scores 1.036 bits per byte on the same text, for reference.
>
> Compute is counted as C = 6·N·D. Every run uses a cosine schedule sized to its own length.

**Panel:** a run table for the `grid` group (v2 filter). Columns: name, `n_params`, `tokens`, `final/val_loss`, `final/val_bpb`, `gpu_seconds`.

## 3. LR sweep

> The optimal LR falls steeply with model size: lr ∝ N^−0.83. Loss rises sharply just above each optimum.
>
> The first pass hit the upper edge of the sweep range for the small models. The fit refuses to write a rule from an edge optimum, so I widened the range and reran.

**Panels:**
- Line plot of `val/loss` against step, group `lr`, grouped by run name.
- Image: `figures/lr_sweep.png`.

## 4. Isoflop grid

> Four budgets (3e15–1e17), five sizes each, at 5–80 tokens per parameter.

**Panels:**
- `val/loss` against step for group `grid` (v2 filter), coloured by `config.budget_flops`.
- Images: `figures/isoflop.png`, `figures/optima.png`, `figures/frontier.png`, `figures/tokens_per_param.png`.

## 5. What didn't work: step starvation (grid v1)

> **Symptom.** The first grid used a 64k-token batch at every budget. The 3e15 budget was noisy: paired seeds differed by up to 0.26 nats, the curve had no parabola, and one run barely beat the bigram baseline.
>
> **Diagnosis.** No spikes or divergence. The seed curves split early and then ran in parallel. Late in training, loss was still falling 2–3× faster than at the larger budgets.
>
> **Probe ($0.24).** Same model, same tokens, but a 16k batch. The two seeds landed at 3.897 and 3.898, against 4.53 and 4.28 at 64k.
>
> **Fix.** A batch size per budget. The exponent moved from 0.747 to 0.545, because step starvation had penalised the large-N points most.

**Panels:**
- `train/loss` for `grid/3e+15/r20` and `/seed1` at 64k, overlaid with `probe/3e+15/r20/b16k` and `/seed1`.

## 6. Training dynamics (16M parameters at 3e16)

**Instability.** At 3× and 10× the tuned LR, attention max-logit grows from about 0.7 to 1e3–4e5, worst in the last layer. Softmax saturates and loss stalls near 6.6. Nothing ever NaNs.

| Run (10× LR, no clipping) | Final val loss |
|---|---|
| Unmodified | 6.105 |
| + QK-norm | **3.574** |
| + z-loss | 5.974 |
| Tuned baseline (1× LR, for reference) | 3.445 |

QK-norm keeps attention logits at 14–30. z-loss acts on the output softmax normaliser, not on attention logits, so it can't fix this failure.

**Panels:**
- `train/loss` and `train/grad_norm` (log y) for `dyn/baseline`, `dyn/lr3x*`, `dyn/lr10x*`, `dyn/lr10x-noclip+qknorm` and `dyn/lr10x-noclip+zloss`.
- `attn_max_logit/layer_0..10` for `dyn/lr10x-noclip` and `dyn/lr10x-noclip+qknorm`.
- Images: `figures/dyn_instability.png`, `figures/dyn_instability_layers_*.png`.

**Warmup.** 5% → 3.445, 1% → 3.520, 0% → 3.680.

**β₂.** 0.95 → 3.445, 0.999 → 3.477.

**Batch size.** At equal tokens and equal LR:

| Batch | Final val loss |
|---|---|
| 32k | 3.433 |
| 64k | 3.445 |
| 128k | 3.472 |
| 256k | 3.687 |
| 512k | 4.591 |

64k sits at the knee of the curve.

**Panels:** `val/loss` against step for the `dyn/warmup*`, `dyn/beta2*` and `dyn/batch*` runs.

## 7. Prediction test (3e17)

> The prediction was committed and tagged before the run was launched:
> - Model: 57.1M parameters, 875M tokens.
> - Parametric fit: predicted loss 3.106, with a prediction interval of 2.92–3.27.
> - Frontier extrapolation: predicted loss 3.008.
>
> **Actual: 3.041.** Both predictions are within about 1σ of seed noise, at 3× extrapolation in compute.

**Panels:**
- `val/loss` for `test/3e+17` overlaid with `grid/1e+17/r20`.
- Image: `figures/optima.png`, which shows the predicted and the trained point.

## 8. Cost

| Phase | Runs | Cost |
|---|---|---|
| LR sweep | 17 | $2.84 |
| Grid (v1 + v2) | 44 | $9.68 |
| Dynamics | 13 | $3.67 |
| Probe | 2 | $0.24 |
| Smoke | 2 | $0.11 |
| Test | 1 | $1.47 |
| **Total** | **79** | **≈ $18** |

Spend: 4.6 H100-hours, ≈ $18 in total.

**Panel:** scalar chart of `gpu_seconds` per run.

Links:
- Model: https://huggingface.co/jfdasher/gptlab-57m-3e17
- Code: GitHub link to be added.
