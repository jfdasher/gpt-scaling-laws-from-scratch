# Small GPT Scaling Laws on Modal — Design

Date: 2026-10-19
Status: Draft for review

## 1. Purpose

A portfolio project that demonstrates hands-on neural-network training, not orchestration of someone else's stack. It has three outcomes:

1. **The training step, written by hand.** Plain PyTorch with no HF Trainer: tokenizer, data loader, AdamW, warmup + cosine schedule, bf16 autocast, gradient clipping, and checkpoint/resume. This is the answer to "walk me through what happens in a training step."
2. **A compute-optimal scaling law.** An IsoFLOP sweep across four compute budgets, cross-checked with a parametric fit, then tested by predicting a larger run *before* training it.
3. **A training-dynamics track.** Controlled single-setting experiments, including a deliberately provoked loss instability that is then diagnosed and fixed. This supplies the debugging story.

Non-goals: reproducing nanochat or modded-nanogpt speedruns, multi-GPU training, μP, WSD schedules, downstream benchmarks, chat/instruction tuning.

## 2. Fixed decisions

| Topic | Decision |
|---|---|
| Compute platform | Modal, H100 for all GPU runs. Data prep on Modal CPU. |
| Budget currency | FLOPs, C ≈ 6·N·D. N = **total** parameters (tied embeddings); non-embedding N reported alongside. |
| Grid budgets | 3e15, 1e16, 3e16, 1e17 FLOPs |
| Test (extrapolation) budget | 3e17 FLOPs |
| Sizes per budget | 5, at tokens/param r ∈ {5, 10, 20, 40, 80}: N_target = √(C / (6r)) (log-even, ratio √2) |
| Seeds | 1, plus a second seed for all 5 runs at 3e15 |
| Schedule | Linear warmup 5% of steps, cosine to 10% of peak, sized to **each run's own length** |
| Dataset | `HuggingFaceFW/fineweb-edu`, config `sample-10BT`. No run repeats data. |
| Tokenizer | Byte-level BPE, vocab 8192, HF `tokenizers`, trained on train documents only |
| Architecture | Llama-style: pre-norm RMSNorm, RoPE (HF `rotate_half` convention), SwiGLU with hidden ≈ 8/3·d, no biases, tied embeddings, `F.scaled_dot_product_attention` |
| Context | 512 tokens; head dim 64 |
| Optimizer | Fused `torch.optim.AdamW`, β = (0.9, 0.95), weight decay 0.1 on tensors with ≥2 dims only, grad-clip 1.0 |
| Batch | 64k tokens (128 × 512) for grid and test runs (see §6.3) |
| LR | Standard parameterization; LR swept at the small and large ends, log-linear rule in N |
| Precision | bf16 autocast, fp32 master weights, no GradScaler; `torch.compile` |
| Logging | W&B (public project and report) |
| Publishing | HF Hub in `LlamaForCausalLM` format, Apache-2.0, FineWeb-Edu (ODC-By) credited |
| Spend | Modal workspace limit $60; expected total $22–41 |

## 3. Repository layout

```
gpt_training/                (git, uv, Python 3.12)
├── gptlab/                  # never imports modal; runs on CPU
│   ├── model.py             GPT; param counts (total, non-embedding); qk_norm, z_loss switches
│   ├── reference.py         hand-written AdamW and causal attention (tested for parity)
│   ├── data.py              memmapped uint16 stream reader; val manifest check
│   ├── tokenizer.py         train 8k BPE; encode documents to shards
│   ├── schedule.py          warmup + cosine to 10% of peak
│   ├── flops.py             C = 6ND; (budget, r) → (n_layer, d_model, steps, tokens)
│   ├── train.py             train(config, run_dir) → result.json
│   └── baselines.py         uniform / unigram / bigram / GPT-2 reference
├── modal_app.py             image, Volumes, Secrets, prep_data, train_one, phase entrypoints
├── sweep.py                 phase grids: smoke, lr, grid, dynamics, test; cost dry-run; --yes
├── analysis/
│   ├── isoflop.py  parametric.py  predict.py  plots.py  costs.py  export_hf.py
├── data/val_manifest.json   committed
├── baselines.json           committed before any training
├── lr_rule.json             committed after the LR sweep
├── predictions/3e17.json    committed + tagged before the test run
├── figures/
├── tests/                   CPU only
├── NOTES.md                 dated log of failures and surprises
└── README.md
```

Only `modal_app.py` imports `modal`. Each run writes `result.json` containing: config, run_id, n_layer, d_model, N_total, N_nonembed, D, C, final val loss (nats/token), bits per byte, GPU-seconds (cumulative across restarts), cost in dollars, W&B URL, and git commit.

## 4. Data and evaluation protocol

### 4.1 Frozen split (done first)

- A document goes to val if `int(sha256(doc.id), 16) % 1000 < 2` (~0.2% of documents). The split is deterministic and independent of file order.
- `val.bin` holds the first ~10M val tokens, in fixed order. A 1M-token prefix is used for evals during training.
- `data/val_manifest.json` records document count, token count, byte count (UTF-8), sha256 of `val.bin`, and sha256 of `tokenizer.json`. It is committed to git.
- `train()` checks the manifest hash at startup and refuses to start on a mismatch.
- Val text is never inspected by hand. Only the encoder and the eval function read it.
- Test: the val and train document-ID sets don't intersect.

### 4.2 Tokenizer

- HF `tokenizers` ByteLevel BPE with vocab 8192, including the special token `<|endoftext|>`, which separates documents. GPT-2-style regex pre-split.
- Trained on ~1GB of train documents.
- Tests: roundtrip on arbitrary text (Unicode, code, whitespace runs), and the special-token ID is stable.

### 4.3 Encoding

- `.map()` over parquet files on Modal CPU containers. Each container shuffles its documents with a seed derived from the file name, joins them with EOT, and writes a uint16 shard.
- Shards are concatenated in seeded order into `train.bin`, stopping once there are at least 2.2B tokens.
- Documents are packed across sequence boundaries with no cross-document attention masking. The README states this.
- The data stream is read sequentially. For seed 0, step k reads tokens `[k·B, (k+1)·B)`. Seed 1 (only for the noise estimate at 3e15) starts at a token offset of 1.0B, so it gets different data and a different init without overlapping seed 0's tokens.

### 4.4 Metrics

- Loss in nats per token on val.
- Bits per byte = (loss / ln 2) × (val tokens / val bytes), using the exact counts from the manifest.

### 4.5 Baselines (`baselines.json`, before any training)

| Baseline | Method |
|---|---|
| Uniform | ln 8192 ≈ 9.01 nats |
| Unigram | Counts from train, add-one smoothing, cross-entropy on val |
| Bigram | Interpolated with unigram; λ tuned on a train slice; cross-entropy on val |
| GPT-2 small (124M) | bpb on the val text with its own tokenizer (~1 GPU-minute); an external reference only |

Any trained model that doesn't beat the bigram is treated as a bug.

## 5. Model and training loop

### 5.1 Model

- Llama-style as in §2. Optional switches: `qk_norm` (RMSNorm on Q and K) and `z_loss` (coefficient on log²Z). Both are off in the grid and test runs.
- Parameter count: N_total = 12·L·d² + V·d (+ norm weights, counted exactly). A test asserts the formula.
- Shapes: `flops.py` picks (L, d) with d a multiple of 64 and d/L between 16 and 64, minimizing |log N − log N_target|. D is then recomputed as C / (6·N_actual) so that C is exact.

### 5.2 One optimizer step

1. Read micro-batches of shape `(B_micro, 512)` from the memmap, copy to the GPU with pinned memory and `non_blocking`.
2. Under bf16 autocast: forward, cross-entropy (+ z-loss if enabled), divide by the number of accumulation steps, backward. Repeat until `batch_tokens` have accumulated.
3. `clip_grad_norm_(…, 1.0)` (can be disabled); log the pre-clip norm.
4. Set the LR from `schedule.lr(step)`.
5. AdamW step; `zero_grad(set_to_none=True)`.
6. Every k steps, log: loss, LR, grad norm, tokens/s, MFU.

### 5.3 Evaluation and logging

- About 10 evals per run on the 1M-token val prefix: loss and bpb, grad norm per layer, max attention logit per layer, and generated samples from fixed prompts with a fixed seed (as a W&B table).
- Final eval on the full ~10M-token val set. This value feeds `result.json` and the fits.

### 5.4 Checkpoint and resume

- `run_id` = hash of the canonical config. `run_dir` lives on the checkpoint Volume.
- A checkpoint is saved every ~10 minutes of wall-clock time and at the end. It contains: model, optimizer, step, config, RNG states, W&B run ID, and cumulative GPU-seconds.
- Writes are atomic: `ckpt.tmp` → `os.replace` → `volume.commit()`. Only the latest checkpoint is kept, plus the final weights.
- On start: if `ckpt.pt` exists, resume from it. W&B resumes with `id=…, resume="must"`.
- If `result.json` already exists, the run is skipped. That gives sweep/grid reuse and idempotent retries.

### 5.5 Failure handling

- On a NaN or Inf loss: save a forensic checkpoint plus the batch token offsets, log loudly, and raise. There is no automatic batch skipping or rollback.
- Spikes that recover are logged and left alone.

### 5.6 Reference implementations (`reference.py`)

- A hand-written AdamW (bias correction, decoupled weight decay). It matches `torch.optim.AdamW` over 50 steps on a small model within tolerance.
- A hand-written causal attention. It matches `F.scaled_dot_product_attention(is_causal=True)` within tolerance.
- These are for understanding and testing only. Real runs use the fused and SDPA versions.

### 5.7 CPU tests

- Parameter-count formula, schedule endpoints, overfitting a single batch, reference parity.
- **Resume test:** 20 straight steps give exactly the same per-step losses as 10 steps, then a checkpoint, a fresh process and 10 more steps (fp32, compile off).
- Manifest check, no train/val overlap, tokenizer roundtrip.
- Fits recover known exponents from synthetic data.
- HF export logits parity.

## 6. Experiments

### 6.1 Phases (each launched by hand with `modal run --detach`, each with a gate)

| # | Phase | Runs | Gate |
|---|---|---|---|
| 0 | Setup | Set the Modal spend limit to $60; W&B and HF Secrets; CPU tests green | — |
| 1 | Smoke test | Tiny H100 run with `--simulate-preempt-at-step`, then automatic retry | Per-step losses identical after resume; W&B run continuous; tokens/s and MFU recorded per grid shape |
| 2 | LR sweep | 5 LRs (×2 apart) at r=80/3e15 (~2.5M) and at r=5/3e16 (~32M); 3 LRs at r=20/1e16 (~9M) | Each optimum lies inside its LR grid, otherwise extend the grid. Fit log LR\* = c₀ + c₁ log N, write `lr_rule.json`, and check the 9M point against it |
| 3 | IsoFLOP grid | 4 budgets × 5 sizes, + 5 seed-1 runs at 3e15; `.map()` with at most 10 containers | Each budget's loss minimum lies inside its 5 sizes; if not, add one size at r=2.5 or r=160 for that budget |
| 4 | Dynamics track | §6.2 (can run alongside phase 3) | — |
| 5 | Fit and predict | §7; commit `predictions/3e17.json` and git tag `prediction-3e17` | Prediction committed and tagged |
| 6 | Test run | One run at 3e17 with N = IsoFLOP N_opt (snapped to a shape), D = C/(6N) | — |
| 7 | Publish | HF Hub, W&B report, README | — |

LR sweep runs whose config matches a grid point (same budget, size and LR) are reused automatically through `run_id`.

### 6.2 Training-dynamics track

The baseline is the 3e16 grid run nearest r=20 (~16M), using its rule-derived LR and the default settings. One setting changes per run (~12 runs):

| Experiment | Settings compared |
|---|---|
| Instability | LR ×3 and ×10 the optimum, with and without clipping. Then, at the setting that diverges, add QK-norm, and separately z-loss (1e-4) |
| Warmup | 0%, 1%, 5% (baseline) |
| β₂ | 0.95 (baseline), 0.999 |
| Batch size | 32k, 64k (baseline), 128k, 256k, 512k tokens at equal FLOPs |

For each experiment, the deliverable is a figure plus a one-paragraph diagnosis in the README. The instability figure shows loss, grad norm and max attention logit, layer by layer, from the spike through the fix.

### 6.3 Batch-size note

The 64k-token batch (rather than the 128k discussed during design) keeps the step count reasonable across the whole grid. Steps per run range from ~760 (3e15, r=5) to ~17.7k (1e17, r=80), and the 3e17 test run is ~15k. With 128k, the smallest-budget corner would have had only ~380 steps, which risks under-optimized points that bias the fit. The batch-size experiment in §6.2 checks whether 64k is below the critical batch size at 3e16.

## 7. Analysis

1. **IsoFLOP (`isoflop.py`)**
   - For each budget, fit a quadratic to final val loss against ln N; the vertex gives N_opt(C), and D_opt = C / (6·N_opt).
   - Uncertainty: a parametric bootstrap with residual noise σ taken from the seed-0 vs seed-1 differences at 3e15.
   - Fit ln N_opt = a·ln C + k and ln D_opt = b·ln C + k′, with intervals.
   - Report tokens/param at the optimum for each budget.
   - The README states that the exponent fits rest on 4 points.
2. **Parametric (`parametric.py`)**
   - Fit L(N, D) = E + A/N^α + B/D^β: Huber loss (δ = 1e-3) on ln L, L-BFGS from a grid of starting points, over all grid runs, using the mean of the two seeds at 3e15.
   - 1000 bootstrap resamples over runs give intervals on E, A, B, α, β and the implied a = β/(α+β).
   - Compare with the IsoFLOP value of a. Report any disagreement, citing Besiroglu et al. (2024).
3. **Prediction (`predict.py`)**
   - Before phase 6, write to `predictions/3e17.json`: from both methods, N_opt, D_opt and the predicted loss, with intervals.
   - It also records the declared training config (the IsoFLOP N_opt) and the predicted loss for that exact (N, D).
   - After phase 6, report the actual loss against each prediction, and whether it falls inside each interval.
4. **Costs (`costs.py`)**: aggregates `result.json` into `costs.csv`, with totals per run and per phase.

### 7.1 Figures

- IsoFLOP curves with the fitted parabolas and their vertices.
- N_opt and D_opt against C on log-log axes, with the fitted lines, a 0.5 reference line, and the 3e17 point (predicted and actual).
- The compute frontier: best loss against C, with the extrapolation and the actual 3e17 result.
- Tokens/param at the optimum against C, with a reference line at 20.
- LR sweep curves and the LR rule.
- One figure per dynamics experiment.

## 8. Modal mechanics

- **Volumes:** `hf-cache` (with `HF_HOME` pointed there), `data` (train.bin, val.bin, tokenizer), `ckpt` (run_dirs, result.json), and `inductor-cache` (`TORCHINDUCTOR_CACHE_DIR`, so runs of the same shape compile once).
- **Secrets:** `wandb`, `huggingface`.
- **GPU functions:**
  - `gpu="H100"`, `retries=modal.Retries(max_retries=3)`.
  - Explicit timeouts: 2h for grid, sweep and dynamics runs; 6h for the 3e17 run.
  - Concurrency is capped at 10.
- **Launch:** every phase is launched with `modal run --detach`.
- **`sweep.py <phase>`** prints the run list, the runs that will be skipped (because `result.json` exists), estimated FLOPs, GPU-hours and dollars, and launches only with `--yes`.
- **Cost:** GPU-seconds are tracked inside each run (summed across restarts), and dollars = GPU-seconds × the H100 rate recorded in config.

### 8.1 Cost estimate (H100 ≈ $3.95/hr; effective 5e13–1e14 FLOP/s)

| Phase | FLOPs | Cost |
|---|---|---|
| Smoke test | — | ~$1 |
| LR sweep | ~2e17 | $2–4.50 |
| Grid + seed 1 | ~7.3e17 | $8–16 |
| Dynamics | ~3.6e17 | $4–8 |
| 3e17 test run | 3e17 | $3.50–6.50 |
| Overhead (startup, compile, eval) | — | $3–5 |
| **Total** | | **$22–41** (limit $60) |

## 9. Publishing

### 9.1 HF export and model cards

- `export_hf.py` converts checkpoints to `LlamaForCausalLM` (`tie_word_embeddings=True`), writing safetensors and `tokenizer.json`. A logits parity test against `gptlab` is required.
- **Pushed models:** the 3e17 model, plus the grid model nearest N_opt at each budget.
- **Model card contents:**
  - YAML metadata: license `apache-2.0`, dataset `HuggingFaceFW/fineweb-edu`, tags.
  - Config, N_total and N_nonembed, tokens, FLOPs, final loss and bpb.
  - The cost line: "this run: X GPU-hours, $Y."
  - Fixed-prompt samples, the W&B link and the git commit.
  - Intended use (research and teaching) and limitations (small, English only, no safety tuning).

### 9.2 README sections

1. TL;DR: a, b, tokens/param at the optimum, prediction vs actual, total cost
2. What happens in a training step (walkthrough linking to lines in `train.py`)
3. Data and evaluation protocol, baselines table
4. Method: grid, LR rule, schedule, parameter-count convention
5. Scaling results: both methods, the disagreement, the prediction test
6. Training dynamics: findings
7. What didn't work (from `NOTES.md`, kept as you go)
8. Cost: table per phase, plus a cost line for every run
9. Reproduce: the commands for each phase
10. Limitations and next steps (WSD, μP, more budgets and seeds)

## 10. Risks

| Risk | Mitigation |
|---|---|
| A budget's minimum falls at the edge of its sizes | Phase 3 gate adds a size at r=2.5 or r=160 |
| Small models have poor MFU, so cost overruns | Phase 1 measures it; `sweep.py` dry-run re-estimates; $60 hard limit |
| The LR rule doesn't extrapolate to ~50M | The large end of the sweep is at ~32M; the r=5 point at 1e17 (~58M) is a further check; a non-smooth isoflop curve is a signal to re-sweep |
| Sub-10M models bend the fit | Bootstrap intervals, the second method and the 3e17 prediction expose it; report, don't hide |
| Preemption mid-run | Checkpoints every 10 min, retries=3, resume test |
| The val split leaks or drifts | Doc-ID hash split, committed manifest, startup check, overlap test |
| The parametric fit is unstable (Besiroglu et al.) | Multi-start L-BFGS, Huber loss, bootstrap intervals; compared against IsoFLOP |
