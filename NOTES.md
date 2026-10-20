# Lab notes

Dated log of failures, surprises and fixes. This is the raw material for the README's
"What didn't work" section — write entries as they happen, not afterwards.

Format: `## YYYY-MM-DD — short title`, then what happened, how it was diagnosed, what fixed it.

## 2026-10-20 — Data prep and baselines
- First `prep_data` attempt failed before running anything: Modal validates every secret an app references at startup, even for functions that don't use it (`wandb` didn't exist yet). Created secrets, reran.
- Data: 4 parquet files encoded, 3 concatenated → 2.66B train tokens; val 10.0M tokens / 8,116 docs (0.5% doc-hash split); 3.87 bytes/token with the 8k BPE.
- Baselines on frozen val (nats/token, bpb): uniform 9.011/3.358, unigram 7.297/2.719, bigram 5.190/1.934, GPT-2 small reference bpb 1.036.
- Bigram λ landed on the grid edge (0.95): the bigram baseline is slightly pessimistic. Not worth re-running — it is a floor every model must clear, not a result.

## 2026-10-20 — Smoke test and throughput bench
- Smoke: simulated preemption at step 100 of 200 resumed via Modal retry; every step logged exactly once. Max |loss_preempt − loss_a| = 0.029 after resume vs 0.046 *before* it — GPU nondeterminism (bf16, SDPA backward), not resume error. Exactness itself is proven by the CPU fp32 test.
- I re-ran `baselines` once by mistake (the first run had already written baselines.json): ~$0.30 wasted.
- Bench (H100, torch.compile, bf16, 64k-token batches): 2.6M params 3.0M tok/s (MFU 4.7%), 7.7M 2.4M tok/s (11%), 18M 1.3M tok/s (15%), 57M 0.69M tok/s (24%). Tiny models are launch/overhead-bound, as expected; the eager fp32 loss over 8192 logits is a likely culprit at the small end.
- Re-estimated: LR sweep $2.82, grid $7.78 (was $14 at the 7e13 FLOP/s guess).

## 2026-10-20 — LR sweep
- First pass: the 2.6M (3e15, r80) and 9.3M (1e16, r20) optima sat on the *upper* edge of their grids (best 1.6e-2 and 6e-3). `fit-lr-rule` refused to write the rule, which is what the gate is for.
- Moved the small-end centre ×4 (4e-3 → 1.6e-2): 3.2e-2 and 6.4e-2 are clearly worse (4.57, 5.22 vs 3.95), so 1.6e-2 is interior.
- Moved the check centre ×4 to 1.2e-2, which put 6e-3 at the *lower* edge of the new 3-point grid; the earlier 3e-3 run (3.702) already brackets it, so I recentred on 6e-3 (3e-3/6e-3/1.2e-2, all already run) — no extra spend.
- Rule: ln lr = 8.090 − 0.8285 ln N (lr ∝ N^−0.83). Check point ratio 0.91. Optima are on a ×2 grid, so each is only known to within ~√2.
- Instability shows up close to the optimum: 9.3M jumps from 3.656 at 6e-3 to 4.391 at 1.2e-2, and 31M from 3.534 at 2e-3 to 4.474 at 8e-3. Useful context for the dynamics track: the "3×" and "10×" LR multiples will be well past this cliff.
- LR sweep total: 17 runs, ~$2.5.

## 2026-10-20 — The 3e15 budget is broken (diagnosis in progress)
- Grid: 25/25 runs finished (~$7.30). 1e16/3e16/1e17 look healthy (optima at 32, 30, 19 tokens/param).
- 3e15 is not: seed-pair noise σ = 0.106 nats (pairs differ by up to 0.26), losses don't form a parabola, and r5 (10M params, 753 steps) ends at 5.08, barely under the bigram baseline (5.19).
- Evidence: no spikes or late divergence (smoothed loss never rises > 0.08 above its running minimum). Seed curves separate in the first ~10% of training and then run parallel. The final-30% slope dL/d ln(step) is −0.66 to −0.77 for 3e15 r5/r20, against −0.27 to −0.42 elsewhere: these runs end while still falling steeply.
- Hypothesis: at 3e15 the bigger models get only 750–1,500 optimizer steps at the 64k batch, so final loss measures optimization progress plus a seed-dependent head start, not capacity. That breaks the isoflop assumption.
- Test: rerun 3e15/r20, both seeds, at a 16k batch (4× steps, same tokens, LR and FLOPs). Prediction: seeds agree to within a few hundredths and loss drops well below 4.24–4.53.
- **Result (confirmed):** at a 16k batch (5,966 steps), 3e15/r20 reaches 3.8970 / 3.8983 for the two seeds, a gap of 0.001. At the 64k batch (1,492 steps) the same pair reached 4.531 / 4.275. Root cause: step starvation. 64k tokens is far above the critical batch size for ~5M-param models over ~100M tokens. Final loss measured optimization progress, not capacity.
- Implication: the bias is systematic. Within each budget the large-N (low tokens/param) runs get the fewest steps and are penalized most, which pushes N_opt down. 1e16 r5 (1,387 steps, end slope −0.42) is probably affected too. The 3e16 batch sweep in the dynamics track will show where 64k becomes safe.

## 2026-10-20 — Training-dynamics track (16M params at 3e16, rule LR 3.52e-3)
- Warmup: 5% gives 3.445, 1% gives 3.520, 0% gives 3.680. Warmup matters a lot at this LR; even 1% costs 0.075 nats.
- β₂: 0.999 gives 3.477, against 3.445 at 0.95. The slower second-moment estimate lags the fast-moving early gradients.
- Batch size at equal tokens and LR: 32k 3.433, 64k 3.445, 128k 3.472, 256k 3.687, 512k 4.591. Here 64k sits at the knee of the critical batch size: halving it gains ~0.01, doubling it costs 0.03 then 0.24 then 1.15. Caveat: same LR at every batch size.
- Instability: 3× LR ends at 4.24 (clip) / 4.46 (no clip); 10× LR ends at 5.38 / 6.10. **None went NaN.** Per-layer diagnostics, every 20 steps, show the mechanism: attention max-logit grows from ~0.7 at init to 1e3–1.4e4 at 3× and 2e4–3e5 at 10×, worst in the last layers. Softmax saturates and attention becomes one-hot. Loss stalls around 6.6 at 10× until the LR decays. Gradient norms in the middle layers collapse to ~0.01 while layer 0 spikes (15 at step 2818 in lr10x; total 2,931 in lr10x-noclip). It's attention-logit growth (Dehghani et al. 2023; Wortsman et al. 2023), not a loss blow-up, which is why a NaN guard alone would never have caught it.
- Clipping helps but does not fix it: at 3× it narrows the gap (4.24 vs 4.46) without stopping logit growth.
- **Fixes at 10× LR, no clip:** QK-norm gives 3.574, against 6.105 unmodified and 3.445 for the tuned baseline; max attention logits stay bounded at ~14–30. z-loss gives 5.974, essentially no help. z-loss constrains the output softmax normaliser (log Z), not attention logits, so it can't touch this failure mode. That's a useful negative result.

## 2026-10-20 — Grid v2: batch per budget
- Reran 3e15/1e16/3e16 at batch 16k/32k/32k (1e17 stays 64k, results reused; grid/3e16/r20 at 32k is the identical run to dyn/batch32k and is reused). 19 new runs, ~$3.40. Minimum steps per run is now ~2,800 (was 753).
- Isoflop optima, v1 (64k everywhere) → v2:
  - 3e15: 2.0M (EDGE) → 4.5M (25 tok/param).
  - 1e16: 7.2M → 9.1M (20).
  - 3e16: 12.9M → 17.5M (16).
  - 1e17: 29.7M (19), unchanged.
- Exponent a: v1 0.747 → v2 **0.545**, against Chinchilla's ~0.5. The v1 bias went the predicted direction: step starvation penalised the large-N points most, which pushed N_opt down at small C and steepened a.
- Seed noise σ = 0.085 nats, but it is dominated by one pair: 3e15/r5 (10M params, 5 tok/param) at 4.270 vs 4.019. r10 differs by 0.098; r20, r40 and r80 by 0.0005, 0.016 and 0.004. The r5 gap isn't instability (max grad norm 5.8, no attention blow-up). Train-loss bumps of ~0.2 appear at identical steps in runs reading the same data region (2335 for seed-0 runs, 4259 for seed-1), so they are data-driven. I'm keeping the pre-registered estimator (RMS over all pairs) rather than switching to a robust one after seeing the data. The isoflop intervals on a (0.12–1.67) are therefore wide and probably pessimistic for the larger budgets.
