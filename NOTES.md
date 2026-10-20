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
