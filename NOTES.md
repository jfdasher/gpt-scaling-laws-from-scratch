# Lab notes

Dated log of failures, surprises and fixes. This is the raw material for the README's
"What didn't work" section — write entries as they happen, not afterwards.

Format: `## YYYY-MM-DD — short title`, then what happened, how it was diagnosed, what fixed it.

## 2026-10-20 — Data prep and baselines
- First `prep_data` attempt failed before running anything: Modal validates every secret an app references at startup, even for functions that don't use it (`wandb` didn't exist yet). Created secrets, reran.
- Data: 4 parquet files encoded, 3 concatenated → 2.66B train tokens; val 10.0M tokens / 8,116 docs (0.5% doc-hash split); 3.87 bytes/token with the 8k BPE.
- Baselines on frozen val (nats/token, bpb): uniform 9.011/3.358, unigram 7.297/2.719, bigram 5.190/1.934, GPT-2 small reference bpb 1.036.
- Bigram λ landed on the grid edge (0.95): the bigram baseline is slightly pessimistic. Not worth re-running — it is a floor every model must clear, not a result.
