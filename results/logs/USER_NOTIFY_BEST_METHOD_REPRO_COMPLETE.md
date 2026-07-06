# Best-Method Reproduction Complete — User Notification

**Time:** 2026-07-07 04:02 UTC+8

**Branch URL:** https://github.com/SelinaMai-source/lora-ours/tree/best-method-repro-results-20260706

## Summary (paper vs local)

| Suite | Method | Paper | Local | Verdict |
|-------|--------|-------|-------|---------|
| CITB Replay(50) | Replay(50) formal 19-task | ROUGE-L AR 40.4 | ROUGE-L AR 32.4417 | No |
| Standard O-LoRA | O-LoRA v57 formal | EM avg 75.8 | EM 76.8059 | Yes / Close |
| ToDCL ADAPTER | ADAPTER 37-domain NLG anchor | BLEU 21.77 / EER 0.164 | — | — |
| ARPER v87 | v87 paper-aligned domain/exemplar500 | BLEU 0.701 / SER 3.63 | BLEU 0.60059 SER error: | TBD |

## What matched

- **standard_olora_v57**: EM 76.8059 (single-GPU grad_accum=8 official-equivalent)

## What did not match

- **citb_replay50**: official_script_500_50_50; gap −7.96 vs paper
- **todcl_adapter**: gpt2 load failure EXIT=1
- **arper_v87**: granularity=0 exemplar 500

## Blockers

- ToDCL ADAPTER: corrupt/missing GPT-2 weights; network unreachable for re-download
