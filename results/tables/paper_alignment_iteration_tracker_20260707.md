# Paper Alignment Iteration Tracker — 2026-07-07

**Tolerance:** primary metrics `|local − paper| ≤ 1.0`  
**Branch:** `sota-24h-campaign-20260706`  
**Audit:** `docs/experiments/paper_alignment_root_cause_20260707.md`

| Suite | Attempt | Paper | Local | Gap | Fix applied | Status |
|-------|---------|-------|-------|-----|-------------|--------|
| **Standard O-LoRA** | v57 formal | EM 75.8 | **76.81** | +1.01 | Official-equivalent 1×GPU + grad_accum=8 | **PASS ±1** |
| **CITB Replay(50)** | v56 formal | ROUGE-L AR 40.4 | **32.44** | −7.96 | Official script 500/50/50; seed469 + tokenizer shim | **FAIL** |
| **CITB Replay(50)** | v2 (queued) | ROUGE-L AR 40.4 | — | — | Official checkpoint-14000 path; native tokenizer; seed50; minimal shims | **queued** (needs Stage-1 or ALLOW_FALLBACK) |
| **ARPER SCLSTM** | v66 formal | BLEU 0.701 | **0.632** | −0.069 | DA-wise granularity=1 (wrong row) | **FAIL** |
| **ARPER SCLSTM** | v87 formal | BLEU 0.701 | **~0.601** (4/6 dom) | −0.10 | Domain-wise but exemplar 500 + batch 64 | **FAIL / running** |
| **ARPER SCLSTM** | v88 (queued) | BLEU 0.701 | — | — | Exact config.cfg: exemplar 250, batch 128, granularity=0 | **queued** (after v87) |
| **ARPER SCLSTM** | v87 formal | SER 3.63 | TBD | — | Same as v87 BLEU row | **running** |
| **ToDCL ADAPTER** | anchor 20260706 | BLEU 21.77 | — | — | GPT-2 corrupt / HF offline | **blocked → fixed** |
| **ToDCL ADAPTER** | anchor retry (queued) | BLEU 21.77 | — | — | Local GPT-2 path + offline transformers | **queued** (after ARPER v87) |
| **ToDCL ADAPTER** | anchor retry | EER 0.164 | — | — | Same run; use ±10% relative for small metric | **queued** |

## Next actions (GPU serial)

1. **Do not interrupt** ARPER v87 (`lora-ours-arper-v87-formal`) — healthy, ~Restaurant epoch 11/100.
2. After v87: **ToDCL ADAPTER** anchor retry → **CITB v2** formal (download Stage-1 checkpoint-14000 first, or diagnostic with `ALLOW_FALLBACK_STAGE1=1`) → **ARPER v88**.
3. Monitor: `scripts/monitor_paper_alignment.sh` (tmux `lora-ours-paper-alignment-watch`).

## ETA (rough)

| Job | Estimate |
|-----|----------|
| ARPER v87 remainder | ~8–12 h (5 domains × ~100 epochs) |
| ToDCL ADAPTER 37-domain | ~6–10 h |
| CITB v2 formal 19-task | ~8–12 h |
| ARPER v88 | ~8–12 h |

*Updated: 2026-07-07T07:15+08:00*
