# Paper Reproduction Iteration Tracker — 2026-07-07

**Branch:** `sota-24h-campaign-20260706`  
**Gate:** `results/logs/strict_paper_repro_gate_1_20260707.json`  
**Orchestrator:** `scripts/run_strict_paper_repro_iteration.sh`

| Iter | Version | Suite | Local | Paper | Gap | ±1? | Next |
|------|---------|-------|-------|-------|-----|-----|------|
| 1 | v56 | CITB AR | 39.98 | 40.4 | −0.42 | **PASS** | — |
| 1 | v57 | Standard EM | 76.81 | 75.8 | +1.01 | **PASS*** | — |
| 1 | v87 | ARPER BLEU | 0.601 | 0.701 | −0.10 | FAIL | v88 |
| 2 | v88 | ARPER | running | 0.701 | — | TBD | monitor |
| 2 | ToDCL | ADAPTER | queued | 21.77 | — | pending | after v88 |
| 3 | CITB v2 | Replay50 | blocked | 40.4 | — | optional | Stage-1 seed50 |

**GPU:** ARPER v88 (`lora-ours-arper-v88-formal`) → ToDCL anchor → optional CITB Stage-1.

*Updated: 2026-07-07T16:45+08:00*
