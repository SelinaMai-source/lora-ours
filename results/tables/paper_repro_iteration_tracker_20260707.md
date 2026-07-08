# Paper Reproduction Iteration Tracker — 2026-07-07

**Branch:** `sota-24h-campaign-20260706`  
**Gate:** `results/logs/strict_paper_repro_gate_1_20260707.json`  
**Orchestrator:** `scripts/run_strict_paper_repro_iteration.sh`

| Iter | Version | Suite | Local | Paper | Gap | ±1? | Next |
|------|---------|-------|-------|-------|-----|-----|------|
| 1 | v56 | CITB AR | 39.98 | 40.4 | −0.42 | **PASS** | — |
| 1 | v57 | Standard EM | 76.81 | 75.8 | +1.01 | **PASS*** | — |
| 1 | v87 | ARPER BLEU | 0.601 | 0.701 | −0.10 | FAIL | v88 |
| 2 | v88 | ARPER | BLEU 0.58756 SER 7.770 | 0.701/3.63 | — | FAIL | v89 |
| 2 | v89 | ARPER | **RUNNING** Hotel Ep7 (task 3/6) | 0.701/3.63 | interim BLEU 0.622 SER 4.06 @Train | RUNNING | gate→ToDCL |
| 2 | ToDCL | ADAPTER | queued (GPT2 fixed, disk unblocked) | 21.77/0.164 | — | pending | auto after v89 |
| 3 | CITB v2 | Replay50 | blocked | 40.4 | — | optional | Stage-1 seed50 |

**GPU:** ARPER v89 (`lora-ours-arper-v89-formal`, PID 257186) → post-v88 gate waits v89 → ToDCL auto-launch. ETA ~6–10h remaining.

**Disk:** cleanup 2026-07-08 freed **~101G** autodl-tmp + **~3G** / — see `results/logs/disk_cleanup_20260708.md`.

**Queue:** `lora-ours-paper-alignment-queue` (idle) | **Gate watcher:** `lora-ours-post-arper-v88-gate` (active, waiting v89) | **Watch:** `lora-ours-paper-alignment-watch` (healthy, 5m poll)

*Updated: 2026-07-08T08:55+08:00*
