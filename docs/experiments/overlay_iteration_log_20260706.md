# Overlay Iteration Log — 2026-07-06

**Branch:** `sota-24h-campaign-20260706`  
**Serial queue:** `scripts/run_ours_overlay_iterate_queue.sh` + `scripts/lora_ours_gpu_queue_continue.sh`  
**Main table:** `results/tables/sota_main_table_20260706.md`

## Version matrix (campaign queue)

| # | Job ID | Version | Suite | State | Key metric | Gate |
|---|--------|---------|-------|-------|------------|------|
| 1 | citb-replay50-smoke | base v55 | CITB | **PASS** | EM/ROUGE-L **50.0** | smoke PASS |
| 2 | standard-v85-smoke | overlay v85 | Standard | **RUNNING** | pending | dbpedia EM≥90 |
| 3 | arper-v86-formal | anchor v86 | Dialogue ARPER | **DONE** | BLEU **0.632** / SER **4.817** | = v66; below paper |
| 4 | todcl-adapter-anchor | anchor | Dialogue ToDCL | **queued** | — | after v85 |
| 5 | citb-ours-v85-smoke | ours v85 | CITB | **DONE** | ROUGE-L AR **24.6** | early gate PASS |

## Blocker fix (2026-07-06 13:34)

ARPER v86 training completed but monitor `while true` loop kept tmux alive → GPU queues blocked 13h. Fixed:
- `monitor_arper` exits on `completed_or_stopped`
- `lora_ours_gpu_queue.sh` releases monitor-only sessions via status JSON
- `run_ours_overlay_iterate_queue.sh` same for ARPER job

## ARPER v86 formal — DONE

- Final: BLEU **0.63231** / SER **4.817** (identical to v66)
- Paper: BLEU **0.701** / SER **3.63** — **not reproduced**
- SOTA target BLEU **0.935** / SER **2.72** — **not reached**
- Next: overlay v87+ (SSRG exemplar on SCLSTM)

## CITB ours v85 smoke — DONE

- 5 segments; final ROUGE-L AR **24.6** (target **53.87**)
- Early gate PASS (seg2 TA AR 31.25% > 15)
- Formal 19-task + Replay50 formal still deferred

## Standard v85 — RUNNING (relaunch)

- SSRG `import re` fix: `11c8ecc`
- tmux: `lora-ours-standard-v85-smoke`
- Early gate: dbpedia EM≥90, amazon EM≥45
- **13:42:** round1 dbpedia EM **98.5 PASS**; round2 amazon in progress
- GPU queue: `lora-ours-sota-gpu-queue` waiting for v85 + idle GPU → ToDCL anchor

## ToDCL — queued

- Preflight PASS; first launch blocked `queued_gpu_busy` at 03:23
- Continue queue: `lora_ours_gpu_queue_continue.sh` → after v85

## SOTA targets (unchanged)

| Suite | Target | Best ours so far |
|-------|--------|------------------|
| CITB ROUGE-L AR | **53.87** | v54 formal **33.1**; v85 smoke **24.6** |
| Standard avg EM | **84.5** | v69 **77.26** |
| Dialogue BLEU | **0.935** | ARPER v86 **0.632** |

**Honest gap:** no numerical SOTA margin reached.
