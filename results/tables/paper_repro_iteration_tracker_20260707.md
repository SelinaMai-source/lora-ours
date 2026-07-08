# Paper Reproduction Iteration Tracker — 2026-07-08

**Branch:** `ours-v1-20260708` (from `sota-24h-campaign-20260706`)  
**Gate:** `results/logs/strict_paper_repro_gate_1_20260707.json`  
**Orchestrator:** `scripts/run_strict_paper_repro_iteration.sh`

| Iter | Version | Suite | Local | Paper | Gap | ±1? | Next |
|------|---------|-------|-------|-------|-----|-----|------|
| 1 | v56 | CITB AR | 39.98 | 40.4 | −0.42 | **PASS** | locked |
| 1 | v57 | Standard EM | 76.81 | 75.8 | +1.01 | **PASS*** | locked |
| 2 | v89 | ARPER | BLEU **0.599** SER **5.94** | 0.701/3.63 | — | anchor locked | v1 SSRG overlay |
| 2 | ToDCL | ADAPTER | **RUNNING** ep8/10 | 21.77/0.164 | — | pending | v1 assess overlay |
| 3 | v1 | CITB overlay | queued | 40.4 | — | smoke | replay_ratio 0.5 |
| 3 | v1 | Standard overlay | queued | 75.8 | — | smoke | class-cov SSRG |

**GPU:** ToDCL anchor (`lora-ours-todcl-adapter-anchor`) active → v1 overlay queue after release.

**Anchors locked:** `docs/experiments/published_base_anchors_locked_20260708.md`

*Updated: 2026-07-08T12:15+08:00*
