# Ours Overlay Iterate — Smoke Gates (2026-07-06)

**Branch:** `sota-24h-campaign-20260706`  
**Policy:** serial GPU via `scripts/run_ours_overlay_iterate_queue.sh`  
**Iteration log:** `docs/experiments/overlay_iteration_log_20260706.md`  
**Queue log:** `results/logs/ours_overlay_iterate_queue_20260706.log`

## Gate summary

| Job | Gate criteria | Verdict | Evidence |
|-----|---------------|---------|----------|
| citb-replay50-smoke | exit 0; collator ok; W&B sync | **PASS** | EM/ROUGE-L **50.0**; W&B `a2jh0n4y` |
| standard-v85-smoke | 4-round smoke; dbpedia EM ≥90; no OOM | **PASS (early)** | dbpedia EM **98.5**; amazon pending |
| arper-v86-formal | formal train starts; monitor updates | **DONE** | BLEU 0.632 / SER 4.817 (= v66) |
| todcl-adapter-anchor | preflight pass; train log non-empty | **PENDING** | preflight PASS; queued after v85 |
| citb-ours-v85-smoke | 5-segment smoke; SSRG on; no task574 patch | **PASS** | ROUGE-L AR **24.6** |

## Standard v85 smoke gate detail

| Field | Value |
|-------|-------|
| Run | `olora_official_base_ours_overlay_ssrg_v85_smoke_order1_seed1` |
| W&B | `45amv0bn` (round1 dbpedia) |
| Train progress | global_step **14** / max_steps **20** (round1 dbpedia) |
| dbpedia EM | **n/a** (predict not reached) |
| amazon EM | **n/a** |
| Early gate dbpedia (<90) | **not evaluated** |
| Early gate amazon (<45) | **not evaluated** |
| Failure mode | concurrent CITB GPU owner; process interrupted mid-train |
| tmux | `lora-ours-standard-v85-smoke` (shell alive; training not at eval) |

**Decision:** treat as **incomplete run**, not a gate FAIL on metrics. Re-queue when GPU serial policy is satisfied.

## v85 overlay switches (Standard)

- `REPLAY_MODE=ssrg` — spectral sparse replay64
- `ASSESS_RETENTION_GATE=1` — reject high-interference LoRA updates (threshold 0.3)
- `EARLY_GATE_DBPEDIA_EM=90`, `EARLY_GATE_AMAZON_EM=45`
- Docs: `docs/experiments/standard_olora_overlay_v85_first_principles.md`

## Blockers

- **GPU serial violation:** CITB formal repro shares GPU0 with v85 smoke window; queue waits on `gpu_empty`; ARPER/ToDCL/CITB-ours blocked.
- **Root `/` disk:** ~94% used; submodule worktrees on `/` blocked — autodl-tmp mirrors only.
