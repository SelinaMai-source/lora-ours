# Standard v85 Smoke Gate — 2026-07-06

**Branch:** `sota-24h-campaign-20260706`  
**Run:** `olora_official_base_ours_overlay_ssrg_v85_smoke_order1_seed1`  
**tmux:** `lora-ours-standard-v85-smoke` (SSRG fix relaunch `11c8ecc`)  
**W&B group:** `published-base-standard-olora-plus-ours-overlay-v85-smoke`

## Round 1 dbpedia — early gate

| Field | Value |
|-------|-------|
| dbpedia EM | **98.5** |
| Threshold | ≥ **90** |
| ROUGE-L | 98.5 |
| predict_samples | 200 (smoke cap) |
| global_step | 20 / max_steps 20 |
| Artifact | `results/runs/olora_official_base_ours_overlay_ssrg_v85_smoke_order1_seed1/official_outputs/1-dbpedia/all_results.json` |
| W&B round1 | `0ze3ek2g` |

**Verdict: PASS** — dbpedia EM 98.5 ≥ 90.

## Round 2+ — pending

| Round | Status | EM | Gate |
|-------|--------|-----|------|
| amazon (round 2) | **running** | n/a | ≥45 |
| yahoo (round 3) | queued | n/a | — |
| agnews (round 4) | queued | n/a | — |

## Full smoke verdict

**INDETERMINATE** — dbpedia early gate **PASS**; full 4-round smoke still in progress. Continue watching for amazon early gate (EM < 45 → FAIL).

## Next actions

- **PASS path:** let smoke complete all 4 rounds; compare cumulative EM to v69 (77.26) and LB-CL (76.7).
- **FAIL path (amazon EM < 45):** draft v86 with single-mechanism delta — relax `ASSESS_RETENTION_GATE` threshold (0.3 → 0.45) or disable gate on round 2 only; SSRG replay unchanged.

## Queue

ToDCL ADAPTER anchor remains queued in `lora-ours-sota-gpu-queue` via `scripts/lora_ours_gpu_queue_continue.sh`.
