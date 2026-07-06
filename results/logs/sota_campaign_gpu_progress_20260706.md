# SOTA GPU Campaign Progress — 2026-07-06 13:42 CST

**Branch:** `sota-24h-campaign-20260706`

## Standard v85 smoke (`lora-ours-standard-v85-smoke`)

| Round | Task | Status | EM | Gate |
|-------|------|--------|-----|------|
| 1 | dbpedia | **PASS** | **98.5** | ≥90 |
| 2 | amazon | **running** (dataset prep / train) | — | ≥45 |
| 3 | yahoo | queued | — | — |
| 4 | agnews | queued | — | — |

- GPU: python PID 8465 (~386 MiB); healthy — do not kill.
- Cumulative avg EM vs v69 (**77.26**) / SOTA (**84.5**): **pending** (need all 4 rounds).
- Formal queue (`standard_olora_overlay_v85_formal.yaml`): **hold** until smoke final.
- Standard v86 fallback (single mechanism): enable `sc_class_coverage_order` replay path (v83 diag) or relax assess gate per gate doc — **draft only if smoke fails v69**.

## ToDCL anchor queue (`lora-ours-sota-gpu-queue`)

- Script: `scripts/lora_ours_gpu_queue_continue.sh` — polls every 60s until `v85_running=no` **and** `gpu_idle=yes`, then launches `run_todcl_adapter_nlg_official_anchor.sh` in `lora-ours-todcl-adapter-anchor`.
- Verified polling at 13:37–13:41: `waiting v85=yes gpu_idle=no` (expected while v85 holds GPU).

## CITB v86 (replay_ratio 0.5)

- Plan: `docs/experiments/citb_ours_overlay_v86_first_principles.md`
- **Launch:** only after GPU free **and** this progress note + v85 smoke final logged.

## Disk

- `/` overlay: **3.2G** free (90%).
