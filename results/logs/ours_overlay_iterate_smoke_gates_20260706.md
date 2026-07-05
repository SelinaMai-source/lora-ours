# Ours Overlay Iterate — Smoke Gates (2026-07-06)

**Branch:** `sota-24h-campaign-20260706`  
**Policy:** serial GPU via `scripts/run_ours_overlay_iterate_queue.sh`  
**Queue log:** `results/logs/ours_overlay_iterate_queue_20260706.log`  
**State JSON:** `results/logs/ours_overlay_iterate_queue_20260706.json`

## Version map

| Job ID | Version | Launcher / config | W&B group | Gate |
|--------|---------|-------------------|-----------|------|
| citb-replay50-smoke | base v55 | `run_citb_instrdialog_replay50_official_base_repro.sh` `SMOKE=1` | `lora-ours-v55-citb-replay50-base` | **PASS** EM/ROUGE-L 50.0 |
| standard-v85-smoke | overlay v85 | `run_olora_standard_order1_official_base_ours_overlay_v85.sh` | `published-base-standard-olora-plus-ours-overlay-v85-smoke` | **running** |
| arper-v86-formal | anchor v86 | `run_arper_woz3_official_sclstm_formal_v86.sh` | n/a (official Path B) | queued |
| todcl-adapter-anchor | anchor | `run_todcl_adapter_nlg_official_anchor.sh` | n/a | queued |
| citb-ours-v85-smoke | ours v85 | `citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml` | `ccfa_citb_strict_ours_v85_smoke` | queued |

## v85 overlay switches (Standard)

- `REPLAY_MODE=ssrg` — spectral sparse replay64
- `ASSESS_RETENTION_GATE=1` — reject high-interference LoRA updates (threshold 0.3)
- `EARLY_GATE_DBPEDIA_EM=90`, `EARLY_GATE_AMAZON_EM=45` — smoke early stop
- Docs: `docs/experiments/standard_olora_overlay_v85_first_principles.md`

## Smoke gate criteria (campaign)

1. **CITB base:** 1-task smoke exits 0; collator preflight ok; W&B sync.
2. **Standard v85:** 4-round smoke completes; dbpedia EM ≥ 90 or documented gate stop; no OOM/traceback.
3. **ARPER v86:** formal train starts; monitor JSON updates; compare to v66 BLEU 0.632 / SER 4.82 at completion.
4. **ToDCL:** preflight JSON pass; train log non-empty; 37-domain ADAPTER NLG.
5. **CITB ours v85:** `core.train` 5-segment smoke; SSRG enabled; no task574 patches.

## Blockers

- **GPU serial violation:** CITB formal repro (no `SMOKE=1`) may share GPU0 with v85 smoke (~26GB combined). Queue waits on `gpu_empty`; ARPER/ToDCL blocked until both release GPU.
- **Root `/` disk:** 94% used; submodule worktrees on `/` remain blocked — use autodl-tmp mirrors only.
