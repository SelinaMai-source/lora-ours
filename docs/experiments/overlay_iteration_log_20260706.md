# Overlay Iteration Log — 2026-07-06

**Branch:** `sota-24h-campaign-20260706`  
**Serial queue:** `scripts/run_ours_overlay_iterate_queue.sh` (tmux `lora-ours-overlay-queue`)  
**Gate artifacts:** `results/logs/ours_overlay_iterate_smoke_gates_20260706.md`  
**Main table:** `results/tables/sota_main_table_20260706.md`

## Version matrix (campaign queue)

| # | Job ID | Version | Suite | Launcher / config | tmux | W&B group | State | Gate |
|---|--------|---------|-------|-------------------|------|-----------|-------|------|
| 1 | citb-replay50-smoke | base v55 | CITB | `run_citb_instrdialog_replay50_official_base_repro.sh` `SMOKE=1` | — | `lora-ours-v55-citb-replay50-base` | **PASS** | EM/ROUGE-L **50.0** |
| 2 | standard-v85-smoke | overlay v85 | Standard | `run_olora_standard_order1_official_base_ours_overlay_v85.sh` | — | `published-base-standard-olora-plus-ours-overlay-v85-smoke` | **FAIL→fix** | round2 `NameError: re` fixed; relaunch queued |
| 3 | arper-v86-formal | anchor v86 | Dialogue ARPER | `run_arper_woz3_official_sclstm_formal_v86.sh` | `lora-ours-arper-v86-formal` | n/a | **running** | Inform epoch 2+, SER improving |
| 4 | todcl-adapter-anchor | anchor | Dialogue ToDCL | `run_todcl_adapter_nlg_official_anchor.sh` | — | n/a | **preflight PASS** | launch queued after GPU |
| 5 | citb-ours-v85-smoke | ours v85 | CITB | `citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml` | — | `ccfa_citb_strict_ours_v85_smoke` | **queued** | after Standard v85 + ARPER |

## Fixes applied (2026-07-06)

1. Queue `SMOKE=1` for CITB job — prevents accidental full formal on serial GPU.
2. SSRG patch: inject `import re` in `uie_dataset_lora.py` (`11c8ecc`).
3. Stopped accidental CITB formal relaunch after smoke PASS.

## ARPER v86 formal — RUNNING

- Launched 00:45 UTC+8; tmux `lora-ours-arper-v86-formal` (train + monitor).
- Task Inform epoch 2: valid SER **14.455** (down from 32.227 ep0).
- Log: `results/logs/arper_woz3_official_sclstm_formal_v86.log`
- Monitor: `results/logs/arper_woz3_official_sclstm_formal_v86_status.md`
- Paper target: BLEU **0.701** / SER **3.63**; v66 ref BLEU **0.632** / SER **4.82**.

## ToDCL preflight — PASS

- `train_total=31425`, `dev=4035`, `test=4742`, 37 domains.
- JSON: `results/logs/todcl_adapter_nlg_official_anchor_20260706_preflight.json`

## Standard v85 — pending relaunch

- Round1 dbpedia PASS (`qrvfbn3t`); round2 amazon failed SSRG `NameError` (fixed).
- Failure note: `results/logs/olora_v85_smoke_failure_20260706.md`
- Relaunch after ARPER v86 completes (serial GPU).

## SOTA targets (unchanged)

| Suite | Target | Best ours so far |
|-------|--------|------------------|
| CITB ROUGE-L AR | **53.87** | v54 **33.1** |
| Standard avg EM | **84.5** | v69 **77.26** |
| Dialogue BLEU | **0.935** | ARPER v66 **0.632** |

**Honest gap:** no numerical SOTA margin reached in this campaign slice.
