# SOTA Main Table — lora-ours 三套件 (2026-07-06)

**Branch:** `sota-24h-campaign-20260706`  
**W&B project:** `lora-ours`  
**Campaign status:** CITB Replay(50) smoke PASS; Standard v85 smoke **rerunning** (solo GPU); ARPER/ToDCL queued. **No numerical SOTA margin yet.**

## Campaign smoke snapshot (2026-07-06)

| Job | Key metric | Value | W&B |
|-----|------------|-------|-----|
| CITB Replay(50) base | EM / ROUGE-L (1-task smoke) | **50.0 / 50.0** | `a2jh0n4y` |
| Standard v85 overlay | cumulative EM | **pending rerun** | `qrvfbn3t` (round1) |

## Main results (order1 seed1)

| Suite | Setting label | Published base | Ours overlay (best) | Δ | SOTA target | Gap |
|-------|---------------|----------------|---------------------|---|-------------|-----|
| **CITB** | `official_script_500_50_50` | Replay(50) ROUGE-L AR **40.4** (paper) | v47 smoke TA AR **25.0**; v54 FT **33.1** | — | **53.87** | Large (−20.8 vs target) |
| **Standard** | `O-LoRA official-equivalent` single-GPU | O-LoRA v57 EM **76.81**; LB-CL ref **76.7** | v69 overlay EM **77.26**; v85 smoke **n/a** | +0.45 vs v57 | **84.5** | −7.2 EM (v69) |
| **Dialogue ARPER** | `ARPER Path B official` SCLSTM | Paper BLEU **0.701** / SER **3.63** | v66 formal BLEU **0.632** / SER **4.82** | Below paper | BLEU **0.935** / SER **2.72** | Large |
| **Dialogue ToDCL** | `ADAPTER NLG` 37-domain | README BLEU **21.77** / EER **0.164** | Bounded smoke only | — | BLEU **29.0** / EER **0.123** | Anchor pending |

## Disclosure notes (CCFA)

1. **CITB:** Main table uses `official_script_500_50_50`; do not mix with paper `500/50/100`. Four short tasks in order1 (preflight exit 3) are expected for official short-stream. Replay50 smoke PASS is single-task capped — not paper-comparable formal.
2. **Standard:** v57/v69/v85 use single-GPU + `grad_accum=8` official-equivalent; differs from literal 8-GPU `order_1.sh`. Overlay rows are **published-base + ours-overlay**, not official-base. v85 smoke incomplete (no eval).
3. **Dialogue:** Path A T5 `core.train` ≠ official SCLSTM Path B. Primary claim must use **published-base + overlay** on SCLSTM/Adapter.
4. **ToDCL:** Legacy py37 env; full anchor queued (`lora-ours-todcl-adapter-anchor`).

## Ablation config stubs (RP contribution points)

| Switch | Config stub | Suite |
|--------|-------------|-------|
| `-drift` | `ours.drift_detector.enabled: false` | all |
| `-bank` | `lora_bank.max_branches: 1` | CITB/Standard Path A |
| `-router` | `router.enabled: false` | Path A |
| `-overlap` | `overlap_loss.weight: 0` | all |
| `-assess_update` | `assess_update.enabled: false` | all |
| `-ssrg` | `spectral_replay.enabled: false` / `REPLAY_MODE=uniform` | all |

Stub paths:

- CITB ours: `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml`
- Standard overlay: `scripts/run_olora_standard_order1_official_base_ours_overlay_v85.sh` (env `REPLAY_MODE=uniform` for `-ssrg`)
- ARPER SSRG stub: `configs/ccfa_three_suite/arper_woz3_official_sclstm_v86_ssrg_exemplar_overlay_stub.yaml`
- ToDCL stub: `configs/ccfa_three_suite/todcl_ours_overlay_stub.yaml`

## Next seeds/orders (post-SOTA)

- Standard/CITB: 3 seeds × 3 orders after first suite hits margin target.
- Dialogue: paper-epoch v86 anchor → overlay v87+.

## Honest gaps (2026-07-06 campaign close)

| Item | Status |
|------|--------|
| CITB Replay50 base smoke | **PASS** (`a2jh0n4y`); accidental formal stopped |
| Standard v85 smoke | **RERUNNING** after GPU contention failure |
| CITB ours v85 | **queued** |
| ARPER v86 formal | **queued** (paper-epoch re-anchor) |
| ToDCL ADAPTER anchor | **queued** (preflight PASS) |
| Numerical SOTA | **not reached** on any suite |
| 3 seeds × 3 orders | **deferred** until first margin hit |
| Full ablation grid | **stub paths only**; not executed |
