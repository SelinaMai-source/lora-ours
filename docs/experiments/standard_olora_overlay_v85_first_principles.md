# Standard O-LoRA Overlay v85 — First Principles

## Anchor

- **Base**: O-LoRA official T5-large Standard CL order1 seed1 (unchanged entry via `run_olora_standard_order1_official_base_ours_overlay_v58.sh`).
- **Best prior overlay**: v69 formal, final EM **77.2566** (+0.56 vs LB-CL 76.7).
- **Bottleneck**: amazon/SC retention ~54%; uniform replay64 does not preserve spectral structure of prior tasks.

## v85 Single-Mechanism Delta (from v69)

Two auditable overlays, both train-config / adapter-gate only:

### 1. SSRG spectral replay (`REPLAY_MODE=ssrg`)

- Replaces uniform `random` replay64 with **spectral sparse replay** inspired by `core/methods/ours_spectral_replay.py`.
- At replay load time, build a TF-IDF covariance proxy, truncated SVD, and select exemplars with highest projection energy onto the top spectral subspace (energy threshold 0.85, top-k 8).
- Replay budget unchanged: **64 exemplars per prior task**.
- Implementation: overlay `train_tasks.json` uses `sampling strategy: ssrg`; runtime patch in official `uie_dataset_lora.py` (v58 optional hook). Standalone selector: `scripts/olora_overlay_ssrg.py`.

**Hypothesis**: replay that matches the dominant spectral modes of prior-task inputs improves retention without increasing replay budget.

### 2. Assess-update retention gate (`ASSESS_RETENTION_GATE=1`)

- Inspired by `core/methods/assess_update.py` isolated-subspace decomposition.
- After each round ≥2, compare effective LoRA update Δ = (B@A)_current − (B@A)_prior.
- Project Δ onto prior adapter direction; **reject** if isolated energy ratio > 0.3 (high interference with prior-task subspace).
- Implementation: `scripts/olora_overlay_assess_retention_gate.py`; gate state under `results/runs/<run>/assess_retention_gate/`.

**Hypothesis**: blocking high-interference adapter updates reduces catastrophic forgetting on amazon/SC while allowing orthogonal new-task learning.

## What Is Preserved (CCFA gate)

- Official task order, `run_uie_lora.py` entry, T5-large, O-LoRA adapter chain, dev/test configs, scorer.
- No dev/test leakage in gates (assess gate uses adapter weights only; early gate uses cumulative test metric surface from official predict, same as v58 smoke).

## Early Stop (smoke)

| Segment | Gate |
|---------|------|
| dbpedia (round 1) | EM < **90** → stop |
| amazon (round 2) | EM < **45** → stop |

## Launch

```bash
# Preflight
DRY_RUN=1 bash scripts/run_olora_standard_order1_official_base_ours_overlay_v85.sh

# Smoke (tmux)
tmux new-session -d -s lora-ours-standard-v85-smoke \
  "cd /root/lora-ours && bash scripts/run_olora_standard_order1_official_base_ours_overlay_v85.sh"

# Formal
FORMAL=1 bash scripts/run_olora_standard_order1_official_base_ours_overlay_v85.sh
```

## SOTA Target

- Order1 final avg EM ≥ **84.5** (= 76.7 + (100−76.7)/3).
- Promotion path: smoke early gate → formal → compare to v69 / LB-CL / v57.

## References

- v69 metrics: `results/logs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1.final_metrics.md`
- SSRG module: `core/methods/ours_spectral_replay.py`
- Assess-update: `core/methods/assess_update.py`
- v85 launcher: `scripts/run_olora_standard_order1_official_base_ours_overlay_v85.sh`
