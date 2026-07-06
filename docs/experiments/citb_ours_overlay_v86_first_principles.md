# CITB Ours Overlay v86 — First Principles (single mechanism)

**Status:** doc only — do **not** launch while GPU busy (Standard v85 → ToDCL queue).  
**Branch:** `sota-24h-campaign-20260706`

## Anchor

- **Base config:** `citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml` (v85 smoke completed).
- **v85 result:** ROUGE-L AR **24.6**; early gate PASS (seg2 seen TA 31.25%) but far below Replay(50) smoke **50.0** and SOTA **53.87**.
- **Diagnosis:** retention collapses after segment 1 — pubmedqa TA 53% → 26% and stays flat; generation segments (565/1714/574) raw accuracy → 0.0 despite SSRG on. v85 uses `replay_ratio: 0.2`, `min_replay_per_segment: 8` (~8% batch replay) while the published Replay(50) anchor uses **50% replay**.

## v86 Single-Mechanism Delta (from v85)

### Replay50 ratio alignment (`spectral_replay.replay_ratio: 0.5`)

- **Change only:** set `replay_ratio` **0.2 → 0.5** and `min_replay_per_segment` **8 → 64** (match official Replay(50) budget per segment).
- **Unchanged:** SSRG spectral selection (`spectral_top_k: 8`, `energy_threshold: 0.85`), router/drift/overlap stack, eval caps, no task574 patches.
- **Hypothesis:** v85 under-replays relative to the published base it overlays; aligning replay mass to Replay(50) should lift seen-average ROUGE-L AR without new modules.

## Early gate (smoke, 5 segments)

| Checkpoint | Criterion |
|------------|-----------|
| segment 2 | seen TA ROUGE-L AR ≥ **35** (v85 was 31.25) |
| segment 4 | seen TA ROUGE-L AR ≥ **40** (Replay50 paper 40.4) |

Stop if segment 2 seen TA AR < **20** (below v85 floor).

## Launch (when GPU free)

```bash
# Config fork only — not created until approved
# configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v86_smoke_strict.yaml

DRY_RUN=1 python -m core.train --config configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v86_smoke_strict.yaml
```

## SOTA target

- ROUGE-L AR ≥ **53.87** (= Replay50 paper 40.4 + (53.87−40.4)/3 margin path).
- Promotion: smoke early gate → 19-task formal only after Replay(50) official base anchor formal completes.

## References

- v85 metrics: `results/logs/citb_instrdialog_order1_seed1_ours_v85_smoke_strict_status.md`
- v85 config: `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml`
- Replay50 smoke: W&B `a2jh0n4y` (EM/ROUGE-L 50.0)
