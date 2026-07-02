# Ours v18 Task-Aware Fallback Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Reference Material Status

- `RP(Lora)_v2` / `RP(Lora)_v3` source documents were not found as standalone repository files by filename/content search.
- Available RP-related material:
  - `scripts/build_rp_lora_v3.py`, which embeds the generated v3 report text.
  - `.cursor/plans/rp_lora_completion_000b6d3e.plan.md`, which records the earlier implementation plan.
  - `docs/official_alignment/v0_setup_audit.md` and `docs/official_alignment/ours_v1_strict_iteration_alignment_note.md`.
- Usable proposal ideas extracted for v18: task-free drift detection, LoRA bank branching, prototype/router calibration, anti-overlap, replay, and explicit audit artifacts. No numeric claim from those documents is treated as a benchmark result.

## Early Stop of Current v17 CITB

- Running command was in `/root/autodl-tmp/Lora-code` with config `/root/autodl-tmp/Lora-Baselines/docs/configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_strict.yaml`.
- The current run repeated the old low-score trajectory:
  - Segment 2-6 current scores were `0.0`.
  - At segment 7, `seen_avg_score=0.05375`.
  - Routing collapsed to branch `b1` for segments 2-7 in the latest run.
  - The log showed repeated restarts of the same v17 CITB config before reaching later suites.
- The prior completed CITB strict run had `ROUGE-L AR=10.1579`, router train accuracy around `0.0919`, drift miss rate `0.8333`, and overlap cosine around `0.9929`.
- Based on this repeated low-efficiency evidence, v17 CITB/supervisor panes were interrupted. This is an early stop of an unpromising duplicate, not a deletion or result rewrite.

## V18 Changes

- `core/methods/router.py`
  - Added optional `router.task_aware_fallback`.
  - The router records a segment-to-branch map learned from per-segment pseudo labels.
  - During evaluation, if prototype routing is low-confidence or low-margin, it can fall back to the recorded branch for the evaluated segment.
- `core/train.py`
  - Records each segment's majority pseudo-label branch after oracle/NLL pseudo-labeling.
  - Records the active branch when all pseudo-labels are filtered out.
- `core/evaluate.py`
  - Passes the evaluated segment id separately to the router so fallback is tied to the task being evaluated, while warmup still uses the current continual step.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v18_strict.yaml`
  - Full strict CITB config with the same official model, stream, epochs, lr, batch size, and generation setup.
- `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v18_smoke_strict.yaml`
  - Early-gate smoke config using the same strict setup but truncated to the first 4 segments.
- `scripts/monitor_ours_v18_strict.py`
  - Writes status to `results/logs/ours_v18_strict_status.json` and `.md`.
  - Detects not-started, running, stale, low-score gate, stop-and-diagnose, and completed states.

## Official Setting Alignment

- CITB: aligned to T5-small LM-adapted + 100 SuperNI stage-1 checkpoint and the processed InstrDialog order1 split `citb_instrdialog_order1_train500_dev50_test100.json`; stage2 lr `1e-5`, epochs `15`, batch size `8`, greedy generation. The smoke config only truncates number of segments for early-gate and must not be reported as full CITB.
- Standard PEFT CL: existing strict configs use T5-large, O-LoRA standard task orders/seeds, lr `1e-3`, one epoch, batch size `8`, and final average accuracy/forgetting/BWT metrics. No v18 Standard run is launched until CITB smoke passes.
- Dialogue NLG: existing strict ARPER WOZ3 config uses official ARPER evidence and BLEU-4/SER reporting. ToDCL remains a separate strict target; v18 does not claim ToDCL coverage yet.

## Non-Claims and Next Gate

- v18 is a small corrective iteration for router collapse and replay/overlap calibration.
- The next allowed step is the CITB v18 smoke run. If it does not clear the early gate, stop and diagnose before launching full CITB or downstream suites.
