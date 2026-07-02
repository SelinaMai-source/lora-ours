# Ours v1 Strict Iteration Alignment Note

Generated: 2026-07-02

This note records implementation alignment only. It is not a benchmark result report.

## Reference Material Status

- RP(Lora)_v2 / RP(Lora)_v3 source documents were not found in the repository by filename/content search.
- Available substitute material: `scripts/build_rp_lora_v3.py`, generated-report text embedded in that script, and the v0 audit in `docs/official_alignment/v0_setup_audit.md`.

## Official Settings Preserved

- CITB InstrDialog config keeps the official T5-small LM-adapt + 100 SuperNI stage-1 checkpoint path, 19-task processed stream, seed, 15 epochs, batch size 8, lr `1e-5`, ROUGE-L AR/BWT/FWT postprocess path, and greedy generation.
- Standard PEFT CL configs keep T5-large, official O-LoRA task orders/seeds, lr `1e-3`, one epoch, batch size 8, and final average accuracy/forgetting/BWT reporting.
- Dialogue NLG configs keep ARPER/WOZ3 processed stream references and BLEU-4/SER reporting; ToDCL remains a separate strict target and is not claimed by this v1 config.

## Ours v1 Internal Changes

- `eval_normalization.task_score_metric: rouge_l` is enabled for the CITB v1 config so the task-aware score matrix tracks generation quality instead of exact-match only.
- `assess_update.isolated_energy_threshold` is configurable and set lower in the v1 CITB config to reduce drift/branch-spawn miss risk.
- Router configuration enables prototype calibration, NLL arbitration, spawn-synced prototype init, segment anchor prototype refresh, and one extra low-oracle recalibration pass.
- Spectral replay is enabled at a conservative 15% ratio to reinforce generation targets without changing official data/order/split.
- Early-stop monitoring now writes `stop_and_diagnose.json` before aborting an unpromising run.

## Non-Claims

- No SOTA or benchmark result is claimed until full strict runs complete and official postprocess outputs are produced.
- Any toy/debug smoke output remains pipeline health only.
