# Best-Method Reproduction — Final Summary (2026-07-06)

**Generated:** 2026-07-07 04:02 UTC+8  
**Branch:** `best-method-repro-results-20260706`  
**Overall:** `complete_with_blocker`

## Per-suite best methods

| Suite | Best method | Paper | Local | Match? | Notes |
|-------|-------------|-------|-------|--------|-------|
| CITB | Replay(50) formal 19-task | ROUGE-L AR 40.4 | ROUGE-L AR 32.4417 | No | official_script_500_50_50; gap −7.96 vs paper |
| Standard | O-LoRA v57 formal | EM avg 75.8 | EM 76.8059 | Yes / Close | single-GPU grad_accum=8 official-equivalent |
| Dialogue ToDCL | ADAPTER 37-domain NLG anchor | BLEU 21.77 / EER 0.164 | — | — | gpt2 load failure EXIT=1 |
| Dialogue ARPER | v87 paper-aligned domain/exemplar500 | BLEU 0.701 / SER 3.63 | BLEU 0.60059 SER error: | TBD | granularity=0 exemplar 500 |

## W&B / tmux

| Suite | tmux session | W&B run / group |
|-------|--------------|-----------------|
| citb_replay50 | `lora-ours-citb-replay50-formal` | `4r1vg0x9 / citb_instrdialog_order1_official_replay50_base_repro` |
| standard_olora_v57 | `(completed prior)` | `published-base-standard-olora-order1-formal-v57` |
| todcl_adapter | `lora-ours-todcl-adapter-anchor` | `—` |
| arper_v87 | `lora-ours-arper-v87-formal` | `—` |

## Launchers

- CITB Replay(50): `scripts/run_citb_instrdialog_replay50_official_base_repro.sh`
- Standard O-LoRA v57: `scripts/run_standard_all_baselines_repro.sh` (METHOD=olora)
- ToDCL ADAPTER: `scripts/run_todcl_adapter_nlg_official_anchor.sh`
- ARPER v87: `scripts/run_arper_woz3_paper_aligned_formal_v87.sh`

## Setting disclosures

1. CITB: official-script `500/50/50` (paper text `500/50/100` not in released scripts).
2. Standard v57: single-GPU + `grad_accum=8` official-equivalent.
3. LB-CL: `paper_only` (no code); O-LoRA v57 is best-available anchor.
4. ARPER v87: domain-wise `granularity=0`, exemplar 500, `task_seq=0,5,2,1,3,4`.
5. ToDCL: legacy py37 env, GPT-2 base required for paper anchor.

## Blockers

- ToDCL ADAPTER: corrupt/missing GPT-2 weights; network unreachable for re-download

