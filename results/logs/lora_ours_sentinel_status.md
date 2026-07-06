# Lora-Ours Three-Suite Sentinel

- updated: `2026-07-06 05:41:59 UTC`
- gpu_compute_empty: `False`
- train_process_count: `0`
- pending_action: `failure_analysis`
- wake_reason: `suite=Standard state=incomplete; suite=Dialogue state=completed_or_stopped`

## Suites

| Suite | State | Run | SOTA gap | Status file | Wake |
| --- | --- | --- | --- | --- | --- |
| CITB | completed | `citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_ft_instr_stage1_v54` | no ROUGE-L AR yet (target 53.87) | `results/logs/citb_official_script_500_50_50_tie_fixed_status.json` | no |
| Standard | incomplete | `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v58` | 67.2632 / target 84.5 (gap +17.2368) | `results/logs/standard_peft_ours_v58_status.json` | yes |
| Dialogue | completed_or_stopped | `arper_woz3_official_sclstm_formal_v86` | no BLEU-4 yet (target 0.935) | `results/logs/arper_woz3_official_sclstm_formal_v86_status.json` | yes |

## Needs Agent Attention

- **Standard**: `incomplete` — suite=Standard state=incomplete (`results/logs/standard_peft_ours_v58_status.json`)
- **Dialogue**: `completed_or_stopped` — suite=Dialogue state=completed_or_stopped (`results/logs/arper_woz3_official_sclstm_formal_v86_status.json`)

## Monitor

- JSON: `results/logs/lora_ours_sentinel_status.json`
- log: `results/logs/lora_ours_sentinel.monitor.log`
- wake flag: `SOTA_AGENT_WAKE.flag`
- agent loop tmux: `lora-ours-agent-loop`
