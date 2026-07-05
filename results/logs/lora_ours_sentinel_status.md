# Lora-Ours Three-Suite Sentinel

- updated: `2026-07-05 16:39:32 UTC`
- gpu_compute_empty: `True`
- train_process_count: `0`
- pending_action: `failure_analysis`
- wake_reason: `suite=Standard state=incomplete`

## Suites

| Suite | State | Run | SOTA gap | Status file | Wake |
| --- | --- | --- | --- | --- | --- |
| CITB | completed | `citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_ft_instr_stage1_v54` | no ROUGE-L AR yet (target 53.87) | `results/logs/citb_official_script_500_50_50_tie_fixed_status.json` | no |
| Standard | incomplete | `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v58` | 67.2632 / target 84.5 (gap +17.2368) | `results/logs/standard_peft_ours_v58_status.json` | yes |
| Dialogue | completed | `dialogue_nlg_arper_woz3_dialogue_act_seed1_ours_strict_v68_formal` | 0.1060 / target 0.935 (gap +0.8290) | `results/logs/dialogue_arper_woz3_v68_formal_status.json` | no |

## Needs Agent Attention

- **Standard**: `incomplete` — suite=Standard state=incomplete (`results/logs/standard_peft_ours_v58_status.json`)

## Monitor

- JSON: `results/logs/lora_ours_sentinel_status.json`
- log: `results/logs/lora_ours_sentinel.monitor.log`
- wake flag: `SOTA_AGENT_WAKE.flag`
- agent loop tmux: `lora-ours-agent-loop`
