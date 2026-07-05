# Standard O-LoRA Replay128 Overlay v70b Formal Launch

- Updated: 2026-07-05
- Run: `olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1`
- Base: O-LoRA official T5-large Standard CL order1 seed1.
- Overlay: ours limited prior-task replay overlay, `REPLAY_PER_TASK=128`.
- Setting: formal no smoke caps; order dbpedia -> amazon -> yahoo -> agnews; seed 1.
- W&B: project `lora-ours`, group `published-base-standard-olora-plus-ours-overlay-v70b-formal`.
- Monitor: sentinel heartbeat JSONL plus tmux session `standard-olora-overlay-v70b-formal`.
- Verification: manifest `max_steps=-1`; `max_predict_samples=null`; active round1 train command has no `--max_steps` or `--max_predict_samples`.
- Boundary: this is an ours-overlay formal candidate on the O-LoRA official-base runtime, not a replacement for the official O-LoRA baseline.
