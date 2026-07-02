# v0 Setup Audit

Generated: 2026-07-02 09:10 CST

This document is an audit trail for the first setup phase. It records only observed
state, command results, and explicit blockers. It is not a benchmark result report.

## Repository State

- Requested target: `/root/lora-ours`
- GitHub remote: `git@github.com:SelinaMai-source/lora-ours.git`
- SSH remote probe succeeded before clone fallback: `git ls-remote ... HEAD` returned `514a0c5356d8c4f86ef6ff24d9eec8154808bc83`.
- Direct SSH clone into `/root/lora-ours` printed only `Cloning into ...` and made no progress within 120 seconds, so it was terminated.
- A clean working tree was recovered from the existing local repository `/root/lora-ours-agent-workspace/lora-ours`, then `origin` was set back to the requested GitHub remote.
- Current setup branch: `ours-v0-setup`.
- Source local repository was on `ours_v10` at `88878d0 fix: resolve KeyError proj_vec in core/train.py`.
- Existing source workspace had unrelated dirty files before this phase: several `__pycache__/*.pyc` files and `test_delta.py`. They were not modified.

## Local Material Discovery

Observed under `/root`:

- Existing related workspace: `/root/lora-ours-agent-workspace/lora-ours`.
- Related checklist/research docs: `CL_Experiments_Alignment_Checklist.md`, `CL_Experiments_Research.md`, and PDF/DOCX versions.
- RP(Lora) / RP(LoRA) v2/v3 document files were not found by filename glob searches. The only RP-LoRA-related project file found was `scripts/build_rp_lora_v3.py`; a Cursor plan file `.cursor/plans/rp_lora_completion_000b6d3e.plan.md` also exists outside the repo.

## Environment Snapshot

Observed by command:

- `git version 2.34.1`
- `tmux 3.2a`
- `python 3.12.3`
- `torch 2.8.0+cu128`
- CUDA available: `True`
- CUDA devices: `1`
- CUDA device name: `NVIDIA vGPU-48GB`
- `wandb` CLI exists at `/root/miniconda3/bin/wandb`
- `wandb` Python package version: `0.27.0`
- Current cloned repo does not include `data/` or `assets/`; official data and model checkpoints are therefore not locally ready.

## Official Remote Probes

Two probe passes were run. An initial parallel HTTPS pass with 30 second shell
timeouts saw CITB, O-LoRA, and ProgressivePrompts time out. A later scripted
preflight pass with 8 second per-repo timeouts succeeded for all six official
repos and wrote `results/preflight/phase1_preflight_latest.json`.

| Suite | Repo | Observed status |
| --- | --- | --- |
| CITB | `https://github.com/hyintell/CITB` | reachable in preflight, HEAD `bf50533b5bced4c388691ecc75e26773da96b3fd` |
| O-LoRA | `https://github.com/cmnfriend/O-LoRA` | reachable in preflight, HEAD `07117e1fc4a5f5ad9308a815a42cee8f46502dc8` |
| LFPT5 | `https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning.git` | reachable, HEAD `cf7d17ce7de6a707d929d0542b3d5e639569855f` |
| ProgressivePrompts | `https://github.com/arazd/ProgressivePrompts` | reachable in preflight, HEAD `01572d6a73c0576b070ceee00dbe4f5bc278423f` |
| ARPER / MultiWOZ NLG | `https://github.com/MiFei/Continual-Learning-for-NLG.git` | reachable, HEAD `99019defe6bf35e8459ca6abd6f25882724bc956` |
| ToDCL | `https://github.com/andreamad8/ToDCL.git` | reachable, HEAD `e70c1edf937f6eb570296ea2897dbc8d6815bc6d` |

The repo already contains older baseline notes under `baselines/advanced_baselines/`.
Those notes mark O-LoRA, LB-CL, Progressive Prompts, Continual-T0, LFPT5, ARPER, and
ToDCL support as partial, scaffold, or blocked. Their unified-entry outputs must not
be treated as strict paper-aligned baseline results until official scripts/data/checkpoints
are audited and run.

## Required Experimental Alignment Targets

These targets are user-provided requirements and remain to be independently verified
against paper/repo artifacts as network access allows.

### CITB Continual Instruction Tuning

- Datasets: InstrDialog and InstrDialog++.
- Baselines: FT-init, L2, EWC, AGEM, Replay, Multi upper bound.
- Paper: CITB, Findings EMNLP 2023.
- Official repo: `https://github.com/hyintell/CITB`.
- Base checkpoint: LM-adapted T5-small for most methods after 100 SuperNI tasks.
- InstrDialog: 19 tasks, 500/50/100 train/dev/test.
- InstrDialog++: 38 tasks, 100/50/100 train/dev/test.
- Seeds: 3.
- Replay/AGEM memory: 10 or 50 examples per task.
- Metrics: ROUGE-L AR, FWT, BWT, Tinit retention, Tunseen retention.

### Standard T5-Large PEFT CL

- References: LFPT5, Progressive Prompts, O-LoRA, LB-CL benchmark.
- Baselines: SeqLoRA, IncLoRA, Replay, LFPT5, Progressive Prompts, O-LoRA, LB-CL, MTL.
- Official/reference repos: O-LoRA, LFPT5, ProgressivePrompts.
- Must verify exact task order, seeds, metrics, and model checkpoint before full runs.
- Metrics: final average accuracy, forgetting/BWT.
- User target baselines: LB-CL 76.7, O-LoRA 75.8, Progressive Prompts 76.1.

### Dialogue NLG / MultiWOZ CL

- References: ARPER MultiWOZ-2.0 and/or ToDCL 37-domain ToD NLG/E2E.
- ARPER repo: `https://github.com/MiFei/Continual-Learning-for-NLG`.
- ToDCL repo: `https://github.com/andreamad8/ToDCL`.
- ARPER replay budgets: 250/500 exemplars.
- ARPER metrics: BLEU-4, SER.
- ToDCL metrics: BLEU, EER.
- User target baselines: ARPER SER 3.63 / BLEU-4 0.701; ToDCL modular NLG Adapter BLEU 21.7719 / EER 0.163975.

## Current Blockers

- Official repo HEADs are reachable as of the scripted preflight, but the code has
  not yet been cloned into `external/official_repos/` for README/script auditing.
- No official data has been materialized in the fresh `/root/lora-ours` clone.
- No model checkpoint is present under `assets/`; Llama/T5 full runs cannot start.
- W&B CLI/package are installed, but login/auth was not proven in this phase.
- Existing unified advanced baselines are explicitly scaffold-level; they cannot support
  SOTA claims without official reproduction.
- The existing configs primarily target `train50/eval10` local streams and Llama-3.1
  debug/publication paths. They do not yet strictly implement the user-requested
  CITB T5-small or Standard T5-large protocols.

## v0 Guardrails

- Toy smoke data is allowed only for pipeline health checks and must be tagged
  `toy_smoke`; it is never a benchmark result.
- Full benchmark runs require preflight success for official repo metadata, data,
  checkpoint, W&B mode/auth decision, and an explicit tmux session name.
- Early segment monitoring should stop or flag a run when the first segments are
  clearly below a declared floor; do not continue blind multi-day sweeps.
- All run logs, config snapshots, final metrics, and preflight JSON must be kept.
