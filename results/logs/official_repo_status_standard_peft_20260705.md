# Standard PEFT CL Official Repo / Code Status

- Updated: `2026-07-05T21:25+08:00`
- Selected runnable base for ours: O-LoRA official T5-large Standard CL order1.
- Current best ours anchored on this base: v69 formal EM `77.2566`, ROUGE-L `81.1919`.

## O-LoRA

- Official repo: `https://github.com/cmnfriend/O-LoRA`
- Local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora`
- Commit: `07117e1fc4a5f5ad9308a815a42cee8f46502dc8`
- Dirty status: only Python `__pycache__` artifacts and `src/model/__pycache__/`; source files remain effectively pristine for official reference.
- Environment: `/root/autodl-tmp/conda_envs/lora_v10_o_lora`.
- Checkpoint/model: `/root/autodl-tmp/model_cache/hf_snapshots/t5-large`.
- Official script: `scripts/order_1.sh`.
- Official script setting:
  - order: `dbpedia -> amazon -> yahoo -> agnews`
  - `model_name_or_path=initial_model/t5-large`
  - 8-GPU DeepSpeed
  - per-device train batch `8`, eval batch `128`
  - grad accumulation `1`
  - LR `1e-03`, epochs `1`
  - `max_source_length=512`, `max_target_length=50`, `generation_max_length=50`
  - `lamda_1=0.5`, `lamda_2=0`
- Local official-equivalent reproduction:
  - Run: `olora_t5large_standard_order1_seed1_official_base_formal_v57`
  - Status: `results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_status.md`
  - Manifest: `results/runs/olora_t5large_standard_order1_seed1_official_base_formal_v57/run_manifest.json`
  - Result: final EM `76.8059`, ROUGE-L `79.9715`
  - Engineering delta: single-GPU runtime copy with grad accumulation `8` to match official global batch; W&B env patch; no sample/step caps in formal.
- Ours overlay already anchored:
  - Run: `olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1`
  - Result: EM `77.2566`, ROUGE-L `81.1919`
  - Status: official-base overlay, not a replacement base.

## LFPT5

- Repo: `https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning`
- Local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/lfpt5`
- Commit: `cf7d17ce7de6a707d929d0542b3d5e639569855f`
- Status: downloaded and clean; not yet rerun in this session as selected base.
- Role: published baseline/reference for Standard CL, not current ours base.

## Progressive Prompts

- Repo: `https://github.com/arazd/ProgressivePrompts`
- Local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/progressive_prompts`
- Commit: `01572d6a73c0576b070ceee00dbe4f5bc278423f`
- Status: downloaded and clean; not yet rerun in this session as selected base.
- Role: published baseline/reference for Standard CL, task-aware prompt baseline.

## LB-CL

- Published method: Learn more, but bother less: parameter efficient continual learning.
- Paper/source verified via web search:
  - NeurIPS/OpenReview paper reports T5-large Standard CL LB-CL `76.9/76.5/76.8`, avg `76.7`.
- Additional search on `2026-07-05`:
  - Web search for `"Learn more, but bother less" LB-CL code GitHub`, `"LB-CL" "Learn more, but bother less" GitHub`, and OpenReview ID `ZxtaNh5UYB` found paper/OpenReview/NeurIPS pages but no canonical code repository.
  - `gh` CLI is not installed in this environment, so GitHub API search could not be run locally.
  - Direct OpenReview fetch is blocked by browser verification; web snippets and downloaded paper PDFs are available, but no code link was exposed there.
- Dedicated official repo: not found in current local external sources and not found via web search.
- Status: paper result can be cited as a published baseline, but code-level reproducibility remains a blocker until an official/author repository or supplement code is located. Do not use third-party code as LB-CL official code.

## Gate Decision

- Standard ours increments are allowed only when they preserve the O-LoRA official-base path above.
- New candidates that modify data/order/eval beyond train-only diagnostic gates must be paused until documented as official-comparable.
- v84 moderate curriculum was stopped under this new gate before completion; no v84 result should be used or promoted.
