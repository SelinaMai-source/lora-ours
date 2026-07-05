# Standard PEFT CL Official Repo / Code Status

- Updated: `2026-07-05T22:15+08:00`
- Selected runnable base for ours: O-LoRA official T5-large Standard CL order1.
- Current best ours anchored on this base: v69 formal EM `77.2566`, ROUGE-L `81.1919`.
- Official archive branch: `official-method-code-archive-20260705` commit `b060357`.

## O-LoRA

- Official repo: `https://github.com/cmnfriend/O-LoRA`
- Local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora`
- Commit: `07117e1fc4a5f5ad9308a815a42cee8f46502dc8`
- Official archive submodule: `official_repos/standard/O-LoRA`.
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
- Official archive submodule: `official_repos/standard/Lifelong-Fewshot-Language-Learning`.
- Status: downloaded and clean; not yet rerun in this session as selected base.
- Role: published baseline/reference for Standard CL, not current ours base.

## Progressive Prompts

- Repo: `https://github.com/arazd/ProgressivePrompts`
- Local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/progressive_prompts`
- Commit: `01572d6a73c0576b070ceee00dbe4f5bc278423f`
- Official archive submodule: `official_repos/standard/ProgressivePrompts`.
- Status: downloaded and clean; not yet rerun in this session as selected base.
- Role: published baseline/reference for Standard CL, task-aware prompt baseline.

## LB-CL

- Published method: Learn more, but bother less: parameter efficient continual learning.
- Paper/source verified via web search:
  - NeurIPS/OpenReview paper reports T5-large Standard CL LB-CL `76.9/76.5/76.8`, avg `76.7`.
- Direct paper evidence:
  - NeurIPS proceedings paper: `https://proceedings.neurips.cc/paper_files/paper/2024/file/b0bc711f48724237b38823c4d9cee10b-Paper-Conference.pdf`.
  - NeurIPS abstract page: `https://proceedings.neurips.cc/paper_files/paper/2024/hash/b0bc711f48724237b38823c4d9cee10b-Abstract-Conference.html`.
  - Paper checklist item 5 says open access to data/code is `[No]` with justification: "We use open-source datasets and models, but do not attach the code."
  - Appendix A.2 gives implementation details, but no repository URL, artifact URL, or runnable command.
- Venue/profile evidence:
  - OpenReview forum ID: `ZxtaNh5UYB`; web-indexed page lists Fuli Qiao and Mehrdad Mahdavi, NeurIPS 2024 poster, CC BY 4.0 paper availability, but no code link in the indexed content. Direct local fetch is blocked by OpenReview browser verification; OpenReview API attempts returned `403`.
  - NeurIPS virtual poster page: `https://neurips.cc/virtual/2024/poster/94599`; links only Paper, Slides, Poster, and OpenReview, with no code/project link exposed.
  - NeurIPS slides: `https://neurips.cc/media/neurips-2024/Slides/94599.pdf`; method overview/results only, no code or repo URL.
  - ML Anthology page: `https://mlanthology.org/neurips/2024/qiao2024neurips-learn/`; citation/DOI only, no code field.
- Additional search on `2026-07-05`:
  - Web search for `"Learn more, but bother less" LB-CL code GitHub`, `"LB-CL" "Learn more, but bother less" GitHub`, and OpenReview ID `ZxtaNh5UYB` found paper/OpenReview/NeurIPS pages but no canonical code repository.
  - With the mihomo proxy, GitHub and HuggingFace are reachable, but unauthenticated GitHub API repository search is rate-limited (`403`).
  - Follow-up web searches for `"Fuli Qiao" "Learn More, but Bother Less" GitHub`, `"Mehrdad Mahdavi" "LB-CL" GitHub`, and `"Learn more, but bother less" "github.com"` found paper/OpenReview/NeurIPS/ML Anthology/proceedings pages and a generic author GitHub profile, but no LB-CL source repository or supplement code.
  - Further searches for `"Learn More, but Bother Less" "code" "Fuli Qiao"` and `"LB-CL" "github" "Fuli Qiao" "Mahdavi"` again found paper/index/slides pages only, with no official repo.
  - Search for `site:github.com "Learn More, but Bother Less" "LB-CL"` returned unrelated repositories and no author repository.
  - Search for `site:github.com "ZxtaNh5UYB" OR "b0bc711f48724237b38823c4d9cee10b"` returned unrelated gist/provenance results, not LB-CL code.
  - Search for `"fvq5015" GitHub OR "Fuli Qiao" "github.com"` found Google Scholar/LinkedIn and unrelated similarly named GitHub accounts, but no public GitHub account tied to Fuli Qiao / PSU / LB-CL.
  - Mehrdad Mahdavi GitHub profile `https://github.com/mehrdadmahdavi` exists but has only `github_tutorial`; no LB-CL repository.
- Non-accepted code candidates:
  - `https://github.com/yaoyz96/low-rank-cl` appears in broad search snippets, but its README identifies an ICLR 2026 paper, "Revisiting Weight Regularization for Low-Rank Continual Learning", and methods such as InfLoRA/SD-LoRA/CL-LoRA/EWC-LoRA. The owner/authors do not match Fuli Qiao or Mehrdad Mahdavi and the repository is not explicitly linked from the LB-CL paper, OpenReview, NeurIPS page, slides, or author profiles; do not treat it as LB-CL official code.
- Dedicated official repo: not found in current local external sources and not found via web search.
- License / availability: paper is publicly available via NeurIPS/OpenReview; official code is not attached and no code license is available.
- Runnable status: not runnable locally because no official/author code or supplement artifact is confirmed.
- Status: paper result can be cited as a published baseline, but code-level reproducibility remains a blocker until an official/author repository or supplement code is located. Do not use third-party code as LB-CL official code.

## Gate Decision

- Standard ours increments are allowed only when they preserve the O-LoRA official-base path above.
- New candidates that modify data/order/eval beyond train-only diagnostic gates must be paused until documented as official-comparable.
- v84 moderate curriculum was stopped under this new gate before completion; no v84 result should be used or promoted.
