# Official Method Matrix

Updated: 2026-07-05

This matrix is the gate for future Ours work. A method can be used as an Ours base only after official code and a runnable anchor are documented.

Official source archive branch: `official-method-code-archive-20260705` at commit `b060357`, with pinned submodules and manifest in `official_repos/OFFICIAL_REPOS.md`.

## CITB Continual Instruction Tuning

Benchmark: InstrDialog / InstrDialog++ from `hyintell/CITB`.

Primary repo status:
- URL: `https://github.com/hyintell/CITB`
- local clean checkout: `/root/autodl-tmp/CITB`
- local working checkout: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/citb`
- commit: `bf50533b5bced4c388691ecc75e26773da96b3fd`
- environment: `/root/autodl-tmp/conda_envs/lora_v10_citb`
- checkpoint: `/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469`
- base model/tokenizer asset: `/root/autodl-tmp/model_cache/hf_snapshots/google__t5-small-lm-adapt`
- status doc: `results/logs/official_repo_status_citb_20260705.md`
- network/proxy: GitHub/HF reachable through temporary mihomo proxy; public web search found no extra CITB release, issue, or hidden download that resolves the paper/public split mismatch. No new CITB download is currently blocked by network; the blocker is the official setting itself.
- official archive path: `official_repos/citb/CITB` is a pinned gitlink/archive pointer; use the clean checkout `/root/autodl-tmp/CITB` at `bf50533b5bced4c388691ecc75e26773da96b3fd` for full-file audits.

| Method | Paper Result | Official Repo Found | Commit / Path | Downloaded | Runnable | Smoke / Repro Result | Reproduced | Gate | Blocker / Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FT-init | InstrDialog ROUGE-L AR `35.7`, BWT `-4.6`; released score order1 `average_train_samples=411.5` | yes | `bf50533...` / CITB repo and `official_repos/citb/CITB` | yes | yes for script-strict short stream | `v54` completed 19/19 under `500/50/50`; local aggregate `predict_official_rougeL=33.109` | partial script-smoke only | setting_blocker | Paper says InstrDialog `500/50/100`, but public short-stream scripts pass `max_num_instances_per_eval_task=50`, and split code uses the same value for dev and test, yielding `500/50/50`. Four order1 tasks cannot satisfy `500/50/100` from released data. |
| L2 | paper InstrDialog AR around `35.6`, BWT around `-3.8` | yes | same CITB repo | yes | public script exists | not rerun locally | no | setting_blocker | Same split blocker as FT-init; not eligible as main claim until official paper-setting data/script exists. |
| EWC | paper InstrDialog AR around `34.5`, BWT around `-6.8` | yes | same CITB repo | yes | public script exists | not rerun locally | no | setting_blocker | Same split blocker as FT-init; not eligible as main claim until official paper-setting data/script exists. |
| AGEM | paper AGEM(10) AR around `33.2`, AGEM(50) AR around `34.9` | yes | same CITB repo | yes | public scripts exist | not rerun locally | no | setting_blocker | Official memory variants exist in scripts, but all short-stream public runners inherit the `500/50/50` eval cap. |
| Replay | paper Replay(10) AR around `38.4`, Replay(50) AR around `40.4`, BWT up to `1.6`; released Replay order1 `average_train_samples=5855.5` | yes | same CITB repo | yes | public scripts exist | no local smoke for Replay; released scores present | no | setting_blocker | Best non-Multi CITB method from paper, but local paper-comparable run is blocked by split/data mismatch; appendix/blocker only. |
| Multi | paper upper bound AR around `42.1` | yes | same CITB repo | yes | stage1/multi scripts present | no local full Multi repro in current gate | no | setting_blocker | Upper bound, not a continual deployable base; also inherits the public `max_num_instances_per_eval_task=50` short-stream setting. |

Current CITB decision:
- Do not run Ours increments.
- Treat CITB as a CCFA blocker/appendix item, not a main claim.
- Public artifacts currently support only script-strict short-stream `500/50/50`; paper-text `500/50/100` and long-stream `100/50/100` are not official-reproducible from the released code/data.
- Any future run must be labeled either `official_script_500_50_50` or a non-official `paper_text_diagnostic`; never mix these rows in the same main comparison.

## Standard T5-Large PEFT CL

Benchmark: standard CL benchmark with T5-large text-classification tasks and three task orders.

Primary selected base status:
- URL: `https://github.com/cmnfriend/O-LoRA`
- local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora`
- commit: `07117e1fc4a5f5ad9308a815a42cee8f46502dc8`
- environment: `/root/autodl-tmp/conda_envs/lora_v10_o_lora`
- checkpoint: `/root/autodl-tmp/model_cache/hf_snapshots/t5-large`
- official script: `scripts/order_1.sh`
- status doc: `results/logs/official_repo_status_standard_peft_20260705.md`
- official archive paths: `official_repos/standard/O-LoRA`, `official_repos/standard/Lifelong-Fewshot-Language-Learning`, `official_repos/standard/ProgressivePrompts`.

| Method | Paper Result | Official Repo Found | Commit / Path | Downloaded | Runnable | Smoke / Repro Result | Reproduced | Gate | Blocker / Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O-LoRA | O-LoRA paper table: T5-large avg about `75.8`; order1/2/3 reported around `75.4/75.7/76.3` in O-LoRA paper; LB-CL paper reports O-LoRA avg `75.4` | yes | `07117e1...` / `external_sources/o_lora` | yes | yes | v55 smoke completed; v57 formal official-equivalent completed | partial official-equivalent | official_equivalent_anchor | v57 final EM `76.8059`, ROUGE-L `79.9715`; single-GPU grad accumulation/eval batch/runtime-copy engineering deltas documented. |
| LFPT5 | O-LoRA/LB-CL tables: avg roughly `72.7` or `71.3` depending paper table | yes | `cf7d17ce7de6a707d929d0542b3d5e639569855f` / `external_sources/lfpt5` | yes | not rerun locally | no local smoke | no | downloaded_not_runnable | reference baseline, not current Ours base. |
| Progressive Prompts | O-LoRA table reports avg about `75.1`; user baseline notes `76.1` in related setting | yes | `01572d6a73c0576b070ceee00dbe4f5bc278423f` / `external_sources/progressive_prompts` | yes | not rerun locally | no local smoke | no | downloaded_not_runnable | task-ID/prompt protocol differs; must be separately reproduced before claims. |
| LB-CL | NeurIPS/OpenReview: T5-large order1/2/3 `76.9/76.5/76.8`, avg `76.7` | no official repo found | none | no | no | no | no | paper_only_baseline | Rechecked paper, OpenReview, NeurIPS virtual poster/slides, ML Anthology, author GitHub/profile searches, and GitHub title/ID/hash searches. Paper checklist item 5 explicitly says code access is `[No]` / "do not attach the code"; NeurIPS page exposes Paper/Slides/Poster/OpenReview only. `yaoyz96/low-rank-cl` is not accepted: different ICLR 2026 paper/authors and not linked by LB-CL authors. Do not use non-official code as official. |
| SeqLoRA | O-LoRA paper table: avg about `43.7`; LB-CL paper table reports lower avg `39.3` | baseline in O-LoRA paper, no separate official repo needed if reproduced through O-LoRA code | O-LoRA repo path | yes via O-LoRA source | not run locally | no local smoke | no | downloaded_not_runnable | implement only if official O-LoRA repo exposes exact baseline path or paper scripts are located. |
| IncLoRA | O-LoRA paper table: avg about `66.4`; LB-CL paper table about `63.6` | baseline in O-LoRA/LB-CL papers | O-LoRA repo path maybe | yes for O-LoRA source | not run locally | no local smoke | no | downloaded_not_runnable | exact official baseline path needs script verification. |
| Replay | O-LoRA paper table: avg about `57.8` | baseline in paper, exact code path not yet verified | not isolated | no dedicated repo | no | no local smoke | no | blocked | only paper reference until exact official implementation/script is identified. |
| MTL | upper bound about `80.0` | baseline/upper-bound protocol, not a CL method repo | not isolated | no dedicated repo | no | no local smoke | no | paper_only_baseline | upper bound for context, not an Ours base. |

Additional newly observed stronger published candidates:
- MoRA appears in later search with avg `77.6`; not in the user-requested matrix, but it may change the “best published method” if official code is verified.
- OLieRA appears in later search with avg near `79.6`; requires separate official repo/code audit before it can affect the base selection.

Current Standard decision:
- O-LoRA remains the only local official-equivalent anchor.
- v69 Ours overlay is valid as O-LoRA-based increment: EM `77.2566`, ROUGE-L `81.1919`.
- No new Standard Ours candidate should run until stronger published candidates are added to this matrix and official-gated.

## Dialogue NLG / MultiWOZ

Benchmarks: ARPER MultiWOZ-2.0 WOZ3 continual NLG; ToDCL 37-domain task-oriented dialogue CL.

Primary ARPER status:
- URL: `https://github.com/MiFei/Continual-Learning-for-NLG`
- local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/arper`
- commit: `99019defe6bf35e8459ca6abd6f25882724bc956`
- official script: `run.sh`
- official config: `config/config.cfg`
- status doc: `results/logs/official_repo_status_dialogue_nlg_20260705.md`
- official archive path: `official_repos/dialogue/Continual-Learning-for-NLG` pinned to `99019defe6bf35e8459ca6abd6f25882724bc956`.

ToDCL status:
- URL: `https://github.com/andreamad8/ToDCL`
- local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl`
- commit: `e70c1edf937f6eb570296ea2897dbc8d6815bc6d`
- status: downloaded clean; official `train.py --help` passes in faithful legacy Python 3.7 / torch 1.4 env, SGD-only preprocess smoke passes, and SGD/Taskmaster/MultiWOZ source archives are downloaded and zip-validated. Full training/reproduction remains blocked by controlled extraction/layout, MultiWOZ conversion, full data-loader smoke, and method-level smoke.
- network/proxy: temporary mihomo proxy is available; `git ls-remote` reaches `google-research-datasets/dstc8-schema-guided-dialogue`, `google-research-datasets/Taskmaster`, and `budzianowski/multiwoz`. Data archives were downloaded through this route into `/root/autodl-tmp/todcl_official_data_20260705/archives`.
- smoke envs: `/root/autodl-tmp/venvs/todcl_official_smoke_py39` for import/help triage; `/root/autodl-tmp/conda_envs/todcl_legacy_py37` for faithful Python 3.7 / torch 1.4.0 / CUDA 10.1 smoke.
- official archive path: `official_repos/dialogue/ToDCL` pinned to `e70c1edf937f6eb570296ea2897dbc8d6815bc6d`.
- data download audit: GitHub repo API sizes are roughly SGD `51 MB`, Taskmaster `111 MB`, MultiWOZ `126 MB`; `git clone --depth 1` failed with GnuTLS/RPC EOF; retrying codeload with retry plus Python `zipfile.testzip()` produced valid archives for SGD `36,962,546` bytes (`234` entries), Taskmaster `138,699,973` bytes (`161` entries), and MultiWOZ `60,601,152` bytes (`70` entries).
- legacy env audit: Python 3.9 pip cannot resolve `torch==1.4.0`; conda can resolve and create Python 3.7 + pytorch 1.4.0 + CUDA 10.1 when `CONDA_PKGS_DIRS` is moved to `/root/autodl-tmp/conda_pkgs` to avoid `/root` space limits.
- data smoke: ToDCL `get_datasets(dataset_list=['SGD'], develop=True)` passes with train/dev/test totals `532/78/158` over `10` domains. This is not the full ToDCL 37-domain smoke.

| Method | Paper Result | Official Repo Found | Commit / Path | Downloaded | Runnable | Smoke / Repro Result | Reproduced | Gate | Blocker / Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARPER | ARPER paper: best non-Full, e.g. exemplar 250/500 tables; user target BLEU4 `0.701`, SER `3.63` | yes | `99019d...` / ARPER repo | yes | yes | v58 one-epoch smoke completed; v66 formal SCLSTM completed | partial | official_equivalent_anchor | v66 final BLEU4 `0.63231`, SER `4.817`; below user target but official SCLSTM path is runnable. |
| Replay / ER | ARPER paper ER/random/prioritized baselines, e.g. ERprio improves over random but below ARPER | yes, inside ARPER repo | ARPER repo | yes | not rerun separately | no local separate smoke | no | downloaded_not_runnable | baseline variants need script/config isolation. |
| LAMOL | ToDCL README NLG Modularized BLEU `3.49649`, EER `0.35664`; E2E BLEU `3.55622`, EER `0.638889` | yes via ToDCL repo | `e70c1...` / ToDCL | yes; all source archives validated | partial data smoke | `train.py --help` passes in legacy env; SGD-only preprocess smoke passes | no | downloaded_not_runnable | needs Taskmaster/MultiWOZ extraction/layout, MultiWOZ conversion, full data-loader smoke, and method-level smoke before reproduction. |
| AdapterCL / ADAPTER | ToDCL README NLG Modularized BLEU `21.7719`, EER `0.163975`; E2E BLEU `16.5768`, EER `0.331949` | yes via ToDCL repo | `e70c1...` / ToDCL | yes; all source archives validated | partial data smoke | same legacy help + SGD preprocess smoke | no | downloaded_not_runnable | likely best non-Multi ToDCL NLG baseline, but full data/method smoke is still blocked. |
| Replay / REPLAY | ToDCL README NLG Modularized BLEU `21.4832`, EER `0.0559855`; E2E BLEU `16.2668`, EER `0.190309` | yes via ToDCL repo | `e70c1...` / ToDCL | yes; all source archives validated | partial data smoke | same legacy help + SGD preprocess smoke | no | downloaded_not_runnable | strong EER; needs full ToDCL data-loader and method smoke. |
| Multi upper bound | ToDCL README NLG Modularized BLEU `26.1462`, EER `0.0341823`; E2E BLEU `23.6073`, EER `0.12558` | yes via ToDCL repo | `e70c1...` / ToDCL | yes; all source archives validated | partial data smoke | same legacy help + SGD preprocess smoke | no | downloaded_not_runnable | upper bound, not deployed CL base; full data-loader smoke still required. |
| VANILLA/L2/EWC/AGEM | ToDCL README includes low NLG BLEU/EER baselines | yes via ToDCL repo | `e70c1...` / ToDCL | yes; all source archives validated | partial data smoke | same legacy help + SGD preprocess smoke | no | downloaded_not_runnable | not best bases; still matrix candidates for comparison. |

Current Dialogue decision:
- ARPER is the only runnable official anchor.
- ToDCL legacy help smoke and SGD-only preprocess smoke pass; SGD/Taskmaster/MultiWOZ source archives are validated. ToDCL remains blocked by extraction/layout, MultiWOZ conversion, full data-loader smoke, and no full method-level smoke/reproduction.
- No Dialogue Ours work on ToDCL until the official smoke passes.

## Immediate Blockers

1. LB-CL official/author code source not found; paper checklist says code is not attached, and no author-linked repo/supplement was confirmed.
2. CITB official setting blocker: paper InstrDialog `500/50/100` and InstrDialog++ `100/50/100` do not match public scripts/data. Short stream is script-strict `500/50/50`; long stream is script-strict `100/25/25`; released data has short tasks that cannot satisfy the paper counts. Network is not the current blocker.
3. ToDCL is partially unblocked: faithful legacy env exists, SGD preprocess smoke passes, and all three source archives are zip-validated. Full gate remains blocked by Taskmaster/MultiWOZ extraction/layout, MultiWOZ conversion, full data-loader smoke, and method-level smoke/reproduction.
4. Potential newer Standard methods (MoRA/OLieRA) may supersede LB-CL/O-LoRA but must first pass this same official repo gate.
