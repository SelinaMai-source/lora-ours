# Phase 0 preflight status (2026-07-06)

## Git branch
- **Branch:** `sota-24h-campaign-20260706` (from `official-repro-gate-followup-20260705` HEAD)
- **Remote:** pushed to `origin` (`git@github.com:SelinaMai-source/lora-ours.git`), tracking set

## Proxy (mihomo)
- **Status:** running (PID visible via `pgrep mihomo`)
- **Config:** `/root/autodl-tmp/Lora-code/configs/clash/runtime/d18255a-GS.no_geoip.yaml`
- **GitHub check:** HTTP 200 via `http://127.0.0.1:7890`

## Official submodules (`official_repos/`, superproject pins)
| Repo | Path | Commit (short) | Worktree |
|------|------|----------------|----------|
| CITB | `official_repos/citb/CITB` | `bf50533` | partial on `/` (order/tasks); full tasks via autodl-tmp mirror for split preflight |
| O-LoRA | `official_repos/standard/O-LoRA` | `07117e1` | checkout blocked: root FS **100% full** during reset |
| ARPER | `official_repos/dialogue/Continual-Learning-for-NLG` | `99019de` | same disk blocker |
| ToDCL | `official_repos/dialogue/ToDCL` | `e70c1ed` | same disk blocker |
| LFPT5 | `official_repos/standard/Lifelong-Fewshot-Language-Learning` | `cf7d17c` | same disk blocker |
| Progressive Prompts | `official_repos/standard/ProgressivePrompts` | `01572d6` | same disk blocker |

**external_sources:** not a top-level dir in `lora-ours`; runtime assets live under `/root/autodl-tmp/lora-baselines-run_v1/external_sources/{citb,arper,todcl}` (referenced by preflight JSON).

**Blocker (mitigated 2026-07-06 00:39 UTC+8):** `/` overlay was **100% full** during submodule reset. Cleanup freed **~2.1G** (`__pycache__`, pip/conda cache purge, stale `/tmp`). Current: **94%** (~2.1G avail on `/`). Large runtime assets remain on `/root/autodl-tmp` (70% used, **46G** free).

**Submodule runtime strategy:** Do not expand `official_repos/*` worktrees on `/`. Use autodl-tmp mirrors:
- CITB: `/root/autodl-tmp/Lora-code/external_baselines/citb_official` (launcher default `CITB_ROOT`)
- O-LoRA: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/o_lora`
- ARPER: `baselines/advanced_baselines/arper_dialog_nlg/external` (in-repo) + data under autodl-tmp
- ToDCL: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl`

## Preflight scripts
| Script | Outcome | Notes |
|--------|---------|-------|
| `preflight_published_method_bases.py` | **PASS** (exit 0) | `gpu_safe=true`, `starts_training=false`, 3 suites; out: `results/logs/published_method_bases_preflight_20260706.json` |
| `preflight_citb_official_split_counts.py` | **PARTIAL** (exit 3) | policy `500/50/50`, 19 tasks, **4 short tasks** (expected for official short-stream); CITB root: `autodl-tmp/.../citb_official` (commit `c964185`, not submodule pin); out: `results/logs/citb_official_split_counts_preflight_20260706.json` |

## tmux (prefix `lora-ours-`)
- `lora-ours-monitor` — placeholder (created 2026-07-06)
- `lora-ours-sentinel` — placeholder (created 2026-07-06)
- `lora-ours-sentinel-v73` — pre-existing monitor session

## Next
- GPU queue: `scripts/run_ours_overlay_iterate_queue.sh` (serial, prefix `lora-ours-`).
- Agent stack: `bash scripts/launch_lora_ours_agent_stack.sh`.
- Phase 1 baselines: CITB Replay50 smoke → Standard v85 → ARPER v86 → ToDCL ADAPTER anchor.
