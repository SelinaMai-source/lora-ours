# External Cursor Skills Review

Updated: 2026-07-03

## Scope

This note records a lightweight review of optional external skills requested for the `lora-ours` workflow:

- `CCFA-Skill` / `CCFA Skill`
- `全流程自动 skill`
- `auto-research skill` / `auto-research Cursor skill`

The review intentionally did not run unknown install scripts, MCP servers, hooks, npm packages, or benchmark loops. Candidate repositories were shallow-cloned only into `/root/.cursor/skills-candidates/` for static inspection. Existing global skills under `/root/.cursor/skills-cursor` were not modified.

## Candidate Summary

| Request name | Best match | Source | Commit reviewed | Downloaded | Confidence |
| --- | --- | --- | --- | --- | --- |
| `CCFA-Skill`, `CCFA Skill` | `mikubaka88/CCFA-Skills` | `https://github.com/mikubaka88/CCFA-Skills.git` | `54578cfd9c04545cb6fe39f8dcf26966fe166c41` | `/root/.cursor/skills-candidates/CCFA-Skills` | High |
| `全流程自动 skill` | `echoVic/boss-skill` | `https://github.com/echoVic/boss-skill.git` | `d6e90658c4e996e148f6239c95f9b50ed5fbc8da` | `/root/.cursor/skills-candidates/boss-skill` | Medium |
| `全流程自动 skill` | `TashanGKD/tashan-cursor-skills` | `https://github.com/TashanGKD/tashan-cursor-skills` | Not cloned | Not downloaded | Low to medium |
| `auto-research skill` | `davebcn87/pi-autoresearch` | `https://github.com/davebcn87/pi-autoresearch.git` | `6e1132c8ad83fbd6e3d6c123e0c62a13e7664dcc` | `/root/.cursor/skills-candidates/pi-autoresearch` | High upstream, but not Cursor-native |
| `auto-research Cursor skill` | `ergenekonyigit/cursor-autoresearch` | `https://github.com/ergenekonyigit/cursor-autoresearch.git` | `f20120abb97309b26a5be5ed6f3d638a63dc4d07` | `/root/.cursor/skills-candidates/cursor-autoresearch` | Medium |
| `auto-research Cursor skill` | `lukeundtrug/cursor-autoresearch` | `https://github.com/lukeundtrug/cursor-autoresearch` | Not cloned | Not downloaded | Low |

Notes:

- `mikubaka88/CCFA-Skills` is the clearest match for the CCFA request. It contains 15 top-level `ccf-*` folders, each with `SKILL.md`, plus shared governance in `ccf-common`.
- `echoVic/boss-skill` is the clearest match for "全流程自动" as a skill-style software delivery pipeline. It is not research-specific and includes a CLI/runtime/hook surface.
- `TashanGKD/tashan-cursor-skills` appears to be a large Cursor skill/rule/subagent collection with self-evolution behavior. Because the request was for a focused optional skill review and this candidate is broad and auto-updating by design, it was recorded but not cloned.
- `davebcn87/pi-autoresearch` appears to be the high-star upstream for autoresearch. It targets `pi` and contains skills, hooks, MCP-like experiment tools, and automatic git behavior.
- `ergenekonyigit/cursor-autoresearch` is a Cursor/VS Code port with workspace skills, rules, MCP server, and optional dashboard. It is lower-signal than the upstream by popularity but is more directly Cursor-oriented.
- `lukeundtrug/cursor-autoresearch` appeared as a zero-star fork in search results. It was not treated as a primary source.

## Static Safety Review

### CCFA Skills

Observed structure:

- Skill-only family with top-level `ccf-*` directories and `SKILL.md` files.
- Shared policy in `ccf-common`.
- Documentation emphasizes ownership boundaries, artifact contracts, no invented citations/results, and privacy/evidence handling.
- Install docs suggest copying `ccf-*` folders into a skills directory, with `ccf-common` required for any partial installation.

Risk assessment:

- Low execution risk: review did not find MCP config, global hook installation, or automatic command execution as part of ordinary runtime skills.
- Moderate workflow risk: if enabled globally, many broad triggers such as literature search, paper writing, review, and submission checking may affect ordinary agent routing.
- Evidence/privacy risk is explicitly addressed by the skills, but private draft text should still not be pasted into web queries without explicit approval.

Recommendation:

- Best candidate for controlled installation, but only after user confirms the exact URL.
- Prefer project-local installation or selective use as read-only references first.
- For `lora-ours`, useful subset would be `ccf-common`, `ccf-literature-searcher`, `ccf-literature-monitor`, `ccf-experiment-designer`, `ccf-paper-reviewer`, `ccf-integrity-auditor`, and `ccf-submission-checker`.

### Boss Skill

Observed structure:

- Main skill entry at `skill/SKILL.md`.
- Additional role prompts, command definitions, templates, references, CLI source, tests, and hook configs.
- README describes npm package `@blade-ai/boss-skill`, `boss-skill install`, event-sourced `.boss/<feature>/` state, quality gates, and hook profiles.

Risk assessment:

- Medium to high execution risk if installed normally: the installer can write to agent configuration directories, merge hook configs, and install a CLI.
- It may create `.boss/<feature>/` artifacts in the project and run tests/build/deploy-style commands depending on workflow.
- It is oriented toward product/software delivery, not ML experiment governance.
- Static tests include safeguards against commands such as `rm -rf /`, force push, and hard reset, but this does not remove the need to review hook behavior before enabling it.

Recommendation:

- Do not install globally for the current research repo.
- If used, keep it as a documentation/template reference for planning and QA gates only.
- Not recommended for active CITB/GPU experiment automation, because it may launch broad build/test workflows unrelated to the current controlled reproduction gate.

### Auto-research Upstream and Cursor Port

Observed structure:

- `pi-autoresearch` has `skills/autoresearch-create`, `skills/autoresearch-finalize`, and `skills/autoresearch-hooks`.
- `cursor-autoresearch` has Cursor/VS Code oriented skills, `.cursor/rules`, MCP server packages, scripts, and a dashboard/extension path.
- Both define a loop of `init_experiment`, `run_experiment`, and `log_experiment`.
- `log_experiment` can auto-commit kept results and auto-revert discarded/crashed/failed-check results.
- Cursor port install scripts can write `.cursor/rules`, `.cursor/skills`, symlink skills, and merge `~/.cursor/mcp.json`.

Risk assessment:

- High workflow risk for this repo if used naively: it is designed to repeatedly edit code, run commands, keep/discard changes, and continue autonomously.
- Git risk is explicit: `git checkout -b`, auto-commit on keep, and revert/clean on discard paths.
- Resource risk is high if the benchmark command is a GPU training or evaluation launch.
- Config risk is present in the Cursor port: bootstrap/install can merge global Cursor MCP config and symlink global agent skills.
- The tool is appropriate only when the measurement command is cheap, deterministic, bounded, and isolated.

Recommendation:

- Do not install globally and do not connect it to live CITB official-base or W&B/GPU jobs.
- If needed later, use only a workspace-local, pinned checkout and a project branch such as `autoresearch/<goal>-<date>`.
- Limit its benchmark script to CPU-only static checks, small parser tests, documentation validators, or tiny smoke tests with explicit timeouts.
- Disable or avoid any "never stop" behavior for the `lora-ours` workflow unless a human explicitly starts a bounded session.

## Fit For Current `lora-ours` Workflow

Recommended uses:

- Literature search: `CCFA-Skills` `ccf-literature-searcher` and `ccf-literature-monitor` can help collect official method references, related continual learning baselines, and venue-aware positioning. Use public queries only.
- Official settings verification: `ccf-integrity-auditor` and `ccf-submission-checker` can be adapted as checklists for claim/evidence consistency, official split settings, page/metadata-like reproducibility gates, and citation integrity.
- Experiment design: `ccf-experiment-designer` is useful for baseline matrices, ablation plans, and reporting templates, but must never invent results.
- CCFA-style review: `ccf-paper-reviewer` and `ccf-integrity-auditor` are useful for CCF-A paper risk review, claim support, numeric consistency, and missing baseline diagnosis.
- Workflow planning: selected `boss-skill` ideas can inspire `.boss`-like gate artifacts, but should remain a reference rather than an enabled automation layer.
- Automatic iteration: `auto-research` should not be used on active GPU training. It may be useful later for CPU-only micro-optimization of log parsers, report generators, config validation, or bounded launcher dry-runs.

Not recommended now:

- Installing any candidate into `/root/.cursor/skills-cursor` or other global skill paths without user confirmation.
- Running `boss-skill install`, `cursor-autoresearch` bootstrap/install scripts, MCP servers, hooks, or npm install/build steps.
- Letting `auto-research` manage the current CITB official-base reproduction branch or any branch with active GPU/W&B experiment state.
- Using broad self-evolving skill collections in this repo before a separate policy review.

## Proposed Next Step

Ask the user to confirm exact URLs and desired installation scope:

1. If the user wants CCFA support, install a pinned, project-local subset from `mikubaka88/CCFA-Skills` only after confirming the URL and selected skills.
2. If the user wants "全流程自动", keep `boss-skill` as a planning reference first; do not enable hooks or CLI install in `lora-ours`.
3. If the user wants `auto-research`, require a bounded CPU-only target, a throwaway branch, a short `maxIterations`, explicit off-limits files, and a benchmark script that cannot launch GPU jobs.

Until confirmed, the downloaded repositories should remain candidates only.
