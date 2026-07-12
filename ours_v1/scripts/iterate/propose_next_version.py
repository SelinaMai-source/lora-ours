#!/usr/bin/env python3
"""Propose exactly one auditable mechanism delta and materialize ours-v{N+1} overlay files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
STATE_PATH = Path(__file__).resolve().parent / "suite_state.json"
PROPOSALS = REPO / "docs" / "experiments" / "proposals"


ALLOWED_DELTAS = {
    "amazon_round_assess_pause": {
        "suites": ["standard"],
        "env": {"ASSESS_RETENTION_SKIP_TASKS": "amazon"},
        "description": "Pause Assess retention gate on amazon round only; keep class-coverage SSRG and O-LoRA base.",
    },
    "assess_retention_threshold": {
        "suites": ["standard", "todcl"],
        "env": {"ASSESS_RETENTION_THRESHOLD": "0.20"},
        "todcl_env": {"OURS_ASSESS_THRESHOLD": "0.20"},
        "description": "Relax assess threshold by one step (0.25→0.20) without adding mechanisms.",
    },
    "replay_budget": {
        "suites": ["standard", "citb_instrdialog"],
        "env": {"REPLAY_PER_TASK": "96"},
        "citb_note": "increase min_replay_per_segment / replay budget in YAML only",
        "description": "Increase replay budget one notch; keep SSRG mode.",
    },
    "ssrg_class_coverage": {
        "suites": ["standard", "arper"],
        "env": {"SC_CLASS_COVERAGE_ORDER": "1", "SSRG_ENERGY_THRESHOLD": "0.80"},
        "description": "Tune SSRG energy threshold slightly; keep selector family.",
    },
    "drift_detection": {
        "suites": ["standard"],
        "env": {"TRAIN_HELDOUT_GATE": "1"},
        "description": "Enable train-heldout drift gate only (no other overlay change).",
    },
    "anti_overlap": {
        "suites": ["todcl"],
        "todcl_env": {"OURS_ORTHOGONAL_PENALTY": "0.15"},
        "description": "Increase orthogonal penalty one step on ToDCL Assess overlay.",
    },
}


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def next_version_name(current: str) -> str:
    m = re.search(r"v(\d+)$", current)
    if not m:
        return "ours-v2"
    return f"ours-v{int(m.group(1)) + 1}"


def choose_delta(suite: str, diagnosis: dict[str, Any] | None) -> str:
    if diagnosis and diagnosis.get("recommended_next_delta") in ALLOWED_DELTAS:
        delta = diagnosis["recommended_next_delta"]
        if suite in ALLOWED_DELTAS[delta]["suites"] or not ALLOWED_DELTAS[delta]["suites"]:
            return delta
    # defaults per suite
    defaults = {
        "standard": "amazon_round_assess_pause",
        "citb_instrdialog": "replay_budget",
        "arper": "ssrg_class_coverage",
        "todcl": "assess_retention_threshold",
    }
    return defaults.get(suite, "replay_budget")


def write_standard_launcher(version: str, delta_key: str, env: dict[str, str]) -> Path:
    launchers = REPO / "ours_v1" / "scripts" / "launchers"
    path = launchers / f"run_standard_{version.replace('ours-', '')}.sh"
    # Prefer editing run_standard_v1 via wrapper that exports delta env.
    ver = version.replace("ours-", "")
    run_smoke = f"olora_official_base_ours_overlay_{ver}_20260712_smoke_order1_seed1"
    run_formal = f"olora_official_base_ours_overlay_{ver}_20260712_formal_order1_seed1"
    env_exports = "\n".join(f"export {k}={v}" for k, v in env.items())
    content = f'''#!/usr/bin/env bash
# Standard {version}: O-LoRA + class-coverage SSRG + single delta [{delta_key}].
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../.." && pwd)"
FORMAL="${{FORMAL:-0}}"

if [[ "${{FORMAL}}" == "1" ]]; then
  RUN_NAME="${{RUN_NAME:-{run_formal}}}"
  WANDB_GROUP="${{WANDB_GROUP:-published-base-standard-olora-plus-ours-overlay-{ver}-formal}}"
  RUN_LABEL="${{RUN_LABEL:-published_base_olora_plus_{delta_key}_{ver}_formal}}"
else
  RUN_NAME="${{RUN_NAME:-{run_smoke}}}"
  WANDB_GROUP="${{WANDB_GROUP:-published-base-standard-olora-plus-ours-overlay-{ver}-smoke}}"
  RUN_LABEL="${{RUN_LABEL:-published_base_olora_plus_{delta_key}_{ver}_smoke}}"
fi

export RUN_NAME WANDB_GROUP RUN_LABEL
export WANDB_PROJECT="${{WANDB_PROJECT:-lora-ours-v1}}"
export REPLAY_MODE="${{REPLAY_MODE:-ssrg}}"
export REPLAY_PER_TASK="${{REPLAY_PER_TASK:-64}}"
export ASSESS_RETENTION_GATE="${{ASSESS_RETENTION_GATE:-1}}"
export ASSESS_RETENTION_THRESHOLD="${{ASSESS_RETENTION_THRESHOLD:-0.25}}"
export SC_CLASS_COVERAGE_ORDER="${{SC_CLASS_COVERAGE_ORDER:-1}}"
export SSRG_SPECTRAL_TOP_K="${{SSRG_SPECTRAL_TOP_K:-8}}"
export SSRG_ENERGY_THRESHOLD="${{SSRG_ENERGY_THRESHOLD:-0.85}}"
export EARLY_GATE_DBPEDIA_EM="${{EARLY_GATE_DBPEDIA_EM:-90}}"
export EARLY_GATE_AMAZON_EM="${{EARLY_GATE_AMAZON_EM:-50}}"
export TRAIN_HELDOUT_GATE="${{TRAIN_HELDOUT_GATE:-0}}"
# Single-mechanism delta for {version}:
{env_exports}
export FORMAL

exec "${{REPO_ROOT}}/scripts/run_olora_standard_order1_official_base_ours_overlay_v58.sh" "$@"
'''
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    # Also update run_standard_v1.sh shim? Keep v1 intact; create symlink alias for loop.
    return path


def write_proposal(suite: str, version: str, delta_key: str, meta: dict[str, Any], diagnosis: dict | None) -> Path:
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    path = PROPOSALS / f"{suite}_{version}_proposal.md"
    body = f"""# Proposal {version} — {suite}

Updated: {datetime.now(timezone.utc).isoformat()}

## Single mechanism delta

- **key:** `{delta_key}`
- **description:** {meta['description']}
- **env / overlay:** `{json.dumps(meta.get('env') or meta.get('todcl_env') or {}, ensure_ascii=False)}`

## Constraints

- Published-base + minimal overlay only
- Official scorer / split / task order / model unchanged
- One mechanism only (no stacking)

## Prior diagnosis

```json
{json.dumps(diagnosis or {}, indent=2, ensure_ascii=False)}
```
"""
    path.write_text(body, encoding="utf-8")
    return path


def maybe_create_git_branch(version: str, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"branch": version, "created": False, "dry_run": True}
    existing = subprocess.run(
        ["git", "rev-parse", "--verify", version],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        # Do not force checkout when the worktree is dirty; files are versioned by path.
        return {"branch": version, "created": False, "checked_out": False}
    # Create branch pointer at HEAD without switching away from the running worktree.
    created = subprocess.run(
        ["git", "branch", version],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        return {
            "branch": version,
            "created": False,
            "error": created.stderr.strip(),
        }
    return {"branch": version, "created": True, "checked_out": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--from-version", default="")
    ap.add_argument("--delta", default="", help="force delta key")
    ap.add_argument("--create-branch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    state = load_state()
    suite_st = state["suites"][args.suite]
    current = args.from_version or suite_st.get("current_version") or "ours-v1"
    version = next_version_name(current)
    diagnosis = None
    if suite_st.get("last_failure"):
        diagnosis = suite_st["last_failure"].get("diagnosis")

    delta_key = args.delta or choose_delta(args.suite, diagnosis)
    if delta_key not in ALLOWED_DELTAS:
        raise SystemExit(f"delta {delta_key} not in RP allow-list")
    meta = ALLOWED_DELTAS[delta_key]
    if meta["suites"] and args.suite not in meta["suites"]:
        raise SystemExit(f"delta {delta_key} not allowed for suite {args.suite}")

    proposal = write_proposal(args.suite, version, delta_key, meta, diagnosis)
    artifacts: dict[str, Any] = {"proposal": str(proposal), "delta": delta_key, "version": version}

    if args.suite == "standard":
        launcher = write_standard_launcher(version, delta_key, meta.get("env") or {})
        artifacts["launcher"] = str(launcher)

    if args.create_branch:
        artifacts["git"] = maybe_create_git_branch(version, args.dry_run)

    suite_st["current_version"] = version
    suite_st["next_version"] = next_version_name(version)
    suite_st["status"] = "proposed"
    suite_st["planned_delta"] = {"key": delta_key, "meta": meta, "proposal": str(proposal)}
    if delta_key == "amazon_round_assess_pause":
        suite_st["overlay_active"] = [
            "class-coverage SSRG",
            "assess-retention gate 0.25 (paused on amazon round)",
        ]
    save_state(state)

    # Write machine-readable proposal sidecar
    side = PROPOSALS / f"{args.suite}_{version}_proposal.json"
    side.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    artifacts["json"] = str(side)
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
