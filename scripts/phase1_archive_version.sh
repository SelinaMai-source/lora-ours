#!/usr/bin/env bash
# Create an auditable git commit for phase-1 scaffold changes after review.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPECTED_BRANCH="${EXPECTED_BRANCH:-ours-v0-setup}"
BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "[blocked] expected branch $EXPECTED_BRANCH, got $BRANCH" >&2
  exit 2
fi

echo "[status]"
git status --short --branch
echo
echo "[diffstat]"
git diff --stat
echo
echo "This script does not auto-commit. Review the status/diff, then commit manually when ready."
