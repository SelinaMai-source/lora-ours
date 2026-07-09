#!/usr/bin/env bash
# Compatibility wrapper. Canonical entry: ours_v1/scripts/launchers/run_todcl_v1.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${REPO_ROOT}/ours_v1/scripts/launchers/run_todcl_v1.sh" "$@"
