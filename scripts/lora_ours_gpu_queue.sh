#!/usr/bin/env bash
# SOTA 24h campaign GPU queue — delegates to baseline reproduction queue (paper alignment first).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${REPO_ROOT}/scripts/run_baseline_reproduction_queue.sh" "$@"
