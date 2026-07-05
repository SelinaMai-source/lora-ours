# Official Gate Proxy Status - 2026-07-05

Scope: official reproducibility gate only. No Ours candidate was launched.

## Safe Inspection

- Config directory checked: `/root/autodl-tmp/Lora-code/configs/clash`
- Sensitive config contents were not printed or committed.
- Initial state: config files existed, but ports `7890/7891/7892` were not listening.

## Runtime Discovery

- Found binary: `/usr/local/bin/mihomo`
- Version: Mihomo Meta `v1.19.21`
- Direct config start with the top-level yaml stayed in initialization and did not open ports.
- Runtime no-geoip config start succeeded:
  - config: `/root/autodl-tmp/Lora-code/configs/clash/runtime/d18255a-GS.no_geoip.yaml`
  - tmux session: `official-gate-mihomo-proxy`
  - ports listening: `7890`, `7891`, `7892`

Temporary environment for official downloads/searches:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7891
```

## Connectivity Verification

- GitHub: reachable through proxy (`200`).
- HuggingFace: reachable through proxy (`200`).
- OpenReview: reachable but redirects to browser verification/challenge; not reliable for non-browser scraping.
- GitHub unauthenticated API repository search: rate-limited (`403`), so missing API results are not evidence that LB-CL code does not exist.
- ToDCL upstream data repos are reachable with `git ls-remote`:
  - `google-research-datasets/dstc8-schema-guided-dialogue`
  - `google-research-datasets/Taskmaster`
  - `budzianowski/multiwoz`

## Gate Impact

- LB-CL remains `paper_only_baseline`: proxy improves access, but no official/author repo has been confirmed; GitHub API search is rate-limited.
- CITB remains blocked by setting ambiguity (`500/50/100` paper vs script-strict `500/50/50`), not by network.
- ToDCL network blocker is reduced: upstream repos are reachable. Remaining blockers are official data layout/download policy and legacy environment setup (`pytorch_lightning` smoke failure).

## Current Runtime State

- Proxy tmux session is running for follow-up official-gate checks.
- Existing sentinel session remains running.
- No GPU training job was started.
