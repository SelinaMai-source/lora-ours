# Official Alignment

This directory stores audit artifacts for aligning `ours` with official continual
learning settings before any benchmark-scale run.

Start here:

```bash
python scripts/phase1_preflight.py --remote-timeout-sec 20
```

Expected v0 behavior:

- Preflight may return non-zero while official repos, data, or checkpoints are missing.
- A non-zero preflight is a blocker for real benchmark training, not a failure to hide.
- Toy smoke runs are allowed only for pipeline health checks.

Local toy smoke, explicitly not a benchmark:

```bash
python scripts/phase1_prepare_data.py --write-toy-smoke
python core/train.py --config configs/phase1/ours_v0_debug_smoke.yaml
python scripts/phase1_monitor_segments.py --run-name phase1_ours_v0_debug_smoke
```

tmux launcher:

```bash
# Strict default: run preflight first and block if official inputs are missing.
bash scripts/phase1_tmux_launch.sh

# Local toy smoke only; do not report as benchmark.
RUN_PREFLIGHT=0 SESSION=lora-ours-v0-smoke bash scripts/phase1_tmux_launch.sh
```

Do not claim SOTA or compare against baselines from this directory unless the
corresponding official alignment audit records a completed strict run with real
data, real checkpoints, logs, and metrics.
