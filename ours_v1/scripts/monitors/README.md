# Monitors

Monitor implementations remain in `scripts/` because existing tmux sessions,
status JSON files, and logs reference those paths.

- ARPER monitor: `scripts/monitor_arper_woz3_formal.py`
- Queue state: `results/logs/ours_v1_20260708_iterate_queue.json`
- Status markdown: `results/logs/*_status.md`

This directory intentionally avoids copying live logs or generated status files.
