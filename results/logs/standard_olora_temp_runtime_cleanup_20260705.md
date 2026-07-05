# Standard O-LoRA Temporary Runtime Cleanup

- Updated: 2026-07-05
- Reason: root filesystem reached `100%` usage, causing O-LoRA/pyarrow `Bus error` and preventing v71 smoke from starting.
- Preserved: committed logs, status files, v69 formal artifacts, v70b formal artifacts, final metrics, and launch/checkpoint records.
- Removed: generated dry-run and smoke runtime copies only, including v70/v70b/v71 temporary O-LoRA runtime directories.
- Space after cleanup: `/root` returned to about `90%` usage with about `3.3G` available.
- Health check after cleanup: O-LoRA official `run_uie_lora.py --help` succeeded again through the conda environment.
