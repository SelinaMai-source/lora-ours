# Scripts

`ours_v1/scripts/` contains only v1 entry points.

- `queue/run_v1_iterate.sh`: serial smoke queue for the v1 overlays.
- `launchers/run_citb_v1.sh`: CITB Replay(50) + SSRG-aligned replay launcher.
- `launchers/run_standard_v1.sh`: Standard O-LoRA + class-coverage SSRG gate launcher.
- `launchers/run_arper_v1.sh`: ARPER SCLSTM + SSRG exemplar launcher.
- `launchers/run_todcl_v1.sh`: ToDCL ADAPTER + assess orthogonal launcher.
- `monitors/README.md`: monitor references; monitor implementations stay in their legacy paths.

Legacy `scripts/*v1_20260708*.sh` paths remain as wrappers.
