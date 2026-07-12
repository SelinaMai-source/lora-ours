# Ours Autonomous Campaign Status

- Updated: 2026-07-12T15:00+08:00
- Active branch: **ours-v2**
- Also pushed: ours-v1 (controller + SSRG hang fix backport)
- Local premature **ours-v3** exists from infra-kill misfire; **not** the active Standard delta until v2 completes a real metric gate
- Standard delta (v2): `ASSESS_RETENTION_SKIP_TASKS=amazon`
- Infra fix: SSRG pool/vocab cap (not a mechanism delta)
- InstrDialog++: **external_blocker** (public split shortfall)
- Controller tmux: `lora-ours-autonomous-loop`
- Loop log: `results/logs/ours_autonomous_loop_20260712.log`

## Queue order (single GPU)
1. Standard v2 smoke→formal→iterate
2. CITB InstrDialog v1 smoke→formal
3. ARPER v1 smoke→formal
4. ToDCL v1 smoke→formal
