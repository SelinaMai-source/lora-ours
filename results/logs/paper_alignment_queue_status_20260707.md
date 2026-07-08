# Paper alignment queue — 20260707

Updated: 2026-07-08T07:12:49+08:00
Phase: post-v88 gate

Priority:
1. ARPER v88 (config.cfg exact: exemplar 250, batch 128)
2. Post-v88 gate → v89 if FAIL → ToDCL ADAPTER
3. CITB Stage-1 seed50 + Replay v2 (optional strict parity; AR already ±1 on v56)

Standard O-LoRA v57: PASS — no relaunch.
CITB v56 AR matrix: PASS ±1 — Stage-1 optional.

Orchestrator: scripts/run_strict_paper_repro_iteration.sh
