# Paper alignment queue — 20260707

Updated: 2026-07-09T10:30:00+08:00
Phase: ARPER v88 running (v89 done FAIL; v90 queued after v88 gate)

Priority:
1. ARPER v88 (config.cfg exact: exemplar 250, batch 128) — **ACTIVE on GPU**
2. Post-v88 gate → v90 if v88+v89 still FAIL
3. CITB Stage-1 seed50 + Replay v2 (optional; AR already ±1 on v56)

Standard O-LoRA v57: borderline FAIL (+1.01) — no relaunch unless 8-GPU available.
CITB v56 AR matrix: PASS ±1 — Stage-1 optional.
ToDCL ADAPTER: PASS ±1 — complete.

Orchestrator: scripts/run_strict_paper_repro_iteration.sh
