#!/usr/bin/env python3
"""Generate evidence-based FAILURE_ANALYSIS.md before the next SOTA version.

Usage:
  python3 scripts/sota_failure_analysis.py --failed v8_sota_1 --next v8_sota_2
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]

SOTA_TARGETS = {
    "seen_avg_score": (">=", 0.6),
    "seen_avg_task_aware_score": (">=", 0.6),
    "forgetting": ("<=", 0.0333),
    "task_aware_forgetting": ("<=", 0.077),
    "token_f1_mean": (">=", 0.6),
    "lcs_overlap_mean": (">=", 0.6),
}

V62_TRAJECTORY = {
    0: 0.0, 1: 0.05, 2: 0.0, 3: 0.25, 4: 0.34, 5: 0.325, 6: 0.426,
    7: 0.373, 8: 0.378, 9: 0.322, 10: 0.274, 11: 0.311, 12: 0.295,
    13: 0.275, 14: 0.297, 15: 0.328, 16: 0.35, 17: 0.381, 18: 0.35,
}

NEXT_DEFAULT = {
    "v8_sota_1": "v8_sota_2",
    "v8_sota_2": "v8_sota_3",
    "v7_sota_1": "v8_sota_1",
    "v7_sota_2": "v8_sota_2",
}


def _run_dir(version: str) -> Path:
    return REPO / "results/runs" / f"paper_instrdialog_ours_full_s123_{version}"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _read_final(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "final_metrics.json"
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("final") or data


def _early_stop_info(version: str) -> Tuple[bool, str]:
    log = REPO / f"{version}.log"
    if not log.is_file():
        return False, ""
    text = log.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if "Early stopping triggered" in line or "Trajectory early stopping" in line:
            return True, line.strip()
    return False, ""


def _metric(row: Dict[str, Any], key: str) -> float:
    return float(row.get(f"eval.{key}", row.get(key, 0)) or 0)


def _trajectory_table(rows: List[Dict[str, Any]]) -> str:
    lines = ["| seg | seen (fail) | seen (v6_2) | Δ | oracle | overlap_cos |", "|-----|-------------|-------------|---|--------|-------------|"]
    for r in rows:
        s = int(r.get("segment_id", -1))
        seen = float(r.get("eval.seen_avg_score", 0) or 0)
        ref = V62_TRAJECTORY.get(s)
        delta = f"{seen - ref:+.3f}" if ref is not None else "—"
        oracle = r.get("routing.oracle_agreement_rate", "—")
        ov = r.get("train.overlap_mean_cosine", r.get("overlap_mean_cosine", "—"))
        ref_s = f"{ref:.3f}" if ref is not None else "—"
        lines.append(f"| {s} | {seen:.3f} | {ref_s} | {delta} | {oracle} | {ov} |")
    return "\n".join(lines)


def _sota_table(row: Dict[str, Any]) -> str:
    lines = ["| 指标 | 值 | 目标 | 达标 |", "|------|-----|------|------|"]
    for key, (op, target) in SOTA_TARGETS.items():
        val = _metric(row, key)
        if op == ">=":
            met = val >= target
            tgt = f"≥ {target}"
        else:
            met = val <= target
            tgt = f"≤ {target}"
        lines.append(f"| {key} | {val:.4f} | {tgt} | {'✅' if met else '❌'} |")
    return "\n".join(lines)


def _guess_status(version: str, rows: List[Dict[str, Any]], final: Optional[Dict[str, Any]]) -> str:
    early, _ = _early_stop_info(version)
    if final and int(final.get("segment_id", -1)) >= 18:
        return "completed"
    if early:
        return "early_stopped"
    if rows and int(rows[-1].get("segment_id", 0)) >= 18:
        return "completed"
    return "crashed_or_incomplete"


def _read_changes_snippet(version: str) -> str:
    p = REPO / "archive" / version / "CHANGES.md"
    if p.is_file():
        return p.read_text(encoding="utf-8")[:2000]
    return "（无 archive CHANGES.md）"


def _suggested_hypothesis(failed: str, row: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    """Data-driven hint for agent; not auto-implemented."""
    seen = _metric(row, "seen_avg_score")
    oracle = float(row.get("routing.oracle_agreement_rate", 0) or 0)
    ov = float(row.get("train.overlap_mean_cosine", row.get("overlap_mean_cosine", 0)) or 0)
    hints = []
    if failed == "v7_sota_1":
        hints.append("回退 v6_sota_2 基座；禁用 NLL 仲裁与原型校准；改评测期正交 soft blend（v8_sota_1 已试）。")
    if failed == "v8_sota_1" or (oracle < 0.45 and seen < 0.4):
        hints.append("若正交门控 soft blend 无效：v8_sota_2 训练期 routing-aware ortho loss（overlap.beta 0.08 + p_i p_j cos 项）。")
    if ov > 0.75:
        hints.append(f"overlap_mean_cosine={ov:.2f} 高 → 分支参数重叠是混合干扰主因。")
    if oracle < 0.45:
        hints.append(f"oracle_agreement={oracle:.2f} 低 → 路由仍是天花板。")
    if not hints:
        hints.append("对照 v6_sota_2 单 segment 差异，选 ONE 增量变更；见 `文档记录/老师建议与下一版规划.md`。")
    return "\n".join(f"- {h}" for h in hints)


def generate(failed: str, next_ver: str, *, overwrite: bool = False) -> Path:
    out_dir = REPO / "archive" / next_ver
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "FAILURE_ANALYSIS.md"
    if out_path.is_file() and not overwrite:
        text = out_path.read_text(encoding="utf-8")
        if "status: ready" in text[:400]:
            return out_path

    run_dir = _run_dir(failed)
    rows = _read_jsonl(run_dir / "metrics.jsonl")
    final = _read_final(run_dir)
    last = final or (rows[-1] if rows else {})
    status = _guess_status(failed, rows, final)
    early, early_line = _early_stop_info(failed)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frontmatter = f"""---
failed_version: {failed}
next_version: {next_ver}
status: draft
generated_at: {now}
agent_signoff: false
---"""

    body = f"""
# 失败分析：{failed} → {next_ver}

> **自动生成证据草稿** — Agent 必须审阅、补全 §3–§5，并将 frontmatter `status` 改为 `ready`、`agent_signoff: true` 后，监控才允许 launch。

## 1. 失败版本与结果摘要

| 项 | 值 |
|---|---|
| 版本 | {failed} |
| 终态 | {status} |
| 最后 segment | {last.get('segment_id', '—')} |
| run_name | paper_instrdialog_ours_full_s123_{failed} |
| 早停 | {'是' if early else '否'} |

### 六项目标（终局/早停点）

{_sota_table(last)}

## 2. 证据链

### 2.1 轨迹 vs v6_sota_2

{_trajectory_table(rows) if rows else '（无 metrics.jsonl）'}

### 2.2 路由 / overlap / 训练（末段）

| 信号 | 值 |
|------|-----|
| oracle_agreement | {last.get('routing.oracle_agreement_rate', '—')} |
| overlap_mean_cosine | {last.get('train.overlap_mean_cosine', last.get('overlap_mean_cosine', '—'))} |
| num_branches | {last.get('num_branches', '—')} |
| train batches | {last.get('train.batches', '—')} |

### 2.3 日志摘录

```
{early_line or '（无早停行；查 ' + failed + '.log）'}
```

### 2.4 上一版 CHANGES 摘要

{_read_changes_snippet(failed)[:1200]}

## 3. 根因分析（Agent 填写）

**数据提示**：
{_suggested_hypothesis(failed, last, rows)}

**主因**：（待填写）

**机制**：（待填写）

## 4. 下一版假设（单点变更，必填）

**基座版本**：（如 v6_sota_2 或 {failed} 部分有效组件）

**唯一变更**：（待填写 — 仅 ONE 点）

**成功判据**：

## 5. Novelty 与审稿风险（Agent 填写）

（待填写）

## 6. 门禁清单

- [ ] Agent 已将 frontmatter `status: ready`
- [ ] `archive/{next_ver}/CHANGES.md`
- [ ] `configs/paper/{next_ver}.yaml` + `run_{next_ver}.sh`
"""
    out_path.write_text(frontmatter + "\n" + body.strip() + "\n", encoding="utf-8")
    return out_path


def analysis_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if re.search(r"^status:\s*ready\s*$", text, re.MULTILINE):
        return True
    if "agent_signoff: true" in text[:500]:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed", required=True)
    ap.add_argument("--next", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    path = generate(args.failed, args.next, overwrite=args.overwrite)
    print(f"Wrote {path}")
    print(f"ready={analysis_ready(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
