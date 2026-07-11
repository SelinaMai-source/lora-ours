# 分支清理候选 — 2026-07-09

**Policy:** 本轮仅执行**低风险**删除；其余分支列为候选，需用户明确同意后再删。

## 本轮已删除（低风险）

| 分支 | 类型 | 理由 |
|------|------|------|
| `ours-v1-20260708` | local + remote | 已完全并入 `ours-v1`（同 commit 线） |
| `ours-v1-strict-iteration` | local only | 旧 v1 诊断分支，无远端，已被 `ours-v1` 取代 |

## 2026-07-11 用户确认批量清理（已执行）

已删除 **本地 + 远端** 共 37 条分支：

- `ours-v18-*` … `ours-v61-*` 中间迭代（**保留 milestone**：`ours-v53`、`ours-v58`、`ours-v62`）
- `sota-24h-campaign-20260706`
- `strict-paper-repro-20260707`
- 远端 `ours_v2`（其余 `ours_v1`…`ours_v9` 远端可能已不存在或需 `git fetch --prune` 刷新）

## 必须保留（当前）

| 分支 | 理由 |
|------|------|
| `ours-v1` | **canonical v1 分支**，含 `ours_v1/` 展示包 |
| `best-method-repro-results-20260706` | 当前 repro 主工作区 |
| `ours_v10` | `origin/HEAD` 默认指向；旧 SOTA 线 |

## 中风险候选（建议打 tag 后再删）

| 分支模式 | 数量 | 说明 |
|----------|------|------|
| `ours-v18-*` … `ours-v62-*` | ~35 | strict smoke 迭代快照；可保留 milestone：`ours-v53`、`ours-v58`、`ours-v62` |
| `sota-24h-campaign-20260706` | 1 | 24h campaign 已结束 |
| `strict-paper-repro-20260707` | 1 | paper repro gate |

## 高风险候选（旧完整代码线）

| 分支 | 说明 |
|------|------|
| `ours_v1` … `ours_v9`（remote） | 2026-06 旧 SOTA 命名体系，与 hyphen `ours-v1` 不同 |
| `main` | 仅 Initial commit |

## 官方归档（保留）

- `official-method-code-archive-20260705`
- `official-repro-gate-followup-20260705`
- `ours-v0-setup`
