# Ours v1 清理记录 — 2026-07-09

**Branch:** `ours-v1`  
**Worktree:** `/root/lora-ours-ours-v1`

## 已删除文件

| 类别 | 数量 | 说明 |
|------|------|------|
| 顶层 `test_*.py` | 164 | 2026-06 V8 时期 T5/format 手动调试脚本，无 import、无 launcher 引用 |
| `run_debug.py` | 1 | 会原地改写 `configs/debug_strict.yaml`，无 pipeline 引用 |
| tracked `__pycache__/*.pyc` | 0 | 本分支无 git 跟踪的 pyc |

示例已删文件：`test_t5_format.py`、`test_t5_local_gen*.py`、`test_config.py` 等。

## 保留的顶层文件

| 文件 | 原因 |
|------|------|
| `requirements.txt` | Python 依赖 |
| `BEST_METHOD_REPRO_COMPLETE.flag` | monitor 脚本读取 |
| `.gitignore`、`.gitmodules` | repo 元数据 |

## 未删除（本轮）

- `core/`、`baselines/`、`scripts/`、`configs/`、`official_repos/`：运行依赖
- `results/`：结果表与状态摘要（大 run 不提交）
- `configs/debug_strict.yaml` 等 debug 配置：在 `configs/` 子目录，非顶层污染

## 结构文档

- 展示入口：[`ours_v1/README.md`](../../ours_v1/README.md)
- 完整目录树：[`ours_v1/STRUCTURE.md`](../../ours_v1/STRUCTURE.md)
