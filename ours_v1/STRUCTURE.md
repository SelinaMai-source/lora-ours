# Ours v1 分支结构说明

本文档描述 `ours-v1` 分支的目录结构。GitHub 阅读请从本目录 [`ours_v1/README.md`](README.md) 开始。

## 顶层结构（清理后）

```text
lora-ours/
├── baselines/              # 官方/高级 baseline 代码与外部方法接入
├── configs/                # 实验配置（含 v1 可运行 YAML）
├── core/                   # 通用训练入口与 Ours 方法实现
├── data/                   # 样例数据入口
├── docs/                   # 实验文档与官方对齐记录
├── official_repos/         # 官方仓库归档/子模块指针
├── ours_v1/                # ★ Ours v1 干净展示入口（推荐先读）
├── results/                # 结果表、状态摘要、运行产物
├── scripts/                # 兼容 launcher/wrapper（转调 ours_v1/scripts）
├── requirements.txt        # Python 依赖
└── BEST_METHOD_REPRO_COMPLETE.flag  # repro 完成标记（monitor 读取）
```

已删除的顶层散落文件：`test_*.py`（164 个）、`run_debug.py`。这些是 2026-06 历史 T5/debug 临时脚本，与 v1 实验流水线无关。

## ours_v1/ 内部结构

```text
ours_v1/
├── README.md               # 总入口：contribution、suite 摘要、运行命令
├── STRUCTURE.md            # 本文件：完整目录树与用途
├── core/
│   └── README.md           # 指向 core/methods/ 的通用 Ours 机制说明
├── suites/
│   ├── citb/               # CITB overlay 说明
│   ├── standard/           # Standard O-LoRA overlay 实现
│   ├── arper/              # ARPER SSRG exemplar overlay 实现
│   └── todcl/              # ToDCL assess orthogonal overlay 实现
├── configs/
│   ├── README.md           # v1 配置索引
│   └── paths.local.example # 外部路径模板（autodl-tmp 等）
├── scripts/
│   ├── queue/run_v1_iterate.sh      # 串行 GPU 队列
│   ├── launchers/run_citb_v1.sh
│   ├── launchers/run_standard_v1.sh
│   ├── launchers/run_arper_v1.sh
│   ├── launchers/run_todcl_v1.sh
│   └── monitors/README.md
└── docs/README.md          # v1 文档阅读顺序索引
```

## 各顶层文件夹用途

| 文件夹 | 做什么 | 是否需要读 |
|--------|--------|------------|
| `ours_v1/` | v1 展示包：contribution、suite overlay、launcher 入口 | **是，首选** |
| `core/` | 训练主入口 `train.py` + Ours 方法模块（SSRG、router、drift 等） | 读算法时 |
| `baselines/` | 官方 baseline 封装（O-LoRA、ARPER、ToDCL 等 vendored 代码） | 复现官方时 |
| `configs/` | 全部实验 YAML；v1 相关在 `ccfa_three_suite/*v1_20260708*` | 改参数时 |
| `scripts/` | 历史 launcher + v1 兼容 wrapper，delegate 到 `ours_v1/scripts/` | 跑实验时 |
| `docs/` | 官方对齐、first-principles、清理记录、结果状态 | 写论文时 |
| `official_repos/` | 官方 repo 归档指针（CITB、O-LoRA、ARPER、ToDCL） | 审计官方代码时 |
| `results/` | 表格、日志、run 产物（大文件不应提交 git） | 看结果时 |
| `data/` | 样例流数据 | 一般不需要 |

## 核心算法文件（contribution code）

| 机制 | 文件 | 用于哪套实验 |
|------|------|--------------|
| SSRG 谱稀疏 replay | `core/methods/ours_spectral_replay.py` | CITB（native）、Standard/ARPER（overlay） |
| Assess-then-Update | `core/methods/assess_update.py` | CITB（native）、ToDCL（overlay） |
| Router / LoRA Bank | `core/methods/router.py`, `core/methods/lora_bank.py` | CITB full Ours |
| Drift detection | `core/methods/drift_detector.py` | CITB full Ours |
| Anti-overlap | `core/methods/overlap_loss.py` | CITB full Ours |
| Standard SSRG proxy | `ours_v1/suites/standard/olora_overlay_ssrg.py` | Standard |
| Standard retention gate | `ours_v1/suites/standard/olora_overlay_assess_retention_gate.py` | Standard |
| ARPER SSRG exemplar | `ours_v1/suites/arper/arper_ssrg_exemplar_selection.py` | ARPER |
| ARPER train wrapper | `ours_v1/suites/arper/arper_ssrg_overlay_train_wrapper.py` | ARPER |
| ToDCL orthogonal hook | `ours_v1/suites/todcl/todcl_assess_orthogonal_overlay.py` | ToDCL |

## 三套实验 + v1 overlay

| 套件 | Benchmark | 底座 method | v1 overlay |
|------|-----------|-------------|------------|
| CITB | InstrDialog / InstrDialog++ | Replay(50) | SSRG replay_ratio=0.5 对齐 |
| Standard | T5-Large PEFT CL | O-LoRA v57 | class-coverage SSRG + retention gate |
| Dialogue ARPER | MultiWOZ WOZ3 | SCLSTM Path B | SSRG exemplar selection |
| Dialogue ToDCL | 37-domain NLG | ADAPTER anchor | Assess orthogonal penalty |

## 兼容 wrapper（旧路径仍可用）

以下 `scripts/` 文件 delegate 到 `ours_v1/scripts/`，保留是为了不破坏历史命令和 W&B run name：

- `scripts/run_ours_v1_20260708_iterate_queue.sh`
- `scripts/run_citb_ours_v1_20260708_smoke.sh`
- `scripts/run_olora_standard_order1_official_base_ours_overlay_v1_20260708.sh`
- `scripts/run_arper_woz3_sclstm_plus_ssrg_overlay_v1_20260708.sh`
- `scripts/run_todcl_adapter_nlg_ours_assess_overlay_v1_20260708.sh`

## 不需要读的文件类型

- 根目录已删除的 `test_*.py`、`run_debug.py`
- `results/runs/` 大运行产物
- `configs/ccfa_three_suite/` 中 100+ 历史 `ours_v18`–`v85` YAML（归档性质）
- `scripts/` 中非 v1 的历史 campaign launcher
