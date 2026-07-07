# CITB Replay(50) 复现差距审计 — 2026-07-07

**分支:** `strict-paper-repro-20260707`  
**本地正式跑:** `formal_v56`（19/19 tasks, EXIT 0）  
**论文目标:** ROUGE-L **AR (`average_accuracy`) = 40.4**  
**±1 门控:** 百分点 ±1.0（非 FR 列）

---

## 执行摘要

| 指标 | 论文 / released JSON | 本地 v56（错误读数） | 本地 v56（官方矩阵） | ±1? |
|------|---------------------|---------------------|---------------------|-----|
| **ROUGE-L AR** (`average_accuracy`) | **40.4** | 32.44（误用 `predict_official_rougeL`） | **39.98** | **PASS** (−0.42) |
| FR on T_unseen (`final_official_test_score`) | **31.8** | 32.44 | **32.44** | **PASS** (+0.64) |
| FR on T_init (`final_initial_multi_test_score`) | **47.1** | — | **46.32** | **PASS** (−0.78) |
| BWT | **1.6** | — | **0.65** | PASS (within 2σ) |
| FWT | **22.9** | — | **19.73** | FAIL (−3.17) |

**关键结论:** 先前报告的 **−7.96 gap 是指标混淆**，不是训练失败。`predict_official_rougeL` 对应论文 FR 列（unseen test），**不是** Table 1 的 AR。用 `scripts/parse_citb_official_results.py` / `baselines/citb_t5/citb_metrics.py` 从 19 个 `metrics.json` 构建 score matrix 后，**AR 已在 ±1 内**。

---

## 假设排序（按证据强度）

### H1 — 指标解析错误（已证实，主因）

- **现象:** Tracker 将 `all_results.predict_official_rougeL`（32.44）当作 AR。
- **证据:** 官方 `collect_results.py` 的 `average_accuracy` = 最终行 `a_{T,i}` 均值；本地矩阵计算 **39.98**。
- **修复:** 所有门控/tracker 改用 `summarize_method()` → `average_accuracy`；禁止用单任务 JSON 尾部字段比 AR。
- **状态:** 需更新 `baseline_reproduction_tracker`、`monitor_paper_alignment.sh`、`run_strict_paper_repro_iteration.sh`。

### H2 — Stage-1 checkpoint 不一致（次要，仍建议对齐）

| 项 | v56 本地 | 官方 `run_cit_replay.sh` |
|----|----------|--------------------------|
| Stage-1 | `seed469` @ `model_cache/citb_superni_stage1/` | `seed50/checkpoint-14000` |
| 磁盘 | 有（safetensors） | **无公开下载** |
| Tokenizer | `google__t5-small-lm-adapt` 覆盖 | checkpoint 原生 |

- **影响:** AR 已接近论文，但 FWT 偏低 3.2pt；完全字节级复现仍需本地训练 Stage-1（`scripts/run_citb_stage1_seed50_train.sh`）。
- **优先级:** P2（±1 门控已通过时不必阻塞）；P0 若用户要求与官方脚本逐 flag 一致。

### H3 — 数据划分 500/50/50 vs 500/50/100（已排除为主因）

- 官方 `run_cit_replay.sh`: `max_num_instances_per_eval_task=50` → **500/50/50**。
- Released scores `average_train_samples=5855.5`（≈411.5/task）匹配 500/50/50。
- Paper 文本 500/50/100：**4/19 order-1 任务数据不足**（`preflight_citb_official_split_counts.py`）。
- **结论:** 不是 −7.96 的来源；保持 `official_script_500_50_50` 披露。

### H4 — 工程 shim（tie_word_embeddings、tokenizer、collator）

| Shim | 假设影响 | v56 下 AR 证据 |
|------|----------|----------------|
| `tie_word_embeddings=False` | 中 | AR 仍 ≈40 |
| lm-adapt tokenizer 覆盖 | 高（若 Stage-1 错） | 未单独消融 |
| GPT-2 metric redirect | 低 | 仅评测路径 |
| collator `add_task_id` | 阻塞级若错树 | v56 已通过 preflight |

### H5 — 随机种子（低）

- 官方 Stage-2: `shuf -i 10-999`；本地固定 `seed=1`。
- AR 方差 released std=0.0；种子可能影响 FWT，不太解释 8pt AR 假象。

### H6 — 聚合 parser vs 官方 score matrix（已验证一致）

- `matrix_from_official_metric_jsons` + `compute_average_accuracy` 与 `collect_results.py` 同定义。
- Released `all_seen_task_scores` 末项 40.4 与本地最终行均值 39.98 一致量级。

---

## Fix plan v1

| 迭代 | 动作 | 预期 | 阻塞 |
|------|------|------|------|
| **0（立即）** | 修正 tracker/monitor 指标口径 | CITB 标记 **PASS ±1** | 无 |
| **1（可选）** | `run_citb_stage1_seed50_train.sh` → checkpoint-14000 | FWT 拉近 22.9 | ~2–4 GPU-h |
| **2（可选）** | `run_citb_replay50_paper_aligned_v2.sh`（官方 Stage-1 + 原生 tokenizer） | 严格脚本 parity | 依赖迭代 1 |
| **诊断** | `ALLOW_FALLBACK_STAGE1=1` 仅 smoke | 不用于论文门控 | — |

**Fix attempt #1（若仍要 strict parity）:** 不在 GPU 上打断 ToDCL；ToDCL 完成后排队 Stage-1 seed50 → Replay v2 formal。

---

## 参考路径

- Released scores: `citb_official/scores/.../CL=REPLAY/scores.json` → `average_accuracy: 40.4`
- v56 结果: `/root/autodl-tmp/citb_official_base_repro/.../formal_v56/results/`
- Split preflight: `results/logs/citb_official_split_counts_500_50_100_replay50.json`
- Stage-1 blocker: `results/logs/citb_stage1_checkpoint14000_preflight_20260707.json`
- v2 launcher: `scripts/run_citb_replay50_paper_aligned_v2.sh`

*Updated: 2026-07-07*
