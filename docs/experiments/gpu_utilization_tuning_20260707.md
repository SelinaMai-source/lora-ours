# GPU 利用率调优记录 — 2026-07-07

## 现状快照（v89 运行中，未中断）

| 指标 | 值 |
|------|-----|
| 会话 | `lora-ours-arper-v89-formal` (PID 227471) |
| 进度 | ~42%（123/290 epoch 量级；Hotel 完成，Booking Epoch 4+） |
| GPU 显存 | **907 MiB / 49140 MiB** (~1.8%) |
| GPU 利用率 | **脉冲式 0–25%**（epoch 间 validation/CPU 阶段接近 0%） |
| 配置 batch | 128（论文对齐） |
| 错误 | 0 |

**决策：不修改 v89 运行中进程。** 进度已超过 5% 且接近 50%，中途改 batch 需从头/任务边界重启，收益不值得。

---

## 低利用率根因分析

1. **模型极小**：SCLSTM `hidden_size=128`，`num_layer=1`，单步计算量远小于 48GB vGPU 吞吐能力。
2. **batch 相对 GPU 容量过小**：bs=128 时实测 forward+backward 峰值仅 **285 MiB**（见 probe），整机 <1GB。
3. **无 PyTorch DataLoader**：`loader/task.py` 手工 Python 循环组 batch，**无 `num_workers`**，数据准备在 CPU 单线程完成。
4. **cuDNN 确定性模式**：`run_woz3.py` 默认 `cudnn.deterministic=True`，未开启 `benchmark`。
5. **RNN 逐 token 解码**：SCLSTM 序列生成无法充分饱和 Tensor Core；利用率呈 epoch 内脉冲而非持续满载。

---

## 显存探测（1 batch forward+backward）

脚本：`scripts/probe_arper_batch_memory.py`  
结果：`results/logs/arper_batch_memory_probe_20260707.json`  
VRAM 上限：**80% × 49140 ≈ 39312 MiB**

| batch_size | peak MiB | 安全 |
|------------|----------|------|
| 128 | 285.4 | ✓ |
| 256 | 557.6 | ✓ |
| 512 | 1113.5 | ✓ |
| 1024 | 2181.0 | ✓ |
| 2048 | 4297.0 | ✓ |
| 4096 | 8616.7 | ✓ |

**max_safe_batch = 4096**（仍远低于 80% 上限）。  
探测在 v89 同卡并行执行，v89 训练未受影响（仍 ~907 MiB）。

---

## 已应用变更

### 1. ARPER v90 下一跑启动器（**未启动**）

- `scripts/run_arper_woz3_paper_aligned_formal_v90.sh`
- `results/logs/arper_woz3_paper_aligned_exemplar500_batch1024_gpu_tuned_formal_v90.cfg`
- **batch_size: 128 → 1024**（4×，峰值 ~2.2GB，留足 20% headroom）
- `ARPER_CUDNN_BENCHMARK=1`（v90 默认开启）
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 2. cuDNN benchmark 可选开关

- `baselines/.../run_woz3.py`：`ARPER_CUDNN_BENCHMARK=1` 时 `benchmark=True`；默认行为不变（v89 不受影响）。

### 3. ToDCL 排队任务 GPU 调优（**未启动**，等 v89 后 gate 触发）

- `scripts/run_todcl_adapter_nlg_official_anchor.sh`
- `train_batch_size`: 10 → **16**
- `gradient_accumulation_steps`: 8 → **5**（有效 batch 仍 = **80**）
- `valid_batch_size`: 默认 **32**
- 可通过环境变量覆盖：`TODCL_TRAIN_BATCH_SIZE`, `TODCL_GRADIENT_ACCUM`, `TODCL_VALID_BATCH_SIZE`

### 4. 显存探测工具

- `scripts/probe_arper_batch_memory.py`（可复用于 v90 前复检）

---

## 预期 GPU 利用率

| 任务 | 当前 (v89) | 优化后预期 | 说明 |
|------|-----------|-----------|------|
| ARPER v89 | ~5–25% 脉冲 | *不变* | 不中断当前跑 |
| ARPER v90 (bs=1024) | — | **30–55%** 脉冲 | 更大 batch + cudnn.benchmark；仍受 CPU/RNN 限制 |
| ToDCL (bs=16) | — | **40–70%** | GPT-2 adapter 计算量更大，提升更明显 |

> 小模型 SCLSTM 很难持续 80%+ 利用率；目标是在 **不 OOM** 前提下缩短 wall-clock，而非追求满载。

---

## 风险与注意事项

| 风险 | 级别 | 缓解 |
|------|------|------|
| OOM | 低 (v90 bs=1024) | probe 验证；80% VRAM cap；`expandable_segments` |
| 训练动态变化 (大 batch) | 中 | v90 仅作 v89 FAIL 后的下一迭代；与论文 bs=128 有偏差，需记录 |
| cuDNN 非确定性 | 低 | 仅 v90 默认开启；seed 仍固定，小幅数值漂移可接受 |
| 中断 v89 | **已避免** | 未改 cfg / 未 kill 进程 |
| ToDCL 有效 batch 漂移 | 低 | 保持 16×5=80=10×8 |

### 激进选项（未默认启用）

- ARPER `batch_size=2048`（~4.3GB peak）：可在 v90 cfg 中手动调整后再跑 probe。
- ToDCL `TODCL_TRAIN_BATCH_SIZE=20 TODCL_GRADIENT_ACCUM=4`：需首次 epoch 观察 `nvidia-smi`。

---

## 监控命令

```bash
# 持续观察
watch -n2 nvidia-smi

# v89 状态
tmux capture-pane -t lora-ours-arper-v89-formal:train -p | tail -20

# 启动 v90（仅 v89 结束或 FAIL 后）
bash scripts/run_arper_woz3_paper_aligned_formal_v90.sh
```

---

## 变更文件清单

- `scripts/probe_arper_batch_memory.py` (new)
- `scripts/run_arper_woz3_paper_aligned_formal_v90.sh` (new)
- `results/logs/arper_woz3_paper_aligned_exemplar500_batch1024_gpu_tuned_formal_v90.cfg` (new)
- `results/logs/arper_batch_memory_probe_20260707.json` (new)
- `baselines/advanced_baselines/arper_dialog_nlg/external/run_woz3.py` (cudnn env gate)
- `scripts/run_todcl_adapter_nlg_official_anchor.sh` (batch env tuning)
