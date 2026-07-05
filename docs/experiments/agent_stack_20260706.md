# Agent Stack — lora-ours 三套件 SOTA 监控 (2026-07-06)

## 概述

Agent stack 由两个 tmux session 组成，统一监控 CITB / Standard / Dialogue 三套件实验状态，并在 blocker 出现时通过 `SOTA_AGENT_WAKE.flag` 唤醒 Cursor agent。

| 组件 | tmux session | 脚本 |
|------|--------------|------|
| Sentinel 监控 | `lora-ours-sentinel` | `scripts/lora_ours_sentinel.py` |
| Agent wake loop | `lora-ours-agent-loop` | `scripts/sota_agent_loop.sh` |

## 一键启动

```bash
cd /root/lora-ours
bash scripts/launch_lora_ours_agent_stack.sh
```

环境变量（可选）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `SENTINEL_POLL_SEC` | `120` | Sentinel 轮询间隔（秒） |
| `SOTA_AGENT_POLL_SEC` | `30` | Agent loop 读 flag 间隔 |
| `SOTA_AGENT_LOOP_SEC` | `300` | Agent heartbeat 间隔 |
| `SENTINEL_EMIT_WAKE` | `1` | Sentinel 是否写 wake flag |

## 状态汇报

Sentinel 每轮扫描 `results/logs/*_status.json` 与 `*_status.md`，按文件名归类三套件，取各套件最新 JSON（若无则 MD），写入：

- `results/logs/lora_ours_sentinel_status.md` — 人类可读汇总表
- `results/logs/lora_ours_sentinel_status.json` — 机器可读 JSON
- `results/logs/lora_ours_sentinel.monitor.log` — 轮询日志

### 套件归类规则

| 套件 | 文件名前缀/关键词 |
|------|-------------------|
| CITB | `citb_`, `instrdialog` |
| Standard | `olora_`, `standard_peft_`, `published_base_olora` |
| Dialogue | `dialogue_`, `arper_`, `todcl_` |

### SOTA 目标（计划内）

| 套件 | 指标 | Baseline | SOTA 目标 |
|------|------|----------|-----------|
| CITB | ROUGE-L AR | 40.4 | 53.87 |
| Standard | avg EM | 76.7 | 84.5 |
| Dialogue | BLEU-4 | 0.701 | 0.935 |

状态表中的 **SOTA gap** 列显示当前最佳 run 相对目标的差距。

### Wake 触发条件

当任一套件状态为 `failed` / `incomplete` / `crashed` / `stalled` / `stopped_incomplete` 等失败态，Sentinel 会：

1. 在 status md 的 **Needs Agent Attention** 段列出详情
2. 写入 repo 根目录 `SOTA_AGENT_WAKE.flag`（JSON payload）
3. 追加 `SOTA_AGENT_WAKE.log`
4. stdout 打印 `AGENT_LOOP_WAKE_LORA_OURS {...}` 供 Cursor `/loop` 捕获

Agent loop（`lora-ours-agent-loop`）每 30s 检测 flag，若存在则 emit `AGENT_LOOP_WAKE_LORA_OURS`；每 300s 发 heartbeat `AGENT_LOOP_TICK_LORA_OURS`。

## 手动检查

```bash
# 是否应唤醒 agent（exit 0 = 是）
bash scripts/sota_agent_check.sh

# 单次 sentinel tick（不写 wake flag）
python3 scripts/lora_ours_sentinel.py

# 单次 tick 并在 blocker 时写 flag
python3 scripts/lora_ours_sentinel.py --emit-wake
```

## 附加 session

```bash
tmux attach -t lora-ours-sentinel    # 看监控轮询
tmux attach -t lora-ours-agent-loop  # 看 wake 输出
tmux ls | rg lora-ours
```

## Agent 工作流（唤醒后）

1. 读 `lora_ours_sentinel_status.md` + 对应 suite 的 `*_status.md/json`
2. 若训练健康则 **不 kill**；仅对失败/early-gate/blocker 做分析
3. 写 `docs/experiments/{suite}_failure_vN.md`，提出 **单机制** vN+1 patch
4. smoke gate → formal；更新 `docs/official_alignment/status.md`

## 文件索引

- Sentinel: `scripts/lora_ours_sentinel.py`
- Agent loop: `scripts/sota_agent_loop.sh`
- Launcher: `scripts/launch_lora_ours_agent_stack.sh`
- Quick check: `scripts/sota_agent_check.sh`
- Wake flag: `/root/lora-ours/SOTA_AGENT_WAKE.flag`
- 计划: `sota-24h-campaign-20260706` 分支
