# Published-Setting Configs

W&B 项目（v2 队列）：`lora-published-setting-run_v2`

## 协议

- **seed**：固定 `123`
- **数据**：`train50/eval10`，`max_segments: -1`（全量 segment，禁止 toy smoke 写入主表）
- **run_name**：必须与 `output.run_name` 一致；manifest `run_name` 列对齐
- **Ours**：使用本目录 `*__ours__s123.yaml`（`mode: ours`），**非** `sota-v3` config

## Advanced baseline scaffold 说明

以下方法在 `baselines/advanced_baselines/README.md` 标注为 **scaffold**，非完整论文复现：

| 方法 | 统一入口 | 说明 |
| --- | --- | --- |
| O-LoRA | `baseline_name: o_lora` | adapter-per-segment + 正交惩罚 hook |
| LB-CL | `baseline_name: lb_cl` | SVD/projection scaffold |
| Progressive Prompts | `baseline_name: progressive_prompts` | soft prompt 模块未完整实现 |
| Continual-T0 | `baseline_name: continual_t0` | instruction replay scaffold，非 T5 mixture |

Basic baselines：`sequential_lora`、`replay_lora`（`mode: baseline`）。

## LFPT5（阻塞）

LFPT5 需独立 T5 环境与 checkpoint（见 `external_baselines/lfpt5/`），**未接入** `core/train.py`。  
Configs：`*__lfpt5__s123.yaml`；wrapper：`scripts/run_lfpt5_published_setting.py`；文档：`docs/lfpt5_published_setting.md`。  
无 `assets/pretrained/lfpt5/.../pytorch_model.bin` 时 manifest 标记 `blocked`，勿伪造结果。

## TRACE 3H delta

全量 TRACE 需 `data/processed/trace_cl_tasks_train50_eval10.json`（8 segments）。  
General / Instruction / Safety delta 指标见 `core/metrics/trace_three_h_delta.py`（stub，待 raw 就绪后实现）。

## MultiWOZ NLG

全量 5-domain：`multiwoz_nlg_cl_domains_train50_eval10.json`。  
ROUGE-L / BLEU 已在 `core/evaluate.py`；slot error 已接入为 AdapterCL-style required act-value missing rate（`eval.slot_error_rate`），历史结果不会回填，需 clean v10 rerun 才能产出该列。

## Seq-GLUE（第 5 基准）

8-task stream：`seqglue_cl_tasks_train50_eval10.json`（sst2→mrpc→rte→cola→boolq→wic→cb→copa）。  
Configs：`seqglue__*__s123.yaml`（8 法）。Gap 队列见 `scripts/run_published_setting_run_v2_gap_queue.sh`。

## TOD37

未选用（TM19/TM20/SGD 未下载）；`scripts/convert_tod37_to_stream.py --check-only` 可写出 blocker manifest，待 AdapterCL/ToDCL 原始数据就绪后再实现 37-domain stream export。
