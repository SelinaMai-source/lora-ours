# O-LoRA Standard PEFT CL Alignment Note

Updated: 2026-07-03

## Decision

The aborted run `olora_t5large_standard_order1_seed1_official_base_formal_v55`
is a diagnostic run only. It must not be reported as an official-comparable
O-LoRA baseline because it used a single GPU with
`gradient_accumulation_steps=1`, while the official script launches eight GPUs
with per-device train batch size 8.

For the next single-GPU formal reproduction, use:

- Task order: `dbpedia -> amazon -> yahoo -> agnews`
- Model: `t5-large`
- Per-device train batch size: `8`
- Gradient accumulation: `8`
- Effective global batch: `64`
- Learning rate: `1e-3`
- Epochs: `1`
- Source/target/generation length: `512/50/50`
- O-LoRA losses: `lamda_1=0.5`, `lamda_2=0`
- W&B project: `lora-ours`

This preserves the released official task order and optimizer-update global
batch as closely as possible under the available single-GPU environment. It
remains a single-GPU official-equivalent reproduction rather than the literal
8-GPU launcher.

## Blocker Handling

If the single-GPU equivalent run diverges from official published metrics, label
the result as `official-equivalent single-GPU port`, record the gap, and do not
claim it as an exact official reproduction until an 8-GPU run or official
metric parity check is available.
