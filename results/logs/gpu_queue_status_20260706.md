# GPU queue status — 2026-07-06

Updated: 2026-07-06T13:40:00+08:00

## Current owner

Standard v85 smoke (`lora-ours-standard-v85-smoke`)

## Phase

running: Standard v85 → ToDCL queued

## Queue (continue)

1. ~~ARPER v86~~ **DONE** (BLEU 0.632 / SER 4.817)
2. **Standard v85 smoke** — RUNNING (`lora-ours-standard-v85-smoke`)
3. ToDCL ADAPTER anchor — queued (`lora_ours_gpu_queue_continue.sh`)

## Blocker resolved

ARPER monitor infinite loop held tmux 13h; fixed in `11c8ecc` follow-up scripts.

## nvidia-smi

Standard v85 training active on GPU0 (amazon round2). Queue script polling every 60s; ToDCL launches when v85 tmux exits and GPU idle.

## Disk

Root `/` cleaned to **3.2G free** (apt cache, old vscode-server, btmp truncate, wandb cache).
