# Best-Method Repro Watch — 2026-07-06

**Updated:** 2026-07-07T04:02:29+08:00  
**Poll interval:** 300s  
**Monitor tmux:** `lora-ours-best-method-watch`  
**Queue tmux:** `lora-ours-baseline-repro-queue`

## Suite progress

| Suite | Best method | Status | Metric | Progress | ETA |
|-------|-------------|--------|--------|----------|-----|
| CITB | Replay(50) formal | **done** | ROUGE-L AR 32.4417 | 100% | 0 |
| Standard | O-LoRA v57 | **done** | EM 76.8059 | 100% | 0 |
| Dialogue ToDCL | ADAPTER 37-domain | **blocker** | — | 0% | — |
| Dialogue ARPER | v87 paper-aligned | **done** | BLEU 0.60059 SER error: | 100% | 0 |

### Details
- CITB: 19/19 tasks
- Standard: v57 formal manifest completed
- ToDCL: gpt2 load failure EXIT=1
- ARPER: formal finished

## GPU
```

```

## Active tmux
lora-ours-baseline-repro-queue lora-ours-best-method-watch 

## Queue
See `results/logs/baseline_reproduction_queue_status_20260706.md`

