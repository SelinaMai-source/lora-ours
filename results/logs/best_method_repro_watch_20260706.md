# Best-Method Repro Watch — 2026-07-06

**Updated:** 2026-07-09T10:30:00+08:00  
**Poll interval:** 300s  
**Monitor tmux:** `lora-ours-paper-alignment-watch`  
**Queue tmux:** `lora-ours-paper-alignment-queue`

## Suite progress

| Suite | Best method | Status | Metric | Progress | ETA |
|-------|-------------|--------|--------|----------|-----|
| CITB | Replay(50) formal | **pass** | ROUGE-L AR **39.9826** | 100% | 0 |
| Standard | O-LoRA v57 | **fail*** | EM **76.8059** (gap +1.01) | 100% | 0 |
| Dialogue ToDCL | ADAPTER 37-domain | **pass** | BLEU **22.6105** / EER **0.1150** | 100% | 0 |
| Dialogue ARPER | v88 strict official | **running** | best completed: BLEU **0.59890** / SER **5.938** (v89 FAIL) | v88 ep~5/48 Train | ~8h |

### Details
- CITB: paper AR uses `average_accuracy`; prior `32.4417` was the wrong FR field.
- Standard: gap `+1.0059` exceeds strict ±1.0 by 0.006.
- ToDCL: GPT-2 blocker resolved; scorer JSON source of truth.
- ARPER: v88 healthy on GPU; v90 not launched (GPU busy).

## GPU
`run_woz3.py` active for `arper_woz3_paper_aligned_exemplar250_formal_v88`

## Active tmux
`lora-ours-arper-v88-formal` `lora-ours-paper-alignment-watch` `lora-ours-paper-alignment-queue`
