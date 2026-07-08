# Published-Base Anchors — Locked 2026-07-08

**Branch:** `ours-v1-20260708` (from `sota-24h-campaign-20260706`)  
**Gate:** `docs/official_alignment/ccfa_experiment_gate.md`

Locked numbers for Ours overlay iteration. Do not move overlay scores into these rows.

| Suite | Published-base run | Metric | Paper ref | Locked local | Gate | Status |
|-------|-------------------|--------|-----------|--------------|------|--------|
| **CITB** | Replay(50) `formal_v56` | ROUGE-L AR | **40.4** | **39.98** | AR ≥ 39.5 | **LOCKED** |
| **Standard** | O-LoRA v57 formal | EM avg | **75.8** | **76.81** | ±1 PASS | **LOCKED** |
| **ARPER** | v89 paper-aligned SCLSTM | BLEU4 / SER | **0.701** / **3.63** | **0.59890** / **5.938** | BLEU ≥ 0.65, SER ≤ 5.0 | **LOCKED** (below paper; best local anchor) |
| **ToDCL** | ADAPTER NLG anchor | BLEU / EER | **21.77** / **0.164** | pending | BLEU ≥ 20, EER ≤ 0.20 | **RUNNING** (epoch 8/10 @ 2026-07-08T12:00) |

## Evidence

- CITB: `results/logs/strict_paper_repro_gate_1_20260707.json` → AR **39.9826**
- Standard: `results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_status.md` → EM **76.81**
- ARPER v89: `results/logs/arper_woz3_paper_aligned_exemplar500_batch128_formal_v89_status.json` → BLEU **0.59890**, SER **5.938**
- ToDCL: `lora-ours-todcl-adapter-anchor` tmux; log `/root/autodl-tmp/lora-ours-logs/todcl_adapter_nlg_official_anchor_20260706.log`

## Setting labels (disclosure)

- CITB: `official_script_500_50_50` — do not mix with paper `500/50/100`
- Standard: single-GPU + `grad_accum=8` official-equivalent port
- ARPER: Path B official SCLSTM; task order `1,7,0,6,4,2,8`; exemplar 500, batch 128
- ToDCL: legacy py37 env; 37-domain ADAPTER NLG; local GPT-2 weights

*Updated: 2026-07-08T12:05+08:00*
