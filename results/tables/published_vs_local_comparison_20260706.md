# Published vs Local Comparison — 2026-07-06

**Branch:** `sota-24h-campaign-20260706`  
**Sources:** `official_method_matrix.md`, `official_target_manifest.yaml`, completed run logs / `final_metrics` artifacts  
**Policy:** Only **completed** runs with extractable metrics. Queued / running / smoke-without-final excluded unless noted as partial.

**Tolerance guide (informal):**
- Standard EM: ±1.0 vs paper avg considered *close* under official-equivalent single-GPU
- ARPER BLEU: ±0.03; SER: ±0.5
- CITB ROUGE-L AR: ±2.0 only if setting is script-strict comparable; paper `500/50/100` vs local `500/50/50` is a **RED FLAG** regardless of numeric gap

---

## Summary verdict

| Suite | Best local formal anchor | Matches paper? | Main blocker |
|-------|--------------------------|----------------|--------------|
| **CITB** | FT-init v54 AR **33.1** | **No** (setting + gap) | Paper `500/50/100` vs script `500/50/50`; Replay(50) formal **running** (~6%) |
| **Standard** | O-LoRA v57 EM **76.81** | **Close / Yes** | Single-GPU `grad_accum=8` official-equivalent |
| **Standard Ours** | v69 overlay EM **77.26** | N/A (increment) | Overlay on official base; beats paper O-LoRA ref |
| **Dialogue ARPER** | v66/v86 BLEU **0.632** SER **4.82** | **No** | Large BLEU/SER gap vs paper SCLSTM |
| **Dialogue ToDCL** | — | **Not run** | ADAPTER anchor queued |

---

## CITB (InstrDialog order1 seed1)

| Suite | Method | Metric | Paper published | Local result | Match? | Notes (setting differences) |
|-------|--------|--------|-----------------|--------------|--------|----------------------------|
| CITB | FT-init | ROUGE-L AR | **35.7** | **33.109** (`v54` formal, 19/19 tasks) | **No** | **RED FLAG:** local uses official-script `500/50/50` (`max_num_instances_per_eval_task=50`); paper text says `500/50/100`. Gap −2.6 even under script-strict setting. Run: `citb_instrdialog_order1_seed1_official_script_500_50_50_tie_fixed_ft_instr_stage1_v54`. |
| CITB | FT-init | BWT | **−4.6** | *not reported* | — | v54 status logs ROUGE-L only; no BWT in artifact. |
| CITB | Replay(50) | ROUGE-L AR | **40.4** | *running* (~6%, task 1/19) | — | Formal launched 2026-07-06 16:53 `lora-ours-citb-replay50-formal`; task0 done, task1 ~16% steps |
| CITB | Replay(50) | ROUGE-L AR (smoke) | **40.4** | **50.0** (`replay50_v55_smoke`, 1 task) | **N/A** | **RED FLAG:** `SMOKE=1`, `max_steps=50`, `max_eval_samples=10`, single task `task848_pubmedqa_classification` — not paper-comparable. W&B `a2jh0n4y`. |
| CITB | Replay(50) | exact_match (smoke) | — | **50.0** | **N/A** | Same smoke run; EM=ROUGE-L=50.0 on n=10 eval. |
| CITB | Ours overlay | ROUGE-L AR | **40.4** (Replay ref) | **24.6** (`ours_v85_smoke_strict`) | **No** | **RED FLAG:** 5-segment smoke (`train50_eval10_v1`), not 19-task formal. BWT **−1.25**. Early gate PASS but far below Replay paper. |
| CITB | Ours overlay | ROUGE-L AR | **40.4** (Replay ref) | **25.0** (`ours_v47_smoke_strict`) | **No** | Same 5-segment smoke protocol as v85; BWT **−0.50**. Superseded by v85. |
| CITB | Ours overlay | seen_avg_score | — | **15.8** (v85) / **15.8** (v47 `eval.seen_avg_score=0.158`) | — | Diagnostic only; smoke scale. |

---

## Standard (T5-large PEFT CL, order1 seed1)

| Suite | Method | Metric | Paper published | Local result | Match? | Notes (setting differences) |
|-------|--------|--------|-----------------|--------------|--------|----------------------------|
| Standard | O-LoRA | Final avg EM | **75.8** (paper avg); order1 ~**75.4** | **76.8059** (`v57` formal) | **Yes / Close** | Official-equivalent: single-GPU, `grad_accum=8`, full 4-task stream, no step caps. ROUGE-L **79.9715**. Forgetting **1.34** (local diagnostic). |
| Standard | O-LoRA | Final ROUGE-L | ~**79–80** (derived from v57) | **79.9715** | **Close** | Same run as above. |
| Standard | LB-CL | Final avg EM | **76.7** | *not run* | — | No official repo; paper-only baseline per matrix. |
| Standard | Progressive Prompts | Final avg EM | **76.1** | *not run* | — | Not rerun locally. |
| Standard | Ours overlay (replay64) | Final avg EM | **75.8** (O-LoRA ref) | **77.2566** (`v69` formal) | **Beat ref** | Ours increment on O-LoRA base; ROUGE-L **81.1919**. Δ vs v57 **+0.45**; vs LB-CL ref **+0.56**. **RED FLAG:** overlay — not a published baseline reproduction. |
| Standard | Ours overlay (replay64) | Final ROUGE-L | — | **81.1919** | — | v69 formal completed 2026-07-05. |
| Standard | Ours overlay (replay128) | Final avg EM | **75.8** (O-LoRA ref) | **75.7434** (`v70b` formal) | **Below ref** | Replay128 hurt retention (amazon/SC **46.96** vs v69 **54.41**). Not new best. |
| Standard | Ours overlay (SSRG v85) | dbpedia EM (partial) | — | **98.5** (round1 only) | **Incomplete** | **RUNNING / partial.** `max_steps=20`, `max_predict_samples=200`. Round2 amazon had `ROUND_EXIT_CODE:1`. No cumulative 4-task final EM. |

### Standard per-round reference (v57 official base)

| Round | Task | EM | ROUGE-L |
|-------|------|-----|---------|
| 1 | dbpedia | 98.8026 | 98.8026 |
| 2 | amazon | 74.1908 | 79.8202 |
| 3 | yahoo | 74.4079 | 79.2690 |
| 4 | agnews (final) | **76.8059** | **79.9715** |

### Standard per-round reference (v69 best overlay)

| Round | Task | EM | ROUGE-L |
|-------|------|-----|---------|
| 1 | dbpedia | 98.8026 | 98.8026 |
| 2 | amazon | 73.9211 | 78.8980 |
| 3 | yahoo | 74.4781 | 80.0636 |
| 4 | agnews (final) | **77.2566** | **81.1919** |

---

## Dialogue (ARPER MultiWOZ WOZ3, SCLSTM Path B)

| Suite | Method | Metric | Paper published | Local result | Match? | Notes (setting differences) |
|-------|--------|--------|-----------------|--------------|--------|----------------------------|
| Dialogue | ARPER (SCLSTM) | BLEU-4 | **0.701** | **0.63231** (`v66`/`v86`) | **No** | Gap −0.069. Local used DA-wise `granularity=1`, `exemplar_size=250`, `n_epochs=100`. Official default `config.cfg` is **domain-wise** (`granularity=0`). **Next:** exemplar_500 sweep when GPU free. |
| Dialogue | ARPER (SCLSTM) | SER | **3.63** | **4.817** (`v66` formal) | **No** | Gap **+1.19** (worse). |
| Dialogue | ARPER (SCLSTM) | BLEU-4 | **0.701** | **0.63231** (`v86` formal) | **No** | Identical to v66; paper-epoch repro did not improve. |
| Dialogue | ARPER (SCLSTM) | SER | **3.63** | **4.817** (`v86` formal) | **No** | Identical to v66. |
| Dialogue | ARPER + Ours repair | BLEU-4 | **0.701** | **0.63080** (repaired) | **No** | Inference-time slot repair overlay (`v81` formal). Raw **0.63231**. Not official base. |
| Dialogue | ARPER + Ours repair | SER | **3.63** | **0.394** (repaired) | **Misleading** | Repair collapses SER artificially; BLEU still below paper. Do not claim paper match. |
| Dialogue | ToDCL ADAPTER | BLEU (modular NLG) | **21.7719** | *not run* | — | Bounded method smoke only; full anchor **queued**. |
| Dialogue | ToDCL ADAPTER | EER | **0.164** | *not run* | — | Same as above. |

---

## Runs explicitly excluded (not completed / no final metrics)

| Run | State | Reason excluded |
|-----|-------|-----------------|
| `olora_official_base_ours_overlay_ssrg_v85_smoke` | running / partial | Only dbpedia EM 98.5; no 4-task cumulative |
| ToDCL ADAPTER anchor | **queued** (priority 2) | No training metrics yet |
| CITB Replay(50) formal 19-task | **running** (~6%) | `lora-ours-citb-replay50-formal`; ETA ~9–12h |
| `olora_t5large_standard_order1_seed1_official_base_smoke_v55` | completed toy | `max_steps=1`, 8 train samples — not benchmark |
| LB-CL / ProgPrompt / LFPT5 Standard baselines | not run | LFPT5 blocked (no LM-adapted T5-large); ProgPrompts blocked (protocol/env) |

---

## RED FLAG checklist (setting mismatches)

1. **CITB split:** Paper InstrDialog **500/50/100** vs all local CITB runs **500/50/50** (official script cap). Do not compare rows across these labels.
2. **CITB smoke caps:** Ours v47/v85 use **5 segments**, `train50_eval10` — not 19-task formal.
3. **CITB Replay50:** Only **1-task smoke** completed; paper Replay(50) needs full order1 stream.
4. **Standard single-GPU:** v57/v69 use **1 GPU + grad_accum=8** vs paper literal **8-GPU** `order_1.sh` — documented official-equivalent, not bit-identical.
5. **Standard v85 smoke:** `max_steps=20`, `max_predict_samples=200` — diagnostic only.
6. **ARPER v81 repair:** Post-decode inference hack — SER not comparable to paper scorer path.

---

## Artifact pointers

| Run | Log / metrics |
|-----|---------------|
| CITB FT v54 | `results/logs/citb_official_script_500_50_50_tie_fixed_status.md` |
| CITB Replay50 smoke | `results/logs/citb_replay50_base_smoke_20260706_status.md` |
| CITB Ours v85 | `results/logs/citb_instrdialog_order1_seed1_ours_v85_smoke_strict_status.md`, `results/runs/.../final_metrics.json` |
| CITB Ours v47 | `results/logs/ours_v47_strict_status.md`, `results/runs/.../final_metrics.json` |
| Standard O-LoRA v57 | `results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_monitor.md` |
| Standard Ours v69 | `results/logs/olora_official_base_ours_overlay_replay64_v69_formal_order1_seed1.final_metrics.md` |
| Standard Ours v70b | `results/logs/olora_official_base_ours_overlay_replay128_v70b_formal_order1_seed1.final_metrics.md` |
| ARPER v66 | `results/logs/arper_woz3_official_sclstm_formal_v66_status.md` |
| ARPER v86 | `results/logs/arper_woz3_official_sclstm_formal_v86_status.md` |
| ARPER v81 overlay | `results/logs/arper_woz3_official_sclstm_v66_plus_ours_overlay_v81_formal_status.md` |

*Generated: 2026-07-06 17:05. CITB Replay(50) formal in progress; numbers from completed artifacts only.*
