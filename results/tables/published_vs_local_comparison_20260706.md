# Published vs Local Comparison — 2026-07-06 (best-method-only)

**Branch:** `sota-24h-campaign-20260706`  
**Scope:** Only the **best non-Multi** method per suite (4 targets across 3 suites).  
**Policy:** Completed runs with extractable metrics; running/queued noted inline.

---

## Target methods summary

| # | Suite | Method | Paper metric | Local result | Status | Verdict |
|---|-------|--------|--------------|--------------|--------|---------|
| 1 | **CITB** | Replay(50) | ROUGE-L AR **40.4** | *in progress* | **running** | Pending formal 19-task |
| 2 | **Standard** | O-LoRA | EM avg **75.8** | EM **76.8059** (`v57`) | **done** | **Close / Yes** — best-available official anchor |
| 2b | **Standard** | LB-CL | EM **76.7** | *not run* | **paper_only** | No official code; O-LoRA v57 substitutes |
| 3a | **Dialogue ARPER** | SCLSTM exemplar | BLEU **0.701**, SER **3.63** | BLEU **0.632**, SER **4.82** (v66/v86); v87 queued | **partial → queued** | Below paper; v87 domain/exemplar500 repro next |
| 3b | **Dialogue ToDCL** | ADAPTER NLG | BLEU **21.77**, EER **0.164** | *not run* | **queued** | After CITB Replay(50) |

---

## 1 — CITB: Replay(50)

| Metric | Paper | Local | Match? | Notes |
|--------|-------|-------|--------|-------|
| ROUGE-L AR | **40.4** | *running* (~22% task 1/19) | — | Formal `lora-ours-citb-replay50-formal` since 16:53 |
| BWT | up to **1.6** | — | — | Pending |

**Setting disclosure:** All local CITB runs use official-script `500/50/50`; paper text says `500/50/100` — disclose as RED FLAG.

**Skipped (not best-method):** FT-init, L2, EWC, AGEM, Replay(10).

---

## 2 — Standard: O-LoRA (LB-CL paper_only)

| Metric | Paper | Local (`v57` formal) | Match? | Notes |
|--------|-------|----------------------|--------|-------|
| Final avg EM | **75.8** (O-LoRA); **76.7** (LB-CL) | **76.8059** | **Yes / Close** | **Completed best-available official anchor** vs LB-CL `paper_only` (no code) |
| Final ROUGE-L | ~79–80 | **79.9715** | **Close** | Single-GPU `grad_accum=8` official-equivalent |

### v57 per-round reference

| Round | Task | EM | ROUGE-L |
|-------|------|-----|---------|
| 1 | dbpedia | 98.8026 | 98.8026 |
| 2 | amazon | 74.1908 | 79.8202 |
| 3 | yahoo | 74.4079 | 79.2690 |
| 4 | agnews (final) | **76.8059** | **79.9715** |

**Skipped (not best-method):** SeqLoRA, IncLoRA, Replay, LFPT5, Progressive Prompts.

---

## 3a — Dialogue ARPER: SCLSTM exemplar

| Metric | Paper | Local v66/v86 | Match? | Notes |
|--------|-------|---------------|--------|-------|
| BLEU-4 | **0.701** | **0.63231** | **No** | DA-wise `granularity=1`, exemplar 250 |
| SER | **3.63** | **4.817** | **No** | Gap +1.19 |

**Next (queued):** v87 paper-aligned — domain-wise (`granularity=0`), exemplar **500**, official `task_seq=0,5,2,1,3,4`.

---

## 3b — Dialogue ToDCL: ADAPTER modular NLG

| Metric | Paper | Local | Match? | Notes |
|--------|-------|-------|--------|-------|
| BLEU | **21.7719** | *not run* | — | Queued priority 2 |
| EER | **0.164** | *not run* | — | 37-domain anchor script ready |

---

## RED FLAG checklist

1. **CITB split:** Paper `500/50/100` vs local `500/50/50` — always disclose.
2. **Standard single-GPU:** v57 uses 1 GPU + `grad_accum=8` vs paper 8-GPU — documented official-equivalent.
3. **LB-CL:** No code — O-LoRA v57 is the operational anchor, not a LB-CL reproduction.
4. **ARPER v66/v86:** DA-wise setting may not match paper domain-wise row; v87 addresses this.

---

## Artifact pointers

| Run | Log / metrics |
|-----|---------------|
| CITB Replay50 formal | `results/logs/citb_replay50_formal_20260706.log` |
| Standard O-LoRA v57 | `results/logs/olora_t5large_standard_order1_seed1_official_base_formal_v57_monitor.md` |
| ARPER v66/v86 | `results/logs/arper_woz3_official_sclstm_formal_v66_status.md` |
| ARPER v87 (queued) | `results/logs/arper_woz3_paper_aligned_exemplar500_formal_v87.cfg` |

*Updated: 2026-07-06 17:15. Best-method-only scope; 4 target rows across 3 suites.*
