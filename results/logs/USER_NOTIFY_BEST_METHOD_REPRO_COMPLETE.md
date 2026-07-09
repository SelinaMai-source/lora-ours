# Best-Method Reproduction Status — User Notification

**Time:** 2026-07-09 10:30 UTC+8  
**Branch URL:** https://github.com/SelinaMai-source/lora-ours/tree/best-method-repro-results-20260706

## ±1 gate summary (3 suites, best method each)

| Suite | Method | Paper | Local | Gap | ±1 Verdict | Suite pass? |
|-------|--------|-------|-------|-----|------------|---------------|
| **CITB** | Replay(50) formal v56 | ROUGE-L AR **40.4** | **39.98** | −0.42 | **PASS** | **YES** |
| **Standard** | O-LoRA v57 formal | EM **75.8** | **76.81** | +1.01 | **FAIL*** | **NO** (borderline) |
| **ToDCL** | ADAPTER 37-domain anchor | BLEU **21.77** / EER **0.164** | **22.61** / **0.115** | +0.84 / −0.049 | **PASS** | **YES** |
| **ARPER** | v89 ex500+bs128 (best completed) | BLEU **0.701** / SER **3.63** | **0.599** / **5.94** | −0.10 / +2.31 | **FAIL** | **NO** |

\* Standard gap +1.0059 exceeds strict ±1.0 by 0.006 — accepted as official-equivalent anchor in prior audits but fails numeric gate.

## 3-suite best-method repro: **NOT complete** (2/3 pass; ARPER blocking)

### What matched (±1)

- **citb_replay50_v56**: AR **39.9826** via `average_accuracy` (prior 32.44 was FR field misread)
- **todcl_adapter_anchor**: BLEU **22.6105**, EER **0.115031** (GPT-2 blocker resolved; scorer JSON fixed)

### What did not match

- **standard_olora_v57**: EM **76.8059** — gap +1.01 vs paper 75.8 (borderline over ±1)
- **arper_v89**: BLEU **0.59890**, SER **5.938** — BLEU gap −0.10, SER gap +2.31

## Active jobs (do not kill)

- **ARPER v88** (`lora-ours-arper-v88-formal`): paper-exact ex250+bs128 — **running** on GPU
- **v90**: **not launched** — awaits v88 completion + GPU idle (v89 already FAIL)

## Monitors / queue

- `lora-ours-paper-alignment-watch` — polling ±1 gate
- `lora-ours-paper-alignment-queue` — serial GPU orchestrator
- `lora-ours-arper-v88-formal` — active training

## Official strictness (honest)

**Not uniformly paper-strict.** See `results/logs/official_strictness_audit_20260706.md`:
- CITB: official script ✅, paper split 500/50/100 ❌
- Standard: official-equivalent 1×GPU ✅, literal 8-GPU ❌
- ToDCL: README `train.py` path ✅
- ARPER: official `run_woz3.py` + paper-aligned hyperparams ✅, metric match ❌

## Next actions

1. Let **v88** finish; post-gate evaluates ±1 → **v90** if still FAIL and GPU free
2. Standard: optional 8-GPU relaunch or document borderline as accepted official-equivalent
3. Push this notification when 3/3 suites pass (currently blocked on ARPER SER)
