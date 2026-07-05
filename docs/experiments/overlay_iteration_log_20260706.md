# Overlay Iteration Log — 2026-07-06

**Branch:** `sota-24h-campaign-20260706`  
**Serial queue:** `scripts/run_ours_overlay_iterate_queue.sh` (tmux `lora-ours-overlay-queue`)  
**Gate artifacts:** `results/logs/ours_overlay_iterate_smoke_gates_20260706.md`  
**Main table:** `results/tables/sota_main_table_20260706.md`

## Version matrix (campaign queue)

| # | Job ID | Version | Suite | Launcher / config | tmux | W&B group | State | Gate |
|---|--------|---------|-------|-------------------|------|-----------|-------|------|
| 1 | citb-replay50-smoke | base v55 | CITB | `run_citb_instrdialog_replay50_official_base_repro.sh` `SMOKE=1` | `lora-ours-citb-replay50-smoke` | `lora-ours-v55-citb-replay50-base` | smoke **PASS** | EM/ROUGE-L **50.0** (1-task cap) |
| 2 | standard-v85-smoke | overlay v85 | Standard | `run_olora_standard_order1_official_base_ours_overlay_v85.sh` | `lora-ours-standard-v85-smoke` | `published-base-standard-olora-plus-ours-overlay-v85-smoke` | **INCOMPLETE** | no eval; GPU serial violation |
| 3 | arper-v86-formal | anchor v86 | Dialogue ARPER | `run_arper_woz3_official_sclstm_formal_v86.sh` | `lora-ours-arper-v86-formal` | n/a (Path B official) | **queued** | blocked on GPU |
| 4 | todcl-adapter-anchor | anchor | Dialogue ToDCL | `run_todcl_adapter_nlg_official_anchor.sh` | `lora-ours-todcl-adapter-anchor` | n/a | **queued** | preflight PASS |
| 5 | citb-ours-v85-smoke | ours v85 | CITB | `citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml` | `lora-ours-citb-ours-v85-smoke` | `ccfa_citb_strict_ours_v85_smoke` | **queued** | awaits serial GPU |

## v85 overlay switches (Standard)

| Switch | Value | Doc |
|--------|-------|-----|
| `REPLAY_MODE` | `ssrg` | spectral sparse replay64 |
| `ASSESS_RETENTION_GATE` | `1` | reject high-interference LoRA updates (threshold 0.3) |
| `EARLY_GATE_DBPEDIA_EM` | `90` | round1 smoke early stop |
| `EARLY_GATE_AMAZON_EM` | `45` | round2 smoke early stop |

Reference: `docs/experiments/standard_olora_overlay_v85_first_principles.md`

## Early-gate rules (campaign)

| Suite | Trigger | Action on fail |
|-------|---------|----------------|
| CITB | First 3 segments TA AR < **15** | stop; sentinel wake → `docs/experiments/citb_failure_vN.md` |
| Standard | dbpedia EM < **90** or amazon EM < **45** | stop; single-mechanism vN+1 patch |
| Dialogue ARPER | segment0 BLEU < base − **0.02** or SER +**0.5** | stop; overlay iteration |
| Dialogue ToDCL | anchor BLEU/EER deviates >5% from README | stop; env/data audit |

**vN+1 protocol:** sentinel wake → failure note → one auditable mechanism patch → smoke → formal.

## Run notes (2026-07-06 00:39–00:45 CST)

### CITB Replay50 official base smoke — PASS

- Completed exit 0; single-task smoke `task848_pubmedqa_classification`.
- Metrics: `eval/exact_match=50.0`, `eval/rougeL=50.0` (n=10 cap).
- W&B run `a2jh0n4y`.
- Detail: `results/logs/citb_replay50_base_smoke_20260706_status.md`.
- **Caveat:** a longer Replay50 repro may still occupy GPU0 after smoke; queue must wait for idle GPU before Standard/ARPER/ToDCL.

### Standard v85 SSRG + assess-gate smoke — INCOMPLETE (gate indeterminate)

- Launched 00:39; W&B run `45amv0bn` (`olora_official_base_ours_overlay_ssrg_v85_smoke_order1_seed1_round1_dbpedia`).
- Reached dbpedia train step **14/20** (~70%); **no predict/eval metrics** written.
- **Blocker:** concurrent CITB compute on GPU0 (~16 GiB + ~8 GiB overlap window); violates serial queue policy. Process interrupted; tmux session restarted shell but training not resumed to eval.
- Status file still `state: running` (stale); log: `results/logs/olora_official_base_ours_overlay_ssrg_v85_smoke_order1_seed1.log`.
- **Gate verdict:** **INDETERMINATE** — cannot PASS/FAIL early gates without dbpedia EM. Re-queue after CITB releases GPU.

### CITB ours v85 smoke — QUEUED

- Config: `configs/ccfa_three_suite/citb_instrdialog_order1_seed1_ours_v85_smoke_strict.yaml`
- Launch plan: `results/logs/citb_ours_v85_smoke_launch_20260706.md`
- Blocked until GPU idle and Standard v85 slot clears in serial order.

### ARPER v86 formal — QUEUED

- Paper-default anchor: 100 epochs, batch 64, exemplar 250, task order `1,7,0,6,4,2,8`.
- Launch metadata: `results/logs/arper_woz3_official_sclstm_formal_v86_launch.json` (`state: queued_gpu_busy`).
- Compare target at completion: v66 BLEU **0.632** / SER **4.82** vs paper **0.701** / **3.63**.

### ToDCL ADAPTER NLG anchor — QUEUED

- Preflight PASS: 37-domain stream (`train=31425`, `dev=4035`, `test=4742`).
- Preflight JSON: `results/logs/todcl_adapter_nlg_official_anchor_20260706_preflight.json`
- Paper reference: BLEU **21.77** / EER **0.164**.

## SOTA targets (unchanged)

| Suite | Target |
|-------|--------|
| CITB ROUGE-L AR | **53.87** |
| Standard avg EM | **84.5** |
| Dialogue BLEU | **0.935** (ARPER); **29.0** (ToDCL) |

**Honest gap:** no suite reached numerical SOTA margin in this campaign slice. Best prior ours: Standard v69 EM **77.26** (−7.2 vs target).

## Next actions (do not kill live tmux)

1. Let CITB Replay50 repro finish; wait `gpu_empty` in overlay queue.
2. Re-run Standard v85 smoke to dbpedia eval; write PASS/FAIL to `ours_overlay_iterate_smoke_gates_20260706.md`.
3. Serial launch ARPER v86 → ToDCL anchor → CITB ours v85 per queue order.
