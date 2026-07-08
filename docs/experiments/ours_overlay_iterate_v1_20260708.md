# Ours v1 Overlay Iterate Protocol — 2026-07-08

Serial GPU queue: `scripts/run_ours_v1_20260708_iterate_queue.sh`

## Order (after ToDCL anchor releases GPU)

1. Standard v1 class-cov SSRG smoke (`lora-ours-standard-v1-smoke`)
2. CITB v1 replay_ratio smoke (`lora-ours-citb-ours-v1-smoke`)
3. ARPER v1 SSRG bounded smoke (`lora-ours-arper-v1-ssrg-smoke`)
4. ToDCL v1 assess overlay bounded smoke (`lora-ours-todcl-v1-assess-overlay`)

## Early stop gates

| Suite | Condition |
|-------|-----------|
| CITB v1 | segment 2 seen TA AR < **20** |
| Standard v1 | dbpedia EM < **90**; amazon EM < **45** |
| ARPER v1 | segment0 BLEU < **0.55**; SER > **7.0** |
| ToDCL v1 | BLEU drop > **2.0** vs anchor mid-run |

## v2 on failure

Write `docs/experiments/{suite}_failure_v1_20260708.md` → branch `ours-v2-YYYYMMDD` with single new mechanism.

Current table: `results/tables/sota_main_table_20260708.md`
