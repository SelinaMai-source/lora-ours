# Ours Overlay Iterate Protocol — 2026-07-06

Serial GPU queue: `scripts/run_ours_overlay_iterate_queue.sh` (tmux `lora-ours-gpu-queue`).

## Order (baselines before overlay)

1. CITB Replay50 official_script 500/50/50 smoke (`lora-ours-citb-replay50-smoke`)
2. Standard v85 SSRG + assess-retention smoke (`lora-ours-standard-v85-smoke`)
3. ARPER v86 paper-epoch formal (`lora-ours-arper-v86-formal`)
4. ToDCL ADAPTER NLG anchor (`lora-ours-todcl-adapter-anchor`)
5. CITB ours v85 smoke (`lora-ours-citb-ours-v85-smoke`)

## Early stop gates

| Suite | Condition |
|-------|-----------|
| CITB | First 3 segments TA AR < 15 |
| Standard | dbpedia EM < 90; amazon EM < 45 |
| Dialogue | segment0 BLEU < base − 0.02; SER +0.5 |

## vN+1 on failure

Sentinel wake → `docs/experiments/{suite}_failure_vN.md` → single-mechanism patch → smoke → formal.

Current gaps documented in `results/tables/sota_main_table_20260706.md`.
