# Standard PEFT CL Published-Base Increment v69

## Base

Use the O-LoRA official T5-large Standard CL order1 seed1 pipeline as the published base. The current local official-equivalent reference is `olora_t5large_standard_order1_seed1_official_base_formal_v57`.

Observed v57 formal metrics:

- completed rounds: `4/4`
- final aggregate exact: `76.8059`
- round-average exact: `81.0518`
- observed average forgetting: `1.3377`
- latest task exact: dbpedia `98.1842`, amazon `50.9868`, yahoo `70.4605`, agnews `87.5921`

Comparability note: v57 uses the official O-LoRA entry/config/order/model and single-GPU grad accumulation to match global batch. It is the strongest local published-base anchor, but final paper-comparability still needs explicit audit of single-GPU vs official launcher.

## Ours Increment

The first Standard increment is an interpretable replay overlay around the official O-LoRA training config, not a replacement runner:

- keep official task order, official entrypoint, T5-large, O-LoRA adapter chain, and official dev/test configs;
- add a small bounded prior-task replay list only to `train_tasks.json` after the first round;
- report replay-overlay metrics separately from raw O-LoRA base metrics.

The existing smoke is `olora_official_base_ours_overlay_replay64_v58_smoke_order1_seed1`:

- replay per prior task: `64`
- completed rounds: `4/4`
- smoke caps: `max_steps=20`, `predict_samples=200`
- round-average exact: `46.625`
- observed average exact: `65.75`
- status: diagnostic only, not formal comparable.

## Current Blocker

The old `core.train` v62 route repeatedly exited with `SIGTERM` during segment0 transient probe training. That route is not the published-base path, but it remains useful as a blocker diagnosis for legacy enforcer/agent-loop interference.

The neutral rerun `standard_peft_cl_o_lora_standard_order1_seed1_ours_strict_v62_formal_neutral_20260705` changes only run/log/W&B identifiers from v62 formal and is currently used to test whether legacy guard matching was the SIGTERM cause.

## Next Gate

Do not launch a Standard O-LoRA overlay formal while the v62 neutral diagnostic owns the GPU. Once the neutral diagnostic either completes or clearly fails:

1. If v62 neutral completes, record the legacy-enforcer blocker as resolved for core.train diagnostics.
2. Launch `FORMAL=1` O-LoRA official-base + ours replay overlay with no smoke caps.
3. If the replay overlay formal does not improve retention/forgetting over v57, keep it as a negative diagnostic and design the next increment as an orthogonal/residual constraint rather than expanding replay.

