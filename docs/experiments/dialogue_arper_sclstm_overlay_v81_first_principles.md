# Dialogue ARPER v81 First-Principles Note

## Base

Use the official ARPER SCLSTM run as the published base. The concrete reference is `arper_woz3_official_sclstm_formal_v66`, which produced BLEU4 `0.63231` and SER `4.817` using the official scorer.

## Ours Increment

v81 is an interpretable overlay on SCLSTM outputs. It does not replace the generator. The first increment is inference-time slot-token count repair:

- compute required delexicalized slot counts from the official DA/slot features;
- remove redundant generated slot tokens;
- append missing required slot tokens;
- recompute metrics with the official ARPER `util.get_slot_error` and `util.get_bleu`.

## Failure Chain

The weak LoRA/router generator path had low BLEU because it did not model the dialogue-act generation distribution well. The SCLSTM base already solves the language modeling part. The remaining target is SER safety: reduce missing/redundant slot errors while keeping BLEU close to the published base.

## Comparable Boundary

- No dev/test target is used to decide repairs.
- Official checkpoint, split, feature parser, and scorer are reused.
- Raw SCLSTM and repaired overlay metrics are reported separately.
- Smoke uses a batch cap only for gate/debug. Formal removes the cap.

