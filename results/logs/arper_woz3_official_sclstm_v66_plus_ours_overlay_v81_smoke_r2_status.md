# ARPER WOZ3 v68 Published-Base + Ours Post-Decode Repair Overlay

- Updated: `2026-07-05T11:21:20`
- State: `completed`
- Label: `published-base ARPER official SCLSTM Path B v66 + ours overlay`
- Base: `arper_woz3_official_sclstm_formal_v66`
- Overlay: `SER-targeted inference-time post-decode delex slot-token count repair`
- Raw BLEU4/SER: `0.4059030836241479` / `5.637982195845697`
- Repaired BLEU4/SER: `0.4058357680169962` / `0.2967359050445104`
- v66 final BLEU4/SER reference: `0.63231` / `4.817`
- SER target reference: `3.63`
- Decision: `repair_improved_below_target`

## Integrity Notes

- Ground truth, data split, checkpoint, and official scorer are unchanged.
- This is inference-time repair and must not be reported as the official SCLSTM base.
- Repair only changes generated delexicalized slot tokens before metric recomputation.

## Repair Actions

- `{'removed_redundant': 20, 'added_missing': 34, 'changed_generations': 32, 'unchanged_generations': 480}`
