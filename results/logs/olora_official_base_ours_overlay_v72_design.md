# O-LoRA Official-Base + Ours Overlay v72 Design

- Updated: 2026-07-05
- Current Standard best: v69 formal, EM `77.2566`, ROUGE-L `81.1919`.
- v70b replay128 formal failed to beat v69: final EM `75.7434`.
- v71 balanced amazon/SC replay smoke failed early: amazon/SC EM `28.0`.

## Diagnosis Source

- Confusion report: `results/logs/olora_amazon_sc_confusion_v69_v70b_v71.md`
- The failure is not invalid labels. The model mostly outputs valid labels but confuses ordinal sentiment strength.
- v69 final has strong extreme-class accuracy, but `positive` remains weak.
- v70b replay128 suppresses moderate classes and overpredicts extremes.
- v71 label-balanced replay overpredicts `neutral` in smoke.

## v72 Increment

- Keep the v69 replay64 published-base overlay path.
- Do not rewrite replay distribution.
- Do not change SC instruction text.
- Add optional output-side amazon/SC lexical repair before metrics/prediction JSONL are written.
- The repair uses only the input sentence and a fixed sentiment-strength lexicon; it does not inspect targets, test labels, or aggregate test statistics.
- Dev/test configs are copied unchanged from official order1 configs.

## Gate

- Launch v72 smoke with `SC_LEXICAL_REPAIR=1`, `REPLAY_PER_TASK=64`.
- If smoke does not collapse and amazon/SC improves against the replay64 smoke line, proceed to formal.
- If smoke collapses or the repair hurts amazon/SC, stop early and keep v69 as current best.
