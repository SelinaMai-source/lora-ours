# Ours v32 Prompt-Copy-Resistant Retry Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Audit From V31

- Run: `citb_instrdialog_order1_seed1_ours_v31_smoke_strict`.
- W&B: project `lora-ours-v31`, run `nz0uhi5b`.
- Segment2 stayed healthy:
  - `task565_circa_answer_generation` current task-aware score: `0.29`.
  - seen task-aware after segment2: `0.3433333333333333`.
- Segment3 routed correctly but generated prompt-template continuations:
  - current task debug routes: `b3=24`, `b2=1`.
  - first generated token: `25/25` `no`.
  - current debug retry accepted: `25/25`.
  - at least `17/25` current debug outputs contained prompt-template or task-definition continuations such as `Now complete`, `Input`, or `concatenated string`.

## Official Output Postprocessing

- Official Tk-Instruct prediction decoding uses `tokenizer.batch_decode(..., skip_special_tokens=True)` and strips prediction strings when saving predict outputs.
- Official `compute_metrics.py` scores the decoded prediction directly:
  - exact match lowercases, removes ASCII punctuation, and collapses whitespace.
  - ROUGE uses the raw decoded prediction against max-over-reference outputs.
- There is no official answer-only truncation that would convert `no Now complete...` into `no` for scoring. V32 therefore keeps the evaluation口径 unchanged and rejects contaminated retry generations instead of normalizing them into answers.

## Task1714 Label-Set Audit

- `task1714_convai3_sentence_generation` category is `Dialogue Generation`.
- The definition asks the model to generate a valid prediction of the user's clarifying response.
- Full task JSON contains `2295` outputs and `1959` unique normalized outputs.
- First-token buckets are common (`no`, `yes`, `i`), but they are not a closed label set; only `233` outputs are exactly bare `yes`/`no`/`i`.
- Conclusion: no official-evidence basis for classification verbalizer or constrained decoding over `yes/no/i`.

## V32 Change

- Keep v31 data, prompt protocol, target protocol, scoring, balanced sampling, and assigned-branch evaluation.
- Keep the task1714 bucket-collapse retry label-free.
- Add retry-only prompt-template blocking via `bad_words_ids` derived from configured bad-word text.
- Strengthen retry acceptance to reject template/task-definition tokens or phrases anywhere in the continuation, not only at the first word.
- Record retry bad-word count and rejection reason in debug details.

## Smoke Gate

Do not launch full strict unless v32 smoke verifies:

- segment2 task-aware remains near `0.29`.
- segment3 does not improve by accepting `no Now complete...` / `no Input...` / task-definition continuations.
- segment3 task-aware improves over v30's clean `0.08`, or the failure is clearly documented as requiring a training-time supervision fix.

## V32 Smoke Result

- Run: `citb_instrdialog_order1_seed1_ours_v32_smoke_strict`.
- W&B: project `lora-ours-v32`, run `fqjw2qzb`.
- Segment2 stayed healthy:
  - current task-aware score: `0.29`.
  - seen task-aware after segment2: `0.37`.
- Segment3 did not pass the full-strict gate:
  - current exact score: `0.01`.
  - current task-aware score: `0.12`.
  - final task-aware AR: `0.28500000000000003`.
  - retry triggered `99` times and accepted `82`, down from v31's `94` accepted retries.
  - current task debug still started `25/25` with `no`.
  - broad contamination audit found `13/25` current debug outputs still contained prompt-template or task-definition variants, and all `13` had been accepted.
  - examples include `no Now finish the following sentence`, `no Now finish the following form`, and `no if so. This is a concatenated`.

Conclusion: v32 proves the exact `Now complete` pattern can be blocked, but the retry remains brittle because the model substitutes nearby template variants. Do not launch full strict. Next step should move away from denylist-only retry acceptance toward a training-time generation supervision fix or a principled decode-time objective that is still official-scoring compatible.
