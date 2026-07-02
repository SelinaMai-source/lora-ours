# Ours v29 Segment3 Calibration Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## V28 Segment3 Audit

- Segment/task: `task1714_convai3_sentence_generation`.
- Official task category: `Dialogue Generation`.
- Official prompt is still CITB/Tk-Instruct style with task definition plus up to two positive examples and `Now complete the following example -`.
- The task definition asks the model to generate a user's response to a clarifying question. It is a generation task, and the relevant official health signal is max-over-reference ROUGE-L, not strict exact-match classification.
- Raw positive examples include free-form and short yes/no style outputs:
  - `i am interested in joint pain treatments in general`
  - `yes`
  - `no i am looking for lyrics of sheet music of neil youngs songs`
- Label/output distribution has a real short-answer prior:
  - processed train: 500 examples; first-token counts `no=250`, `yes=113`, `i=80`, with 29 exact `no` and 24 exact `yes`.
  - processed eval/test: 100 examples; first-token counts `no=60`, `yes=23`, `i=6`, with 7 exact `no` and 4 exact `yes`.
- V28 training supervision was weak on this segment:
  - `train.loss=3.4283`
  - `train.answer_token_acc=0.3953`
  - `mean_batch_acc=0.0592`
- V28 current-task debug slice:
  - 25/25 predictions started with `no`.
  - common outputs included `no Now complete the following`, `no if so`, and `no - no`.
  - `effective_min_new_tokens=4` forced after v27 emitted short `no`; this converted the collapse into prompt-template continuations instead of fixing the task.
  - current debug branch choices were `b2=20`, `b3=5`; NLL arbitration changed 5/25 examples but did not improve task3 output quality.
  - oracle answer-NLL on the same current debug slice often preferred old branches, so v28 prompt-reconstruction NLL was not reliable enough as the main fix for task1714.

## V29 Change

- Keep official data, prompt protocol, target protocol, and evaluation outputs.
- Remove v28's segment-specific `min_new_tokens=4`; v29 sets `enable_segment_min_new_tokens=false`.
- Fix `auto_official` metric inference so generation prompts containing the word `intent` are not misclassified as yes/no label tasks when they clearly ask to generate/produce a response.
- Add config-scoped first-token balanced generation sampling in routed training:
  - enabled only by config under `train.balanced_generation_sampling`.
  - v29 enables it only for segment name patterns `task1714` / `sentence_generation`.
  - current-task examples are bucketed as `no`, `yes`, `i`, and `other`, then interleaved per branch with a capped total multiplier.
  - replay examples from previous tasks are passed through rather than rebalanced.

## Smoke Gate

Do not launch full strict unless v29 smoke verifies:

- segment 2 remains comparable to v25-v28: task-aware score near `0.29`.
- segment 3 current debug examples are no longer dominated by prompt-template continuations such as `no Now complete the following`.
- segment 3 current-task task-aware ROUGE-L improves over v28 `0.08`.
- train metrics include balanced sampling bucket counts for task1714, confirming the calibration path was active.
