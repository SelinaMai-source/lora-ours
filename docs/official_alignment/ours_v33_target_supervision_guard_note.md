# Ours v33 Target Supervision Guard Note

Generated: 2026-07-02

This is an implementation and experiment-control note. It does not claim SOTA.

## Seq2Seq Training Audit

- Training inputs are formatted with the official CITB/Tk-Instruct T5 prompt, but labels are tokenized from `targets` only via `text_target`; source prompts, definitions, and positive examples are not concatenated into labels.
- Label padding is masked to `-100`; T5 target tokenization appends EOS, and task1714 target lengths are far below `max_target_len=128`.
- No label smoothing is configured in the core seq2seq wrapper. The loss is standard supervised cross entropy over non-`-100` target labels.
- `decoder_start_token_id` is supplied for generation and the model uses the seq2seq labels path for training, so decoder inputs are produced by the HF model's shift-right behavior.

## Task1714 Target Audit

- Raw task: `task1714_convai3_sentence_generation`, category `Dialogue Generation`.
- Raw task has `2295` instances and no multi-reference instances (`min/mean/max references = 1/1/1`).
- Processed train split has `500` examples, `434` unique normalized targets, no prompt-template target contamination, and no multi-reference examples.
- Processed train first-token counts: `no=250`, `yes=113`, `i=80`, with only `53/500` exact bare `yes/no/i` targets.
- Processed target length is short but not label-only: word length median `9`, token length median `12`, max target token length `31`, and `0` examples exceed `max_target_len=128`.
- Positive examples legitimately include short `yes` / `no ...` style outputs, but the train targets are not incorrectly normalized to bare `no`.

## V33 Change

- Keep official prompt, target, decoding, and scoring protocol unchanged.
- Disable v32 bucket-collapse retry for smoke; do not rely on more prompt-template denylist/retry logic.
- Add a seq2seq target supervision guard that fails if pad tokens are supervised, and logs supervised EOS/pad counts.
- Add config-gated continuation-token loss weighting for generation targets that start with `no`, `yes`, or `i` and contain at least four words. On task1714 train, this targets `369/500` rows and emphasizes learning the continuation after the frequent first token.

## Smoke Gate

Do not launch full strict unless v33 smoke verifies:

- Segment2 remains healthy near the v32 task-aware score (`0.29`).
- Segment3 improves without bucket-collapse retry acceptance.
- Current task1714 debug outputs stop collapsing to bare `no` or prompt-template continuations.
