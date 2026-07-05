# v75 Train-Heldout Retention/Gating Diagnostic

## Design

- Base remains O-LoRA official T5-large Standard CL + ours replay overlay; no base replacement.
- Heldout source: `amazon/train.json` slice `[4500:5000]`, 500 examples. This is training-split-only and does not read `dev.json`/`test.json`; manifests record `uses_test_json=false`.
- Diagnostic target: estimate whether a train-heldout retention gate can detect harmful replay/retention choices before launching more formal runs.
- Candidate gate: after amazon and after each later task, evaluate amazon/SC train-heldout. If heldout EM drops more than ~3 points from the best previous adapter, or if moderate labels collapse (`negative`/`positive` near zero predictions), reject that candidate overlay and do not promote it to formal.

## Heldout Results

| run | adapter | heldout EM | heldout ROUGE-L | signal |
| --- | --- | ---: | ---: | --- |
| v69 replay64 | round2 amazon | 53.4 | 64.0333 | baseline heldout after amazon |
| v69 replay64 | round3 yahoo | 53.4 | 71.6333 | stable |
| v69 replay64 | round4 final | 55.2 | 71.0 | stable/improved |
| v70b replay128 | round2 amazon | 51.4 | 65.6 | weaker than v69 |
| v70b replay128 | round3 yahoo | 51.4 | 70.3 | stable but weak |
| v70b replay128 | round4 final | 47.2 | 68.9 | harmful retention drop |

## Failure Explanation

- v70b final drops `4.2` heldout EM points from its own round2/round3 level (`51.4 -> 47.2`). This mirrors the formal test-surface regression previously observed for v70b (`50.3421/50.3158 -> 46.9605`).
- v69 replay64 does not show this retention failure on heldout (`53.4 -> 55.2`), consistent with v69 remaining the best formal result.
- v70b final heldout predictions collapse toward extremes: `very negative=224`, `very positive=191`, while `negative=11`, `positive=2`; per-label `positive` accuracy is `0.0` and `negative` accuracy is `7.5472`. This explains why replay distribution/retention changes harmed amazon/SC.

## Gate Decision

- Evidence is strong enough to explain v70b/v71/v72-style failures and to justify adding a train-heldout retention gate as a controller.
- Evidence is not sufficient to launch a new v75 smoke/formal improvement, because the gate rejects harmful candidates but does not yet introduce a module that improves over v69.
- Current Standard best remains v69 formal: EM `77.2566`, ROUGE-L `81.1919`.

## Next Step

- Implement the train-heldout gate in the O-LoRA overlay launcher as a promotion/early-stop controller for future candidates.
- The next actual improvement candidate should be a small retention regularizer or adapter-selection policy that is selected by this gate, not another prompt calibration/replay distribution rewrite/lexical repair/plain NLL scoring path.
