# v78 Behavior-Retention Regularizer Diagnostic

- Leakage policy: train-heldout predictions only; no dev/test split or targets.
- Teacher anchor heldout EM: `53.4`
- Teacher prediction counts: `{'very negative': 84, 'negative': 40, 'neutral': 228, 'positive': 13, 'very positive': 135}`

## v69_r4

- Decision: `reject`
- Heldout EM: `55.2`
- Teacher agreement: `64.2`
- Label distribution L1: `0.624`
- Prediction counts: `{'very negative': 128, 'negative': 107, 'neutral': 72, 'positive': 38, 'very positive': 155}`
- Reject reasons: `label distribution L1 0.624 > 0.35 | v76 heldout EM 55.2 does not exceed baseline 55.2`

## v70b_r4

- Decision: `reject`
- Heldout EM: `47.2`
- Teacher agreement: `59.2`
- Label distribution L1: `0.784`
- Prediction counts: `{'very negative': 224, 'negative': 11, 'neutral': 72, 'positive': 2, 'very positive': 191}`
- Reject reasons: `teacher agreement 59.2 < 60.0 | label distribution L1 0.784 > 0.35 | v76 heldout EM 47.2 does not exceed baseline 55.2 | moderate label collapse: negative=11, positive=2`
