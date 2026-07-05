# v77 Train-Heldout Adapter-Selection Candidates

- Leakage policy: uses only v75 train-heldout diagnostics; no dev/test predictions or targets.
- Pass rule: selected heldout EM must exceed v69 heldout baseline and final adapter must not collapse moderate labels.

## v69

- Decision: `reject`
- Selected adapter: `r4` with heldout EM `55.2`
- Final adapter heldout EM: `55.2`
- Final prediction counts: `{'negative': 107, 'very negative': 128, 'very positive': 155, 'neutral': 72, 'positive': 38}`
- Reject reasons: `best heldout EM 55.2 does not exceed v69 baseline 55.2`

## v70b

- Decision: `reject`
- Selected adapter: `r2` with heldout EM `51.4`
- Final adapter heldout EM: `47.2`
- Final prediction counts: `{'negative': 11, 'very negative': 224, 'very positive': 191, 'neutral': 72, 'positive': 2}`
- Reject reasons: `best heldout EM 51.4 does not exceed v69 baseline 55.2 | final heldout EM 47.2 below v69 baseline 55.2 | final moderate-label collapse: negative=11, positive=2`
