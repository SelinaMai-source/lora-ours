# Official Method Repository Archive

Branch: `official-method-code-archive-20260705`

This archive records only official repositories or explicitly documented official-equivalent anchors. The code is represented as pinned git submodules so the main repository stores official source provenance and commit hashes without copying third-party source trees into this repository history.

To materialize the official source trees:

```bash
git submodule update --init --recursive official_repos
```

If network access is needed in this environment, use the temporary mihomo proxy already documented in `results/logs/official_proxy_status_20260705.md`. Do not commit proxy configs or node details.

## CITB Continual Instruction Tuning

| Method / Base | Official URL | Submodule Path | Pinned Commit | Downloaded | Runnable / Smoke | Gate Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CITB official code | `https://github.com/hyintell/CITB.git` | `official_repos/citb/CITB` | `bf50533b5bced4c388691ecc75e26773da96b3fd` | yes, local official checkout exists | script-strict FT-init run exists under public `500/50/50` path | blocked | Paper says `500/50/100`; public script/data path behaves as `500/50/50` with short-task train-count effects. No Ours increment until comparable official setting is resolved. |

Method coverage in this official repo: FT-init, L2, EWC, AGEM, Replay, and Multi upper bound. Replay/AGEM memory variants 10/50 and three-seed paper setting still require official-comparable smoke/reproduction after the split blocker is resolved.

## Standard T5-Large PEFT CL

| Method / Base | Official URL | Submodule Path | Pinned Commit | Downloaded | Runnable / Smoke | Gate Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O-LoRA | `https://github.com/cmnfriend/O-LoRA.git` | `official_repos/standard/O-LoRA` | `07117e1fc4a5f5ad9308a815a42cee8f46502dc8` | yes | v55 smoke and v57 official-equivalent formal completed | official_equivalent_anchor | Local v57 uses documented single-GPU engineering substitutions; official order/metric/data path preserved. |
| LFPT5 | `https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning.git` | `official_repos/standard/Lifelong-Fewshot-Language-Learning` | `cf7d17ce7de6a707d929d0542b3d5e639569855f` | yes | not rerun locally | downloaded_not_runnable | Reference baseline; must be separately smoke/reproduced before using as an Ours base. |
| Progressive Prompts | `https://github.com/arazd/ProgressivePrompts.git` | `official_repos/standard/ProgressivePrompts` | `01572d6a73c0576b070ceee00dbe4f5bc278423f` | yes | not rerun locally | downloaded_not_runnable | Task-ID/prompt protocol differs; must be separately smoke/reproduced before using as an Ours base. |
| LB-CL | not confirmed | none | none | no official repo found | no code smoke possible | paper_only_baseline | Web/OpenReview/NeurIPS/author searches have not confirmed an official or author repository. Do not add third-party code or label it official. |

Current Standard anchor: O-LoRA official-equivalent v57; current Ours overlay v69 is anchored on O-LoRA and remains the best local Standard result.

Method coverage tracked for Standard: SeqLoRA, IncLoRA, Replay, LFPT5, ProgressivePrompts, O-LoRA, LB-CL, and MTL. O-LoRA/LFPT5/ProgressivePrompts have official repos archived above; LB-CL has no confirmed official repo; SeqLoRA/IncLoRA/Replay/MTL require exact official script/code-path verification before they can be treated as reproduced methods.

## Dialogue NLG / MultiWOZ

| Method / Base | Official URL | Submodule Path | Pinned Commit | Downloaded | Runnable / Smoke | Gate Status | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARPER / SCLSTM | `https://github.com/MiFei/Continual-Learning-for-NLG.git` | `official_repos/dialogue/Continual-Learning-for-NLG` | `99019defe6bf35e8459ca6abd6f25882724bc956` | yes | v58 smoke and v66 official SCLSTM formal completed | official_equivalent_anchor | ARPER WOZ3 BLEU/SER must not be conflated with ToDCL 37-domain BLEU/EER. |
| ToDCL | `https://github.com/andreamad8/ToDCL.git` | `official_repos/dialogue/ToDCL` | `e70c1edf937f6eb570296ea2897dbc8d6815bc6d` | yes | `train.py --help` passes in isolated smoke venv | downloaded_not_runnable | Data layout/download and faithful legacy training env remain unresolved; README pins `torch==1.4.0` and `transformers==3.5.1`. |

Method coverage tracked for Dialogue: ARPER, Replay, AdapterCL, LAMOL, and Multi upper bound. ARPER/SCLSTM is the current runnable official anchor; ToDCL contains LAMOL, REPLAY, ADAPTER/AdapterCL, VANILLA/L2/EWC/AGEM, and MULTI protocols, but only entrypoint help smoke has passed so far.

## Notes

- These submodules intentionally point to official upstream URLs.
- The pinned commits match local official checkouts already audited under `/root/autodl-tmp`.
- LB-CL is omitted from submodules until an official/author repository or supplement code is confirmed.
- Do not launch new Ours candidates from this archive alone; method-specific official reproduction gates still apply.
