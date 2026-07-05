# Official Reproducibility Workflow

Updated: 2026-07-05

This is the only allowed workflow for CITB, Standard PEFT CL, and Dialogue NLG experiments.

## Required Order

1. Search published papers for the benchmark and list the strongest reported methods and numbers.
2. For every method that will be compared, reproduced, or used as a base, download the official repository when available and record:
   - repo URL
   - commit hash
   - local path
   - environment
   - data
   - checkpoint/model
   - official scripts
   - paper setting
   - locally runnable setting
   - smoke result
   - reproduction result
   - blockers
3. Verify the official code path before adding Ours:
   - run a no-training/preflight check first;
   - run a cheap smoke if possible;
   - only then run enough of the official method to anchor or reproduce the paper result.
4. Add RP(LoRA)/Ours modules only after the best available published method has an official reproduction or explicitly documented official-equivalent anchor.
5. Ours must be an explainable incremental module on the official base, not a separate weak base.
6. Keep settings comparable:
   - no non-official task order, split, seed, metric, backbone, checkpoint, or hidden hyperparameter changes in a SOTA row;
   - no test target/confusion leakage for rule design;
   - no third-party implementation can be labeled as official code.

## Gate Labels

- `official_reproduction`: literal official repo/script/setting, or only documented environmental substitutions.
- `official_equivalent_anchor`: official entry/data/order/metric preserved, with documented engineering changes needed for local hardware.
- `paper_only_baseline`: paper result is known, but official code is missing or not runnable locally.
- `downloaded_not_runnable`: official repo exists locally, but environment/data/checkpoints are incomplete.
- `blocked`: public artifacts do not currently support a comparable run.
- `ours_overlay`: Ours module layered on an official reproduction/equivalent anchor.
- `adapted_variant`: useful diagnostic, but not comparable to the official method row.

## Current Suite Gates

- CITB: blocked for paper-comparable `500/50/100`; script-strict `500/50/50` official run exists.
- Standard PEFT CL: O-LoRA official-equivalent anchor exists; LB-CL remains paper-only because no official/author code is found.
- Dialogue NLG: ARPER official SCLSTM anchor exists; ToDCL repo and faithful legacy env are available, SGD/Taskmaster/MultiWOZ source archives are downloaded and laid out, MultiWOZ conversion completed, full 37-domain data-loader smoke passes, and bounded VANILLA/ADAPTER/REPLAY NLG method smokes pass. ToDCL is still not an official reproduction because official-scale strongest-method reproduction/paper-number alignment is pending.

## Network And Proxy Gate

- Optional proxy config location: `/root/autodl-tmp/Lora-code/configs/clash`.
- Do not print or commit proxy node/subscription contents.
- Verified runtime on 2026-07-05:
  - binary: `/usr/local/bin/mihomo`
  - usable config: `/root/autodl-tmp/Lora-code/configs/clash/runtime/d18255a-GS.no_geoip.yaml`
  - tmux session: `official-gate-mihomo-proxy`
  - local ports: HTTP `127.0.0.1:7890`, SOCKS `127.0.0.1:7891`, redir `127.0.0.1:7892`
- Use temporary environment variables only when needed:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7891
```

- Current verification:
  - GitHub and HuggingFace are reachable through the proxy.
  - OpenReview still redirects to a browser verification page.
  - GitHub unauthenticated API repository search is reachable but rate-limited (`403`), so do not treat API search failure as proof that code does not exist.
  - ToDCL upstream data repos are reachable by `git ls-remote`.
  - ToDCL SGD, Taskmaster, and MultiWOZ source archives were downloaded to `/root/autodl-tmp/todcl_official_data_20260705/archives` and passed Python `zipfile.testzip()` validation.
  - ToDCL full data layout and full 37-domain data-loader smoke now pass under `/root/autodl-tmp/conda_envs/todcl_legacy_py37`.
  - ToDCL bounded VANILLA/ADAPTER/REPLAY NLG method smokes now pass, including checkpoint reload, model/tokenizer save, bounded final generation, and scorer entry.
  - Large official data files must continue to live under `/root/autodl-tmp`, not `/root`.

## Launch Rule

No new Ours candidate may start unless the corresponding official base row in `docs/official_alignment/official_method_matrix.md` has `gate = official_reproduction` or `gate = official_equivalent_anchor`.

If a new paper appears to outperform the current base, first add it to the matrix, find/download official code, and run the official gate. Do not continue optimizing Ours against an outdated or weak base while the stronger method is unaudited.
