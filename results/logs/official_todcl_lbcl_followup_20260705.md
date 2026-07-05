# Official Gate Follow-up: ToDCL Smoke And LB-CL Search - 2026-07-05

Scope: official reproducibility gate only. No new Ours candidate or GPU training job was launched.

## Proxy State

- Runtime: `/usr/local/bin/mihomo`
- Session: `official-gate-mihomo-proxy`
- Config used: `/root/autodl-tmp/Lora-code/configs/clash/runtime/d18255a-GS.no_geoip.yaml`
- Ports listening: `7890`, `7891`, `7892`
- Temporary environment used:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7891
```

## ToDCL Official Smoke

- Official repo: `https://github.com/andreamad8/ToDCL`
- Local path: `/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl`
- Commit: `e70c1edf937f6eb570296ea2897dbc8d6815bc6d`
- README data setup: `pip install -r requirements.txt`, then `cd data && bash download.sh`.
- `data/download.sh` clones:
  - `google-research-datasets/dstc8-schema-guided-dialogue`
  - `google-research-datasets/Taskmaster`
  - `budzianowski/multiwoz`
  - then unzips/converts MultiWOZ 2.1/2.2.
- Upstream data repos are reachable with `git ls-remote` through the proxy.
- Local ToDCL checkout footprint: about `552M`.
- Isolated help-smoke venv: `/root/autodl-tmp/venvs/todcl_official_smoke_py39`, about `37M`.
- The smoke venv inherits existing Python 3.9 system-site packages from the O-LoRA environment and installs minimal missing packages inside the venv only:
  - `pytorch-lightning==1.2.5`
  - `torchmetrics==0.3.2`
  - `dictdiffer==0.8.1`
  - `termcolor==1.1.0`
  - `tabulate==0.8.9`
  - `sentencepiece==0.1.99`
  - `sacremoses==0.0.45`
- Result: `/root/autodl-tmp/venvs/todcl_official_smoke_py39/bin/python train.py --help` exits `0` and prints the official argument parser.

### Remaining ToDCL Blockers

- This is only an entrypoint smoke, not a data-loader or training smoke.
- Data has not been downloaded, to avoid uncontrolled large downloads.
- A fully faithful legacy runtime remains unresolved because the README pins `torch==1.4.0`, `transformers==3.5.1`, and `pytorch-lightning==1.2.5`, while the smoke venv inherits newer local torch/transformers.
- Next safe step: controlled data download/layout into `/root/autodl-tmp`, followed by a cheap data-loader smoke before any training smoke.

## LB-CL Official Code Search

- Method: Learn More, but Bother Less: Parameter Efficient Continual Learning.
- Published numbers: T5-large Standard CL order1/2/3 `76.9/76.5/76.8`, avg `76.7`.
- Proxy-enabled network status:
  - GitHub and HuggingFace are reachable.
  - OpenReview/NeurIPS/ML Anthology direct local fetch is unreliable due SSL/browser-access issues, but web-indexed snippets and PDF text are accessible.
  - GitHub unauthenticated API repository search is rate-limited (`403`).
- Searches tried:
  - `"Learn more, but bother less" LB-CL code GitHub`
  - `"LB-CL" "Learn more, but bother less" GitHub`
  - `ZxtaNh5UYB`
  - `"Fuli Qiao" "Learn More, but Bother Less" GitHub`
  - `"Mehrdad Mahdavi" "LB-CL" GitHub`
  - `"Learn more, but bother less" "github.com"`
- Result: no official/author LB-CL repository or supplement code confirmed.

## Gate Impact

- ToDCL moves from `import blocker` to `entrypoint smoke passed`, but remains `downloaded_not_runnable` for any official training/reproduction.
- LB-CL remains `paper_only_baseline`; do not use third-party code as official.
- No Ours candidate should be launched from these updates alone.
