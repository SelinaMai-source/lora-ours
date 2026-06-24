#!/usr/bin/env python3
"""Smoke-check LFPT5 protocol-token decoding without loading the T5 model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
SUM_DIR = REPO / "external_baselines/lfpt5/Summarization"
if str(SUM_DIR) not in sys.path:
    sys.path.insert(0, str(SUM_DIR))

from model import T5forSummarization  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LFPT5 decode-time protocol token stripping")
    parser.add_argument("--out", type=Path, default=Path("results/tables/lfpt5_decoding_protocol_smoke.json"))
    args = parser.parse_args()

    probe = object.__new__(T5forSummarization)
    probe.args = SimpleNamespace()
    probe.blocked_generation_token_ids = [32100, 32101, 32102]
    probe.answer_token_text = "__ans__"
    probe.task_token_pattern = re.compile(r"\bcitbtask\d+\b")

    samples = [
        "citbtask0 __ans__ positive",
        "citbtask37 answer __ans__ with  extra   spaces",
        "plain answer",
    ]
    cleaned = [probe._strip_protocol_tokens(sample) for sample in samples]
    generation_kwargs = probe._blocked_generation_kwargs()
    result = {
        "status": "pass",
        "strict_allowed": False,
        "reason": "LFPT5 protocol repair smoke only; a full strict rerun still requires paper hparam/protocol audit.",
        "samples": samples,
        "cleaned": cleaned,
        "generation_kwargs": generation_kwargs,
        "checks": {
            "task_tokens_removed": all("citbtask" not in text for text in cleaned),
            "answer_token_removed": all("__ans__" not in text for text in cleaned),
            "bad_words_ids_ready": generation_kwargs == {"bad_words_ids": [[32100], [32101], [32102]]},
        },
    }
    if not all(result["checks"].values()):
        result["status"] = "failed"

    out = args.out if args.out.is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
