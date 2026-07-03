"""Runtime shims for local CITB official reproduction launches.

This module is loaded automatically by Python when its directory is present on
PYTHONPATH. Keep shims narrow: they should only bridge local runtime/cache gaps
without changing CITB training or evaluation semantics.
"""

from __future__ import annotations

import os
from pathlib import Path


def _patch_gpt2_tokenizer() -> None:
    try:
        from transformers import AutoTokenizer
    except Exception:
        return

    original = AutoTokenizer.from_pretrained
    local_gpt2 = os.environ.get("CITB_GPT2_TOKENIZER_NAME")

    if not local_gpt2:
        return

    def from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path == "gpt2":
            local_path = Path(local_gpt2)
            if not local_path.exists():
                raise FileNotFoundError(
                    "CITB_GPT2_TOKENIZER_NAME points to a missing local GPT-2 tokenizer: "
                    f"{local_path}"
                )
            return original(str(local_path), *args, **kwargs)
        return original(pretrained_model_name_or_path, *args, **kwargs)

    AutoTokenizer.from_pretrained = from_pretrained


_patch_gpt2_tokenizer()
