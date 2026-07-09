#!/usr/bin/env python3
"""Compatibility wrapper for the ARPER SSRG train overlay."""

from __future__ import annotations

import runpy
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "ours_v1/suites/arper/arper_ssrg_overlay_train_wrapper.py"
runpy.run_path(str(TARGET), run_name="__main__")
