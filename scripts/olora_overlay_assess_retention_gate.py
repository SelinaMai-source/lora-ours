#!/usr/bin/env python3
"""Compatibility wrapper for the Standard assess-retention gate."""

from ours_v1.suites.standard.olora_overlay_assess_retention_gate import *  # noqa: F401,F403
from ours_v1.suites.standard.olora_overlay_assess_retention_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
