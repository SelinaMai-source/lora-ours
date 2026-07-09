#!/usr/bin/env python3
"""Compatibility wrapper for the ToDCL assess orthogonal overlay."""

from ours_v1.suites.todcl.todcl_assess_orthogonal_overlay import *  # noqa: F401,F403
from ours_v1.suites.todcl.todcl_assess_orthogonal_overlay import main


if __name__ == "__main__":
    raise SystemExit(main())
