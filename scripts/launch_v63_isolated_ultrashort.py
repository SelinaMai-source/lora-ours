#!/usr/bin/env python3
"""Disabled legacy override entrypoint.

The v63 ultrashort diagnostic was created by an external override after v62
formal was selected. Keep this path as an explicit guard so stale automation
cannot silently relaunch capped diagnostic runs.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "Refusing v63 ultrashort diagnostic override; use the v62 formal launch path instead.",
        file=sys.stderr,
    )
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
