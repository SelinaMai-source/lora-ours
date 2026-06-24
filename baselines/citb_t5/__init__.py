"""CPU-safe helpers for CITB T5 strict baseline preflight.

The modules in this package intentionally do not start training. They codify
the official CITB data, replay, command, and metric protocols so the strict
runner can distinguish reproducible resources from remaining blockers.
"""

