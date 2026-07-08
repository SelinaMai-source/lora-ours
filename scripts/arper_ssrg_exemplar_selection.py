"""SSRG exemplar selection for ARPER SCLSTM Path B overlay."""

from __future__ import annotations

import numpy as np


def construct_exemplar_indices_ssrg(
    features: np.ndarray,
    m: int,
    *,
    spectral_top_k: int = 8,
    energy_threshold: float = 0.85,
) -> list[int]:
    """Select m exemplar indices with highest spectral projection energy."""
    if m <= 0:
        return []
    n = len(features)
    if n <= m:
        return list(range(n))

    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    mean = arr.mean(axis=0, keepdims=True)
    centered = arr - mean
    if centered.shape[0] < 2:
        return np.random.choice(n, m, replace=False).tolist()

    try:
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.random.choice(n, m, replace=False).tolist()

    rank = max(1, min(spectral_top_k, len(s)))
    total_energy = float((s ** 2).sum()) or 1.0
    cumulative = 0.0
    keep = rank
    for idx, val in enumerate(s[:rank], start=1):
        cumulative += float(val ** 2)
        if cumulative / total_energy >= energy_threshold:
            keep = idx
            break

    basis = u[:, :keep]
    proj = centered @ basis
    scores = np.sqrt((proj ** 2).sum(axis=1))
    ranked = np.argsort(-scores, kind="stable")
    return ranked[:m].tolist()
