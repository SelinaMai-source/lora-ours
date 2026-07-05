#!/usr/bin/env python3
"""Text-spectral replay selection for O-LoRA overlay (SSRG proxy).

Mirrors core.methods.ours_spectral_replay spectral gating without a live model
forward pass: build a TF-IDF covariance matrix, truncated SVD, and keep exemplars
with highest projection energy onto the top spectral subspace.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable, Sequence


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def _instance_text(instance: dict[str, Any]) -> str:
    sentence = str(instance.get("sentence", instance.get("text", "")))
    label = str(instance.get("label", ""))
    return f"{sentence} {label}".strip()


def _build_vocab(docs: Sequence[str], max_features: int = 4096) -> dict[str, int]:
    freq: dict[str, int] = {}
    for doc in docs:
        for tok in set(_tokenize(doc)):
            freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    vocab = {tok: idx for idx, (tok, _) in enumerate(ranked[:max_features])}
    return vocab


def _tfidf_matrix(docs: Sequence[str], vocab: dict[str, int]) -> list[list[float]]:
    if not docs or not vocab:
        return []
    n = len(docs)
    rows: list[list[float]] = []
    for doc in docs:
        counts: dict[str, int] = {}
        for tok in _tokenize(doc):
            if tok in vocab:
                counts[tok] = counts.get(tok, 0) + 1
        row = [0.0] * len(vocab)
        if not counts:
            rows.append(row)
            continue
        max_tf = max(counts.values())
        for tok, count in counts.items():
            tf = 0.5 + 0.5 * (count / max_tf)
            row[vocab[tok]] = tf
        rows.append(row)
    return rows


def _mean_center(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return matrix
    dim = len(matrix[0])
    mean = [0.0] * dim
    for row in matrix:
        for idx, value in enumerate(row):
            mean[idx] += value
    mean = [value / len(matrix) for value in mean]
    return [[row[idx] - mean[idx] for idx in range(dim)] for row in matrix]


def _covariance(matrix: list[list[float]]) -> list[list[float]]:
    if len(matrix) < 2:
        return []
    dim = len(matrix[0])
    cov = [[0.0] * dim for _ in range(dim)]
    denom = max(1, len(matrix) - 1)
    for row in matrix:
        for i in range(dim):
            for j in range(dim):
                cov[i][j] += row[i] * row[j] / denom
    return cov


def _power_iteration_scores(
    matrix: list[list[float]],
    *,
    top_k: int,
    energy_threshold: float,
) -> list[float]:
    """Approximate top spectral scores without torch."""
    if not matrix:
        return []
    cov = _covariance(matrix)
    if not cov:
        return [0.0] * len(matrix)
    dim = len(cov)
    rank = max(1, min(top_k, dim))
    basis: list[list[float]] = []
    work = [row[:] for row in cov]
    singular_values: list[float] = []
    for _ in range(rank):
        vec = [1.0 / math.sqrt(dim)] * dim
        for _ in range(16):
            nxt = [0.0] * dim
            for i in range(dim):
                for j in range(dim):
                    nxt[i] += work[i][j] * vec[j]
            norm = math.sqrt(sum(value * value for value in nxt)) or 1.0
            vec = [value / norm for value in nxt]
        singular = math.sqrt(max(0.0, sum(vec[i] * sum(work[i][j] * vec[j] for j in range(dim)) for i in range(dim))))
        singular_values.append(singular)
        basis.append(vec)
        for i in range(dim):
            for j in range(dim):
                work[i][j] -= singular * vec[i] * vec[j]
    total_energy = sum(value * value for value in singular_values) or 1.0
    cumulative = 0.0
    keep = rank
    for idx, value in enumerate(singular_values, start=1):
        cumulative += value * value
        if cumulative / total_energy >= energy_threshold:
            keep = idx
            break
    basis = basis[:keep]
    scores = []
    for row in matrix:
        energy = 0.0
        for vec in basis:
            proj = sum(row[i] * vec[i] for i in range(dim))
            energy += proj * proj
        scores.append(math.sqrt(max(0.0, energy)))
    return scores


def select_instances(
    instances: Sequence[dict[str, Any]],
    *,
    max_num_instances: int,
    spectral_top_k: int = 8,
    energy_threshold: float = 0.85,
    seed: int = 1,
) -> list[dict[str, Any]]:
    if max_num_instances is None or max_num_instances < 0 or len(instances) <= max_num_instances:
        return list(instances)
    docs = [_instance_text(item) for item in instances]
    vocab = _build_vocab(docs)
    matrix = _mean_center(_tfidf_matrix(docs, vocab))
    if len(matrix) < 2:
        return list(instances[:max_num_instances])
    scores = _power_iteration_scores(
        matrix,
        top_k=spectral_top_k,
        energy_threshold=energy_threshold,
    )
    ranked = sorted(
        enumerate(instances),
        key=lambda item: (-scores[item[0]], _stable_key(item[1], seed)),
    )
    return [instance for _, instance in ranked[:max_num_instances]]


def _stable_key(instance: dict[str, Any], seed: int) -> str:
    payload = _instance_text(instance)
    digest = hashlib.sha1(f"{seed}:{payload}".encode("utf-8")).hexdigest()
    return digest


def select_instances_from_iterable(
    instances: Iterable[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return select_instances(list(instances), **kwargs)
