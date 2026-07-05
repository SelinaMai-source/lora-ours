#!/usr/bin/env python3
"""Assess-update inspired retention gate for O-LoRA overlay adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _flatten_effective(adapter: dict[str, torch.Tensor]) -> torch.Tensor:
    by_layer: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in adapter.items():
        if not (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")):
            continue
        layer = key.rsplit(".lora_", 1)[0]
        side = "A" if ".lora_A." in key else "B"
        by_layer.setdefault(layer, {})[side] = tensor.float()
    chunks: list[torch.Tensor] = []
    for layer in sorted(by_layer):
        parts = by_layer[layer]
        if "A" not in parts or "B" not in parts:
            continue
        chunks.append((parts["B"] @ parts["A"]).reshape(-1))
    return torch.cat(chunks, dim=0) if chunks else torch.zeros(0)


def load_adapter_vector(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"expected adapter dict at {path}")
    return _flatten_effective(payload)


def assess_update(prior_path: Path, current_path: Path, *, threshold: float) -> dict:
    prior = load_adapter_vector(prior_path)
    current = load_adapter_vector(current_path)
    if prior.numel() == 0 or current.numel() == 0:
        return {"action": "merge", "isolated_energy_ratio": 0.0, "best_sim": 0.0, "fail_reasons": []}
    if prior.shape != current.shape:
        raise ValueError(f"adapter shape mismatch: prior={prior.shape}, current={current.shape}")
    delta = current - prior
    total_energy = float(delta.norm().item() ** 2)
    if total_energy <= 1e-12:
        return {"action": "merge", "isolated_energy_ratio": 0.0, "best_sim": 1.0, "fail_reasons": []}
    prior_norm = prior.norm().item()
    if prior_norm < 1e-8:
        isolated_ratio = 1.0
        best_sim = 0.0
    else:
        coeff = float(torch.dot(prior, delta).item() / (prior_norm ** 2))
        residual = delta - prior * coeff
        isolated_ratio = float(residual.norm().item() ** 2 / total_energy)
        best_sim = float(torch.dot(prior, current).item() / (prior_norm * current.norm().item() + 1e-8))
    fail_reasons = []
    action = "spawn" if isolated_ratio > threshold else "merge"
    if action == "spawn":
        fail_reasons.append(f"isolated_energy_ratio {isolated_ratio:.4f} > threshold {threshold:.4f}")
    return {
        "action": action,
        "isolated_energy_ratio": isolated_ratio,
        "best_sim": best_sim,
        "fail_reasons": fail_reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-adapter", required=True)
    parser.add_argument("--current-adapter", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--gate-state", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    result = assess_update(
        Path(args.prior_adapter) / "adapter_model.bin",
        Path(args.current_adapter) / "adapter_model.bin",
        threshold=args.threshold,
    )
    gate_state = Path(args.gate_state)
    gate_state.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(gate_state.read_text(encoding="utf-8")) if gate_state.exists() else {"evaluations": [], "decision": "running"}
    record = {
        "round": args.round,
        "task": args.task,
        "prior_adapter": args.prior_adapter,
        "current_adapter": args.current_adapter,
        **result,
    }
    state.setdefault("evaluations", []).append(record)
    state["last_record"] = record
    state["decision"] = "rejected" if result["fail_reasons"] else "running"
    gate_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 42 if result["fail_reasons"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
