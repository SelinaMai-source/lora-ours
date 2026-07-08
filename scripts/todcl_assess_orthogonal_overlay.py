#!/usr/bin/env python3
"""ToDCL ADAPTER overlay: Assess-then-Update orthogonal penalty on adapter weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def flatten_adapter(adapter_state: dict[str, torch.Tensor]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for key in sorted(adapter_state):
        if "adapter" in key.lower() or "lora" in key.lower():
            chunks.append(adapter_state[key].float().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0)


def assess_isolated_energy(prior: torch.Tensor, current: torch.Tensor) -> float:
    if prior.numel() == 0 or current.numel() == 0 or prior.shape != current.shape:
        return 0.0
    delta = current - prior
    total = float(delta.norm().item() ** 2)
    if total <= 1e-12:
        return 0.0
    prior_norm = prior.norm().item()
    if prior_norm < 1e-8:
        return 1.0
    coeff = float(torch.dot(prior, delta).item() / (prior_norm ** 2))
    residual = delta - prior * coeff
    return float(residual.norm().item() ** 2 / total)


def orthogonal_penalty(prior: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    if prior.numel() == 0 or current.numel() == 0 or prior.shape != current.shape:
        return torch.tensor(0.0)
    delta = current - prior
    prior_norm = prior.norm()
    if float(prior_norm) < 1e-8:
        return delta.norm() ** 2
    proj = torch.dot(prior, delta) / (prior_norm ** 2) * prior
    residual = delta - proj
    return residual.norm() ** 2


def install_training_hook(model, prior_state: dict, *, threshold: float, penalty_weight: float) -> None:
    prior_vec = flatten_adapter(prior_state)
    original_training_step = model.training_step

    def training_step(batch, batch_idx):
        loss = original_training_step(batch, batch_idx)
        if prior_vec.numel() == 0 or penalty_weight <= 0:
            return loss
        current_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        current_vec = flatten_adapter(current_state)
        iso = assess_isolated_energy(prior_vec, current_vec)
        if iso > threshold:
            pen = orthogonal_penalty(prior_vec, current_vec)
            loss = loss + penalty_weight * pen
        return loss

    model.training_step = training_step


def main() -> int:
    todcl_root = Path(os.environ.get("TODCL_ROOT", "/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl"))
    prior_ckpt = os.environ.get("OURS_PRIOR_CKPT", "")
    threshold = float(os.environ.get("OURS_ASSESS_THRESHOLD", "0.25"))
    penalty_weight = float(os.environ.get("OURS_ORTHOGONAL_PENALTY", "0.1"))

    if not prior_ckpt or not Path(prior_ckpt).exists():
        print("OURS_PRIOR_CKPT missing; overlay blocked", file=sys.stderr)
        return 77

    sys.path.insert(0, str(todcl_root))
    os.chdir(todcl_root)

    prior_state = torch.load(prior_ckpt, map_location="cpu")
    if not isinstance(prior_state, dict):
        raise TypeError("prior checkpoint must be state dict")

    import CL_learner as cl

    _orig_init = cl.Seq2SeqToD.__init__

    def _init_with_overlay(self, hparams):
        _orig_init(self, hparams)
        install_training_hook(self, prior_state, threshold=threshold, penalty_weight=penalty_weight)

    cl.Seq2SeqToD.__init__ = _init_with_overlay

    if len(sys.argv) <= 1:
        sys.argv = [
            "train.py", "--task_type", "NLG", "--CL", "ADAPTER",
            "--bottleneck_size", "50", "--lr", "6.25e-3",
            "--n_epochs", os.environ.get("N_EPOCHS", "10"),
            "--train_batch_size", "16", "--gradient_accumulation_steps", "5",
            "--dataset_list", "SGD,TM19,TM20,MWOZ", "--setting", "single", "--seed", "1",
            "--model_checkpoint", str(todcl_root / "gpt2"),
        ]

    import train  # noqa: F401
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
