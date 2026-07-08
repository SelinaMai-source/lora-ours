#!/usr/bin/env python3
"""Train wrapper: ARPER v89 base + SSRG exemplar selection overlay."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARPER_ROOT = REPO / "baselines/advanced_baselines/arper_dialog_nlg/external"
CONFIG = os.environ.get("ARPER_OVERLAY_CONFIG", "")
TOP_K = int(os.environ.get("SSRG_SPECTRAL_TOP_K", "8"))
ENERGY = float(os.environ.get("SSRG_ENERGY_THRESHOLD", "0.85"))

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ARPER_ROOT))

from scripts.arper_ssrg_exemplar_selection import construct_exemplar_indices_ssrg
import construct_exemplar as ce

_orig_construct = ce.construct_exemplars


def _ssrg_indices(x, m):
    import numpy as np

    arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    return construct_exemplar_indices_ssrg(arr, m, spectral_top_k=TOP_K, energy_threshold=ENERGY)


ce.construct_exemplar_indices_ssrg = _ssrg_indices


def construct_exemplars_ssrg(model, task, exemplar_size, config, sv_len_weight, return_selected_ids=False, generation=False):
    """SSRG overlay: extract herding features, select via spectral subspace."""
    import copy
    import math
    import torch

    experiment_type = config["EXPERIMENT"]["experiment"]
    exemplar_selection = config["EXPERIMENT"]["exemplar_selection"].split(",")
    batch_size = config.getint("DATA", "batch_size")
    exemplar_type = ""
    for scheme in exemplar_selection:
        if scheme in experiment_type:
            exemplar_type = scheme

    if exemplar_type != "ssrg":
        return _orig_construct(
            model, task, exemplar_size, config, sv_len_weight,
            return_selected_ids=return_selected_ids, generation=generation,
        )

    model.eval()
    with torch.no_grad():
        task.reset()
        feature_map = {"train": [], "valid": []}
        for dtype in ["train", "valid"]:
            for _ in range(task.n_batch[dtype]):
                input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes, _, do_label, da_label, sv_label = task.next_batch(dtype)
                if model.model_type == "lm":
                    _, last_hidden = model(input_var, task, feats_var, keep_last=False)
                    norm = feats_var.norm(p=1, dim=1, keepdim=True)
                    feature_map[dtype].append(feats_var / norm)
                else:
                    target_var = input_var.clone()
                    model.set_prior(False)
                    _, last_hidden = model(
                        input_var, input_lengths=lengths, target_seq=target_var,
                        target_lengths=lengths, conds_seq=feats_var, dataset=task,
                    )
                    feature_map[dtype].append(last_hidden)
            feature_map[dtype] = torch.cat(feature_map[dtype])

        m = {
            "train": math.ceil(exemplar_size["train"] / batch_size) * batch_size,
            "valid": math.ceil(exemplar_size["valid"] / batch_size) * batch_size,
        }
        exemplars = {}
        selected_ids = {}
        from loader.task import Exemplars

        for dtype in ["train", "valid"]:
            dtype_selected_ids = _ssrg_indices(feature_map[dtype], m[dtype])
            del feature_map[dtype]
            if not generation:
                dtype_exemplar_data = [task.data[dtype][i] for i in dtype_selected_ids]
            else:
                dtype_exemplar_data = []
            exemplar_data = {dtype: dtype_exemplar_data}
            exemplar_data[dtype] = exemplar_data[dtype][: exemplar_size[dtype]]
            exemplars[dtype] = Exemplars(exemplar_data[dtype], None)
            selected_ids[dtype] = dtype_selected_ids[: exemplar_size[dtype]]

    if return_selected_ids:
        return exemplars, selected_ids
    return exemplars


ce.construct_exemplars = construct_exemplars_ssrg

os.chdir(str(ARPER_ROOT))
if not CONFIG:
    raise SystemExit("ARPER_OVERLAY_CONFIG required")

sys.argv = [
    "run_woz3.py", "--mode", "train", "--random_seed", "1111",
    "--sv_len_weight", "0.5", "--adaptive", "True", "--ewc_importance", "300000",
    "--lr", "0.005", "--dropout", "0", "--_lambda", "2.0",
    "--config_file", CONFIG,
]
import run_woz3  # noqa: F401
