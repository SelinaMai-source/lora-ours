#!/usr/bin/env python3
"""Probe ARPER SCLSTM peak VRAM for candidate batch sizes (1 forward+backward)."""
import argparse
import configparser
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARPER_ROOT = REPO / "baselines/advanced_baselines/arper_dialog_nlg/external"
sys.path.insert(0, str(ARPER_ROOT))

import torch
import numpy as np

from loader.dataset_woz3 import DatasetWoz3
from loader.task import generate_task
from model.lm_deep import LM_deep


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--batch-sizes", default="128,256,512,1024,2048,4096")
    p.add_argument("--vram-cap-mib", type=int, default=int(49140 * 0.80))
    p.add_argument("--output", default="")
    return p.parse_args()


def load_config(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def make_args():
    class A:
        random_seed = 1111
        sv_len_weight = 0.5
        adaptive = True
        ewc_importance = 300000
        lr = 0.005
        dropout = 0
        _lambda = 2.0
        l2_weight = 0
    return A()


def probe_batch(base_config_path, args, batch_size, vram_cap_mib):
    config_path = Path(base_config_path)
    if not config_path.is_absolute():
        config_path = (REPO / config_path).resolve()
    prev_cwd = os.getcwd()
    os.chdir(ARPER_ROOT)
    try:
        config = configparser.ConfigParser()
        config.read(str(config_path))
        config.set("DATA", "batch_size", str(batch_size))

        dataset = DatasetWoz3(config, config["DATA"]["data_split"], percentage=1.0)
        task_seq = [int(x) for x in config.get("DATA", "task_seq").split(",")]
        task, task_name, _task_voc = generate_task(dataset, [task_seq[0]], None)

        vocab_size = len(dataset.word2index)
        feat_size = dataset.do_size + dataset.da_size + dataset.sv_size
        hidden_size = config.getint("MODEL", "hidden_size")
        n_layer = config.getint("MODEL", "num_layer")
        dec_type = config.get("MODEL", "dec_type")

        model = LM_deep(dec_type, args, vocab_size, vocab_size, hidden_size, feat_size, n_layer=n_layer, dropout=0)
        if torch.cuda.is_available():
            model.cuda()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        task.reset_merged()
        input_var, label_var, feats_var, lengths, *_rest = task.next_merged_batch("train")
        if torch.cuda.is_available():
            input_var = input_var.cuda()
            label_var = label_var.cuda()
            feats_var = feats_var.cuda()

        model.zero_grad(set_to_none=True)
        _ = model(input_var, task, feats_var)
        loss = model.get_loss(label_var, lengths)
        loss.backward()

        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        ok = peak_mib <= vram_cap_mib
        row = {
            "batch_size": batch_size,
            "peak_mib": round(peak_mib, 1),
            "ok_under_cap": ok,
            "actual_batch": int(input_var.size(0)),
        }
        del model, task, dataset, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return row
    finally:
        os.chdir(prev_cwd)


def main():
    args = parse_args()
    config = load_config(args.config)
    train_args = make_args()
    sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    vram_cap = args.vram_cap_mib
    results = []
    max_safe = None

    print(f"VRAM cap (80%): {vram_cap} MiB", flush=True)
    for bs in sizes:
        try:
            row = probe_batch(args.config, train_args, bs, vram_cap)
            results.append(row)
            print(f"bs={row['batch_size']:5d} peak={row['peak_mib']:8.1f} MiB ok={row['ok_under_cap']}", flush=True)
            if row["ok_under_cap"]:
                max_safe = row["batch_size"]
            else:
                break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                results.append({"batch_size": bs, "peak_mib": None, "ok_under_cap": False, "error": "OOM"})
                print(f"bs={bs:5d} OOM", flush=True)
                torch.cuda.empty_cache()
                break
            raise

    payload = {
        "config": args.config,
        "vram_cap_mib": vram_cap,
        "results": results,
        "max_safe_batch": max_safe,
    }
    out = args.output or str(REPO / "results/logs/arper_batch_memory_probe.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
