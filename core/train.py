from __future__ import annotations

import atexit
import sys
import argparse
import csv
import json
import math
import os
import re
import shutil
import signal
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

# Allow running as: `python core/train.py --config ...` without requiring package installs.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.ccfa_metrics import matrix_columns_from_eval, write_ccfa_postprocess_outputs
from core.data import ContinualStream, Example, Segment, load_continual_stream
from core.evaluate import evaluate_stream
from core.methods.drift_detector import AnchorSet, DriftDetector, DriftEvent, build_anchor_set
from core.methods.lora_bank import LoRABank
from core.methods.overlap_loss import compute_anti_overlap_training_loss, compute_overlap_loss
from core.methods.ours_spectral_replay import SpectralSparseReplayGate
from core.methods.router import Router
from core.models.base_model import build_backbone
from core.models.lora_wrapper import build_lora_wrapper
from core.formatting import format_for_infer
from core.run_artifacts import (
    collect_overfit_stale_artifacts,
    finalize_run_manifest,
    init_overfit_run_manifest,
    init_run_manifest,
    manifest_add_artifact,
)
from core.utils import (
    RunPaths,
    SimpleLogger,
    append_jsonl,
    ensure_dir,
    load_yaml_config,
    make_run_paths,
    save_csv,
    save_json,
    set_seed,
)
from core.wandb_tracker import WandbTracker, parse_tracking_cfg


def _summarize_lora_info(lora: Any) -> Dict[str, Any]:
    info = lora.info() if hasattr(lora, "info") else {}
    if not isinstance(info, dict):
        return {"raw_info": str(info)}
    trainable_names = info.get("trainable_parameter_names", [])
    if not isinstance(trainable_names, list):
        trainable_names = []
    return {
        "enabled": bool(info.get("enabled", False)),
        "r": info.get("r"),
        "alpha": info.get("alpha"),
        "dropout": info.get("dropout"),
        "target_modules": info.get("target_modules", []),
        "active_adapter": info.get("active_adapter"),
        "num_adapters": len(info.get("adapters", {}) if isinstance(info.get("adapters", {}), dict) else {}),
        "frozen_adapters": info.get("frozen_adapters", []),
        "total_parameters": info.get("total_parameters"),
        "trainable_parameters": info.get("trainable_parameters"),
        "num_trainable_parameter_tensors": len(trainable_names),
        "trainable_parameter_name_preview": trainable_names[:8],
    }


class EvalTimeoutError(TimeoutError):
    pass


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _flush_runtime_outputs(tracker: Optional[WandbTracker] = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    if tracker is not None and hasattr(tracker, "flush"):
        try:
            tracker.flush()
        except Exception:
            pass


def _write_runtime_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(str(path.parent))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _diagnostics_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = cfg.get("diagnostics", {}) if isinstance(cfg.get("diagnostics", {}), dict) else {}
    return diagnostics


def _eval_timeout_seconds(cfg: Dict[str, Any]) -> int:
    diagnostics = _diagnostics_cfg(cfg)
    raw = diagnostics.get("per_segment_eval_timeout_seconds", diagnostics.get("eval_timeout_seconds", 0))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _write_process_exit_artifact(
    *,
    run_paths: RunPaths,
    cfg: Dict[str, Any],
    event: str,
    status: str,
    context: Dict[str, Any],
    exit_code: Optional[int] = None,
    error: str = "",
    tb: str = "",
) -> None:
    payload = {
        "updated_at": _now_iso(),
        "event": str(event),
        "status": str(status),
        "pid": int(os.getpid()),
        "ppid": int(os.getppid()),
        "cwd": str(Path.cwd()),
        "run_id": str(run_paths.run_id),
        "config": str(cfg.get("__config_path__", "")),
        "exit_code": exit_code,
        "phase": str(context.get("phase", "")),
        "current_segment_id": context.get("current_segment_id"),
        "current_segment_name": context.get("current_segment_name"),
        "error": str(error),
        "traceback": str(tb),
    }
    for path in [
        Path(run_paths.run_dir) / "process_exit.json",
        Path(run_paths.log_file).with_suffix(".process_exit.json"),
    ]:
        try:
            _write_runtime_json(path, payload)
        except Exception:
            pass


def _install_process_exit_diagnostics(
    *,
    run_paths: RunPaths,
    cfg: Dict[str, Any],
    context: Dict[str, Any],
    tracker_getter: Any,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {"finalized": False}

    def _record(event: str, status: str, exit_code: Optional[int] = None, error: str = "", tb: str = "") -> None:
        _write_process_exit_artifact(
            run_paths=run_paths,
            cfg=cfg,
            event=event,
            status=status,
            context=context,
            exit_code=exit_code,
            error=error,
            tb=tb,
        )
        _flush_runtime_outputs(tracker_getter())

    def _atexit_record() -> None:
        if not state.get("finalized", False):
            _record(event="atexit", status="unknown_atexit")

    def _signal_handler(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name if signum in {int(s) for s in signal.Signals} else str(signum)
        state["signal_event"] = f"signal:{signal_name}"
        state["finalized"] = True
        _record(
            event=str(state["signal_event"]),
            status="signaled",
            exit_code=128 + int(signum),
            error=f"received {signal_name}",
        )
        raise SystemExit(128 + int(signum))

    atexit.register(_atexit_record)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            pass
    return state


def _run_eval_with_diagnostics(
    *,
    cfg: Dict[str, Any],
    run_paths: RunPaths,
    logger: SimpleLogger,
    tracker: Optional[WandbTracker],
    seg: Segment,
    seen_segments: List[Segment],
    model: Any,
    max_new_tokens: int,
    router: Optional[Any],
    lora_bank: Optional[Any],
    normalization_cfg: Dict[str, Any],
    historical_best_per_segment: Dict[int, float],
    historical_best_task_aware_per_segment: Dict[int, float],
) -> Dict[str, Any]:
    seg_dir = Path(ensure_dir(str(Path(run_paths.run_dir) / f"segment_{seg.segment_id:03d}")))
    heartbeat_path = Path(run_paths.run_dir) / "eval_heartbeat.jsonl"
    progress_path = Path(run_paths.run_dir) / "eval_progress.jsonl"
    status_path = seg_dir / "eval_status.json"
    timeout_seconds = _eval_timeout_seconds(cfg)
    started_at = time.time()

    def _status(event: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "updated_at": _now_iso(),
            "event": event,
            "segment_id": int(seg.segment_id),
            "segment_name": str(seg.segment_name),
            "num_seen_segments": int(len(seen_segments)),
            "seen_segment_ids": [int(s.segment_id) for s in seen_segments],
            "total_seen_eval_examples": int(sum(len(s.eval) for s in seen_segments)),
            "timeout_seconds": int(timeout_seconds),
            "elapsed_seconds": float(time.time() - started_at),
        }
        if extra:
            payload.update(extra)
        _write_runtime_json(status_path, payload)
        append_jsonl(str(heartbeat_path), payload)
        return payload

    def _progress(payload: Dict[str, Any]) -> None:
        row = {
            "updated_at": _now_iso(),
            "segment_id": int(seg.segment_id),
            "segment_name": str(seg.segment_name),
            "elapsed_seconds": float(time.time() - started_at),
            **payload,
        }
        append_jsonl(str(progress_path), row)
        _write_runtime_json(status_path, {**row, "timeout_seconds": int(timeout_seconds)})
        _flush_runtime_outputs(tracker)

    _status("eval_start")
    logger.log(
        "Eval start heartbeat: "
        + json.dumps(
            {
                "segment_id": int(seg.segment_id),
                "segment_name": str(seg.segment_name),
                "num_seen_segments": int(len(seen_segments)),
                "timeout_seconds": int(timeout_seconds),
            },
            ensure_ascii=False,
        )
    )
    _flush_runtime_outputs(tracker)

    old_alarm_handler = None
    alarm_enabled = timeout_seconds > 0 and hasattr(signal, "SIGALRM")
    if alarm_enabled:
        old_alarm_handler = signal.getsignal(signal.SIGALRM)

        def _alarm_handler(_signum: int, _frame: Any) -> None:
            raise EvalTimeoutError(
                f"segment {seg.segment_id} eval exceeded {timeout_seconds}s timeout"
            )

        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_seconds)
    try:
        eval_metrics = evaluate_stream(
            model=model,
            segments_seen=seen_segments,
            max_new_tokens=max_new_tokens,
            router=router,
            lora_bank=lora_bank,
            segment_id=seg.segment_id,
            normalization_cfg=normalization_cfg,
            save_debug_examples_dir=str(Path(run_paths.run_dir) / "eval_debug"),
            historical_best_per_segment=historical_best_per_segment,
            historical_best_task_aware_per_segment=historical_best_task_aware_per_segment,
            progress_callback=_progress,
        )
    except BaseException as exc:
        tb = traceback.format_exc()
        event = "eval_timeout" if isinstance(exc, EvalTimeoutError) else "eval_exception"
        payload = _status(
            event,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "traceback": tb,
            },
        )
        _write_runtime_json(seg_dir / f"{event}.json", payload)
        logger.log(f"Eval failed artifact: {seg_dir / f'{event}.json'}")
        _flush_runtime_outputs(tracker)
        raise
    finally:
        if alarm_enabled:
            signal.alarm(0)
            if old_alarm_handler is not None:
                signal.signal(signal.SIGALRM, old_alarm_handler)

    _status("eval_end", {"status": "completed"})
    logger.log(
        "Eval end heartbeat: "
        + json.dumps(
            {
                "segment_id": int(seg.segment_id),
                "segment_name": str(seg.segment_name),
                "elapsed_seconds": float(time.time() - started_at),
            },
            ensure_ascii=False,
        )
    )
    _flush_runtime_outputs(tracker)
    return eval_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified continual instruction tuning pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    cfg["__config_path__"] = str(Path(args.config).resolve())
    mode = str(cfg.get("mode", "")).strip()
    if mode not in {"debug", "baseline", "ours"}:
        raise ValueError(f"Invalid mode={mode}. Expected one of: debug | baseline | ours")

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    results_dir = str(cfg.get("paths", {}).get("results_dir", "results"))
    experiment_name = str(cfg.get("experiment_name", "experiment"))
    run_name = str(cfg.get("output", {}).get("run_name", "")) if isinstance(cfg.get("output", {}), dict) else ""
    run_paths = make_run_paths(results_dir=results_dir, experiment_name=experiment_name, run_name=run_name)
    logger = SimpleLogger(run_paths.log_file)
    tracker = WandbTracker.from_config(
        cfg=cfg,
        run_id=run_paths.run_id,
        config_path=str(cfg.get("__config_path__", "")),
        run_dir=run_paths.run_dir,
    )
    process_context: Dict[str, Any] = {
        "phase": "startup",
        "current_segment_id": None,
        "current_segment_name": None,
    }
    exit_state = _install_process_exit_diagnostics(
        run_paths=run_paths,
        cfg=cfg,
        context=process_context,
        tracker_getter=lambda: tracker,
    )
    tracking_cfg = parse_tracking_cfg(cfg)
    if tracking_cfg["use_wandb"]:
        logger.log(
            f"W&B enabled: project={tracking_cfg['project']} "
            f"group={tracking_cfg['group']} mode={tracking_cfg['mode']}"
        )

    logger.log(f"Config: {args.config}")
    logger.log(f"Mode: {mode}")
    logger.log(f"Run ID: {run_paths.run_id}")

    # Snapshot config for reproducibility (yaml preferred for readability).
    config_snapshot_path = Path(run_paths.run_dir) / "config_snapshot.yaml"
    config_snapshot_path.write_text(yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True), encoding="utf-8")
    manifest_path = Path(run_paths.run_dir) / "run_manifest.json"
    run_manifest = init_run_manifest(
        cfg=cfg,
        run_id=run_paths.run_id,
        config_path=str(cfg.get("__config_path__", "")),
        config_snapshot_path=str(config_snapshot_path),
        run_dir=Path(run_paths.run_dir),
    )
    manifest_add_artifact(run_manifest, config_snapshot_path)
    manifest_add_artifact(run_manifest, Path(run_paths.log_file))
    manifest_add_artifact(run_manifest, manifest_path)

    def _flush_run_manifest(status: str, error: str = "") -> None:
        finalize_run_manifest(run_manifest, status=status, error=error)
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _flush_run_manifest(status="running")

    try:
        process_context["phase"] = "load_stream"
        stream = _load_stream(cfg, mode=mode, logger=logger)
        logger.log(f"Loaded stream: benchmark={stream.benchmark} version={stream.version} segments={len(stream.stream)}")

        debug_loading = str(cfg.get("debug", {}).get("model_loading", "dummy")) if isinstance(cfg.get("debug", {}), dict) else "dummy"
        process_context["phase"] = "build_model"
        print("Calling build_backbone...", file=sys.stderr)
        backbone = build_backbone(cfg.get("model", {}), mode=mode, seed=seed, debug_loading=debug_loading)
        print("Calling build_lora_wrapper...", file=sys.stderr)
        lora = build_lora_wrapper(backbone, cfg.get("lora", {}))
        print("Done build_lora_wrapper.", file=sys.stderr)
        logger.log(f"LoRA: {json.dumps(_summarize_lora_info(lora), ensure_ascii=False)}")

        debug_tools = cfg.get("debug_tools", {}) if isinstance(cfg.get("debug_tools", {}), dict) else {}
        if bool(debug_tools.get("enable_overfit_8_mode", False)):
            _run_overfit_8_mode(
                cfg=cfg,
                stream=stream,
                backbone=backbone,
                lora=lora,
                run_paths=run_paths,
                logger=logger,
            )
            if bool(debug_tools.get("stop_after_overfit_mode", True)):
                logger.log("Stop after overfit_8_mode as requested.")
                for artifact in [
                    Path(run_paths.log_file),
                    config_snapshot_path,
                    manifest_path,
                ]:
                    if artifact.exists():
                        manifest_add_artifact(run_manifest, artifact)
                _flush_run_manifest(status="completed")
                tracker.finish(success=True)
                return

        segment_metrics_rows: List[Dict[str, Any]] = []

        if mode == "debug":
            baseline_name = str(cfg.get("baseline", {}).get("baseline_name", "sequential_lora"))
            final_metrics = run_baseline(
                cfg=cfg,
                stream=stream,
                backbone=backbone,
                lora=lora,
                baseline_name=baseline_name,
                run_paths=run_paths,
                logger=logger,
                segment_metrics_rows=segment_metrics_rows,
                mode="debug",
                tracker=tracker,
            )
        elif mode == "baseline":
            baseline_name = str(cfg.get("baseline_name", "")).strip()
            if not baseline_name:
                raise ValueError("baseline mode requires config field: baseline_name")
            final_metrics = run_baseline(
                cfg=cfg,
                stream=stream,
                backbone=backbone,
                lora=lora,
                baseline_name=baseline_name,
                run_paths=run_paths,
                logger=logger,
                segment_metrics_rows=segment_metrics_rows,
                mode="baseline",
                tracker=tracker,
            )
        else:
            process_context["phase"] = "run_ours"
            final_metrics = run_ours(
                cfg=cfg,
                stream=stream,
                backbone=backbone,
                lora=lora,
                run_paths=run_paths,
                logger=logger,
                segment_metrics_rows=segment_metrics_rows,
                tracker=tracker,
                process_context=process_context,
            )

        process_context["phase"] = "postprocess"
        ccfa_outputs = write_ccfa_postprocess_outputs(
            run_dir=run_paths.run_dir,
            cfg=cfg,
            stream=stream,
            segment_metrics_rows=segment_metrics_rows,
        )
        final_metrics.update(ccfa_outputs)
        summary = ccfa_outputs.get("ccfa_summary") if isinstance(ccfa_outputs.get("ccfa_summary"), dict) else {}
        citb = summary.get("citb") if isinstance(summary.get("citb"), dict) else {}
        if citb.get("ar") is not None:
            final_metrics["rouge_l_ar"] = float(citb["ar"]) * 100.0 if float(citb["ar"]) <= 1.5 else float(citb["ar"])
        if citb.get("fwt") is not None:
            final_metrics["fwt"] = float(citb["fwt"]) * 100.0 if float(citb["fwt"]) <= 1.5 else float(citb["fwt"])
        if citb.get("bwt") is not None:
            final_metrics["bwt"] = float(citb["bwt"]) * 100.0 if abs(float(citb["bwt"])) <= 1.5 else float(citb["bwt"])
        if citb.get("tinit") is not None:
            final_metrics["t_init"] = float(citb["tinit"]) * 100.0 if float(citb["tinit"]) <= 1.5 else float(citb["tinit"])
        if citb.get("tunseen") is not None:
            final_metrics["t_unseen"] = float(citb["tunseen"]) * 100.0 if float(citb["tunseen"]) <= 1.5 else float(citb["tunseen"])

        # Save final metrics + per-segment table
        save_json(run_paths.metrics_json, final_metrics)
        save_csv(run_paths.segment_metrics_csv, segment_metrics_rows)
        logger.log(f"Saved final metrics: {run_paths.metrics_json}")
        logger.log(f"Saved per-segment table: {run_paths.segment_metrics_csv}")

        for artifact in [
            Path(run_paths.log_file),
            Path(run_paths.metrics_json),
            Path(run_paths.segment_metrics_csv),
            config_snapshot_path,
            manifest_path,
            Path(run_paths.run_dir) / "anchor_monitor.jsonl",
            Path(run_paths.run_dir) / "anchor_set.json",
            Path(run_paths.run_dir) / "anchor_set_history.jsonl",
            Path(run_paths.run_dir) / "branch_registry.json",
        ]:
            if artifact.exists():
                manifest_add_artifact(run_manifest, artifact)
        if tracking_cfg.get("log_artifacts", True):
            tracker.log_artifacts(
                [
                    Path(run_paths.metrics_json),
                    Path(run_paths.segment_metrics_csv),
                    config_snapshot_path,
                ]
            )
        tracker.log_final(final_metrics)
        tracker.finish(success=True)
        _flush_run_manifest(status="completed")
        exit_state["finalized"] = True
        _write_process_exit_artifact(
            run_paths=run_paths,
            cfg=cfg,
            event="main_completed",
            status="completed",
            context=process_context,
            exit_code=0,
        )
    except BaseException as exc:
        tb = traceback.format_exc()
        is_system_exit = isinstance(exc, SystemExit)
        exit_code: Optional[int]
        if is_system_exit:
            code = exc.code
            exit_code = int(code) if isinstance(code, int) else 1
            status = "signaled" if exit_state.get("signal_event") else "system_exit"
        else:
            exit_code = 1
            status = "failed"
        exit_state["finalized"] = True
        _write_process_exit_artifact(
            run_paths=run_paths,
            cfg=cfg,
            event=str(exit_state.get("signal_event") or "main_exit"),
            status=status,
            context=process_context,
            exit_code=exit_code,
            error=repr(exc),
            tb=tb,
        )
        tracker.finish(success=False, error=repr(exc))
        _flush_run_manifest(status=status, error=repr(exc))
        raise


def _load_stream(cfg: Dict[str, Any], *, mode: str, logger: SimpleLogger) -> ContinualStream:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    sample_stream_path = paths.get("sample_stream_path")
    processed_dir = paths.get("processed_stream_dir")
    processed_file = str(paths.get("processed_stream_file", "")).strip()
    processed_task_order_file = str(paths.get("processed_task_order_file", "")).strip()
    raw_citb_root = str(paths.get("raw_citb_root", "data/raw/citb")).strip()

    if mode == "debug":
        dbg = cfg.get("debug", {}) if isinstance(cfg.get("debug", {}), dict) else {}
        max_segments = int(dbg.get("max_segments", -1))
        max_train = int(dbg.get("max_train_examples_per_segment", -1))
        max_eval = int(dbg.get("max_eval_examples_per_segment", -1))
        return load_continual_stream(
            mode="debug",
            sample_stream_path=str(sample_stream_path),
            processed_stream_dir=None,
            processed_stream_file="",
            max_segments=max_segments,
            max_train_examples_per_segment=max_train,
            max_eval_examples_per_segment=max_eval,
        )

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    processed_stream_name = str(data_cfg.get("stream_name", "")).strip()
    auto_prepare_processed = bool(data_cfg.get("auto_prepare_processed", False))
    processed_stream_train_instances_per_task = int(data_cfg.get("processed_stream_train_instances_per_task", 50))
    processed_stream_eval_instances_per_task = int(data_cfg.get("processed_stream_eval_instances_per_task", 10))
    processed_stream_dev_instances_per_task = int(data_cfg.get("processed_stream_dev_instances_per_task", 50))
    processed_stream_limit_tasks = int(data_cfg.get("processed_stream_limit_tasks", -1))
    max_segments = int(data_cfg.get("max_segments", -1))
    max_train = int(data_cfg.get("max_train_examples_per_segment", -1))
    max_eval = int(data_cfg.get("max_eval_examples_per_segment", -1))
    skip_empty_segments = bool(data_cfg.get("skip_empty_segments", False))

    if processed_dir is None:
        raise ValueError("processed_stream_dir is required for baseline/ours modes")

    if processed_stream_name:
        logger.log(
            f"Requested processed stream alias='{processed_stream_name}' "
            f"(auto_prepare_processed={auto_prepare_processed})"
        )

    return load_continual_stream(
        mode=mode,
        sample_stream_path=None,
        processed_stream_dir=str(processed_dir),
        processed_stream_file=processed_file,
        processed_stream_name=processed_stream_name,
        processed_task_order_file=processed_task_order_file,
        auto_prepare_processed=auto_prepare_processed,
        raw_citb_root=raw_citb_root,
        seed=int(cfg.get("seed", 0)),
        processed_stream_train_instances_per_task=processed_stream_train_instances_per_task,
        processed_stream_eval_instances_per_task=processed_stream_eval_instances_per_task,
        processed_stream_dev_instances_per_task=processed_stream_dev_instances_per_task,
        processed_stream_limit_tasks=processed_stream_limit_tasks,
        max_segments=max_segments,
        max_train_examples_per_segment=max_train,
        max_eval_examples_per_segment=max_eval,
        skip_empty_segments=skip_empty_segments,
    )


def _drift_anchor_refresh_segment_count(cfg: Dict[str, Any]) -> int:
    drift_cfg = cfg.get("drift", {}) if isinstance(cfg.get("drift", {}), dict) else {}
    return max(1, int(drift_cfg.get("anchor_refresh_segments", 1)))


def _maybe_build_drift_anchor_set(
    *,
    stream: ContinualStream,
    source_segments: List[Segment],
    drift: Optional[DriftDetector],
    cfg: Dict[str, Any],
    run_paths: RunPaths,
    logger: SimpleLogger,
    reason: str,
    model: Optional[Any] = None,
) -> Optional[AnchorSet]:
    if drift is None or not source_segments:
        return None
    drift_cfg = cfg.get("drift", {}) if isinstance(cfg.get("drift", {}), dict) else {}
    anchor_stream = ContinualStream(
        benchmark=stream.benchmark,
        version=stream.version,
        stream=list(source_segments),
    )
    anchor_set = build_anchor_set(anchor_stream, drift_cfg, seed=int(cfg.get("seed", 0)), model=model)
    save_json(str(Path(run_paths.run_dir) / "anchor_set.json"), anchor_set.to_dict())
    source_segment_ids = [int(seg.segment_id) for seg in source_segments]
    append_jsonl(
        str(Path(run_paths.run_dir) / "anchor_set_history.jsonl"),
        {
            "reason": reason,
            "source_segment_ids": source_segment_ids,
            "num_source_segments": len(source_segment_ids),
            **anchor_set.summary(),
        },
    )
    logger.log(
        "Built drift anchor set "
        f"({reason}): {json.dumps({**anchor_set.summary(), 'source_segment_ids': source_segment_ids}, ensure_ascii=False)}"
    )
    return anchor_set


def _compute_anchor_monitor_metrics(model: Any, anchor_set: AnchorSet) -> Dict[str, Any]:
    core_pairs = [(x.instruction, x.input_text) for x in anchor_set.core]
    core_targets = [x.output for x in anchor_set.core]
    probe_pairs = [(x.instruction, x.input_text) for x in anchor_set.probe]
    probe_targets = [x.output for x in anchor_set.probe]

    core_nlls = model.score_answer_nlls(core_pairs, core_targets)
    probe_nlls = model.score_answer_nlls(probe_pairs, probe_targets)
    return {
        "core_mean_nll": float(sum(core_nlls) / max(1, len(core_nlls))),
        "probe_mean_nll": float(sum(probe_nlls) / max(1, len(probe_nlls))),
        "core_num_examples": int(len(core_nlls)),
        "probe_num_examples": int(len(probe_nlls)),
    }


def _record_drift_monitor(
    *,
    run_paths: RunPaths,
    seg: Segment,
    active_adapter: str,
    monitor_metrics: Dict[str, Any],
    event: DriftEvent,
) -> Dict[str, Any]:
    row = {
        "segment_id": int(seg.segment_id),
        "segment_name": str(seg.segment_name),
        "active_adapter": str(active_adapter),
        **monitor_metrics,
        **event.to_row(),
    }
    append_jsonl(str(Path(run_paths.run_dir) / "anchor_monitor.jsonl"), row)
    if event.triggered:
        append_jsonl(str(Path(run_paths.run_dir) / "drift_events.jsonl"), row)
    return row


def _summarize_drift_proxy_quality(stream: ContinualStream, drift: Optional[DriftDetector]) -> Dict[str, Any]:
    if drift is None:
        return {}
    history = drift.monitor_history()
    events = drift.drift_events()
    if not history:
        return {}

    true_shift_segments = [int(seg.segment_id) for seg in stream.stream[1:]]
    detected_segments = [int(row.get("segment_id", -1)) for row in events]
    matched_event_indices: set[int] = set()
    delays: List[float] = []
    misses = 0
    for shift_seg in true_shift_segments:
        match_idx = next(
            (idx for idx, det_seg in enumerate(detected_segments) if idx not in matched_event_indices and det_seg >= shift_seg),
            None,
        )
        if match_idx is None:
            misses += 1
            continue
        matched_event_indices.add(int(match_idx))
        delays.append(float(detected_segments[match_idx] - shift_seg))

    false_alarms = len([idx for idx in range(len(detected_segments)) if idx not in matched_event_indices])
    return {
        "num_monitor_points": int(len(history)),
        "num_true_shifts": int(len(true_shift_segments)),
        "num_detected_events": int(len(detected_segments)),
        "false_alarm_count": int(false_alarms),
        "false_alarm_rate": float(false_alarms / max(1, len(history))),
        "miss_count": int(misses),
        "miss_rate": float(misses / max(1, len(true_shift_segments))),
        "detection_delay_mean": float(sum(delays) / max(1, len(delays))) if delays else -1.0,
    }


def _routing_row_metrics(eval_metrics: Dict[str, Any]) -> Dict[str, Any]:
    extra = eval_metrics.get("extra", {}) if isinstance(eval_metrics.get("extra", {}), dict) else {}
    routing = extra.get("routing", {}) if isinstance(extra.get("routing", {}), dict) else {}
    return {
        "eval.anytime_score": float(extra.get("anytime_score", eval_metrics.get("seen_avg_score", 0.0))),
        "eval.current_task_aware_score": float(eval_metrics.get("current_task_aware_score", 0.0)),
        "eval.seen_avg_task_aware_score": float(eval_metrics.get("seen_avg_task_aware_score", 0.0)),
        "eval.task_aware_forgetting": float(eval_metrics.get("task_aware_forgetting", 0.0)),
        "eval.anytime_task_aware_score": float(
            extra.get("anytime_task_aware_score", eval_metrics.get("seen_avg_task_aware_score", 0.0))
        ),
        "eval.task_aware_score_mean": float(extra.get("task_aware_score_mean", 0.0)),
        "routing.num_routed": int(routing.get("num_routed", 0)),
        "routing.oracle_agreement_rate": float(routing.get("oracle_agreement_rate", 0.0)),
        "routing.decision_confidence_mean": float(routing.get("decision_confidence_mean", 0.0)),
        "routing.decision_entropy_mean": float(routing.get("decision_entropy_mean", 0.0)),
        "routing.oracle_margin_mean": float(routing.get("oracle_margin_mean", 0.0)),
    }


def run_baseline(
    *,
    cfg: Dict[str, Any],
    stream: ContinualStream,
    backbone: Any,
    lora: Any,
    baseline_name: str,
    run_paths: RunPaths,
    logger: SimpleLogger,
    segment_metrics_rows: List[Dict[str, Any]],
    mode: str,
    tracker: Optional[WandbTracker] = None,
) -> Dict[str, Any]:
    logger.log(f"Baseline selected: {baseline_name}")

    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    lr = float(train_cfg.get("lr", 1e-3))
    epochs = int(train_cfg.get("epochs_per_segment", 1))
    batch_size = int(train_cfg.get("batch_size", 1))
    eval_max_new_tokens = int(model_cfg.get("gen_max_new_tokens", 64))

    # Optional shared components (used by some baselines)
    lora_bank = LoRABank(max_branches=int(cfg.get("periodic", {}).get("max_branches", 8)) if isinstance(cfg.get("periodic", {}), dict) else 8)
    router = Router(cfg.get("router", {}) if isinstance(cfg.get("router", {}), dict) else {})
    drift = DriftDetector(cfg.get("drift", {}) if isinstance(cfg.get("drift", {}), dict) else {})
    drift_anchor_set: Optional[AnchorSet] = None
    anchor_refresh_segments = _drift_anchor_refresh_segment_count(cfg)

    method = _build_baseline_method(baseline_name, cfg)
    debug_tools = cfg.get("debug_tools", {}) if isinstance(cfg.get("debug_tools", {}), dict) else {}
    normalization_cfg = cfg.get("eval_normalization", {}) if isinstance(cfg.get("eval_normalization", {}), dict) else {}

    seen_segments: List[Segment] = []
    historical_best_per_segment: Dict[int, float] = {}
    historical_best_task_aware_per_segment: Dict[int, float] = {}
    stream_segments = stream.stream
    last_eval_metrics: Dict[str, Any] = {}
    if bool(debug_tools.get("enable_single_segment_mode", False)):
        target_segment_idx = int(debug_tools.get("single_segment_index", 0))
        stream_segments = [stream.stream[target_segment_idx]]
        logger.log(f"Single-segment mode enabled: segment_index={target_segment_idx}")
    for seg in stream_segments:
        logger.log(f"=== Segment {seg.segment_id}: {seg.segment_name} ===")

        if baseline_name == "bank_no_router" and drift is not None and drift_anchor_set is None:
            drift_anchor_set = _maybe_build_drift_anchor_set(
                stream=stream,
                source_segments=[seg],
                drift=drift,
                cfg=cfg,
                run_paths=run_paths,
                logger=logger,
                reason=f"bootstrap_segment_{seg.segment_id}",
                model=backbone,
            )

        hook_info = {}
        if hasattr(method, "on_segment_start"):
            hook_info = method.on_segment_start(segment=seg, model=backbone, lora=lora)

        # Train
        if baseline_name == "router_only":
            train_metrics = method.train_on_segment(
                segment=seg,
                model=backbone,
                lora=lora,
                router=router,
                lora_bank=lora_bank,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
            )
        elif baseline_name == "bank_no_router":
            train_metrics = _train_bank_no_router(
                segment=seg,
                model=backbone,
                lora=lora,
                lora_bank=lora_bank,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
            )
        else:
            train_metrics = method.train_on_segment(
                segment=seg, model=backbone, lora=lora, lr=lr, epochs=epochs, batch_size=batch_size
            )

        logger.log(f"Train metrics: {json.dumps(train_metrics, ensure_ascii=False)}")

        # Evaluate on seen segments (unified)
        seen_segments.append(seg)
        active_adapter_before_eval = lora.get_active_adapter_name()
        eval_metrics = evaluate_stream(
            model=backbone,
            segments_seen=seen_segments,
            max_new_tokens=eval_max_new_tokens,
            router=router if baseline_name == "router_only" else None,
            lora_bank=lora_bank if baseline_name == "router_only" else None,
            segment_id=seg.segment_id,
            normalization_cfg=normalization_cfg,
            save_debug_examples_dir=str(Path(run_paths.run_dir) / "eval_debug"),
            historical_best_per_segment=historical_best_per_segment,
            historical_best_task_aware_per_segment=historical_best_task_aware_per_segment,
        )
        last_eval_metrics = eval_metrics
        if lora_bank.list_branches() and active_adapter_before_eval in lora.list_adapters():
            lora.set_active_adapter(active_adapter_before_eval)
        logger.log(f"Eval metrics: {json.dumps(eval_metrics, ensure_ascii=False)}")

        drift_row: Optional[Dict[str, Any]] = None
        if baseline_name == "bank_no_router" and drift_anchor_set is not None:
            active_branch = lora_bank.get_active_branch() if lora_bank.list_branches() else lora.get_active_adapter_name()
            if active_branch in lora.list_adapters():
                lora.set_active_adapter(active_branch)
            monitor_metrics = _compute_anchor_monitor_metrics(backbone, drift_anchor_set)
            event = drift.update(
                core_mean_nll=float(monitor_metrics["core_mean_nll"]),
                probe_mean_nll=float(monitor_metrics["probe_mean_nll"]),
                segment_id=seg.segment_id,
                monitor_step=seg.segment_id + 1,
            )
            drift_row = _record_drift_monitor(
                run_paths=run_paths,
                seg=seg,
                active_adapter=active_branch,
                monitor_metrics=monitor_metrics,
                event=event,
            )
            logger.log(
                "Drift detector: "
                f"triggered={event.triggered} calibrated={event.calibrated} "
                f"probe_mean_nll={event.probe_mean_nll:.4f} core_mean_nll={event.core_mean_nll:.4f} "
                f"probe_cusum={event.probe_cusum:.4f} threshold={event.threshold:.4f} reason={event.reason}"
            )
            if event.triggered:
                lora_bank.freeze_current_branch()
                new_b = lora_bank.spawn_new_branch(lora_wrapper=lora, segment_id=seg.segment_id)
                train_metrics["spawned_branch"] = new_b
                logger.log(f"Spawned new branch due to drift: {new_b}")
                drift.reset(keep_history=True)
                drift_anchor_set = _maybe_build_drift_anchor_set(
                    stream=stream,
                    source_segments=seen_segments[-anchor_refresh_segments:] or [seg],
                    drift=drift,
                    cfg=cfg,
                    run_paths=run_paths,
                    logger=logger,
                    reason=f"refresh_after_drift_segment_{seg.segment_id}",
                    model=backbone,
                )

        row = {
            "run_id": run_paths.run_id,
            "mode": mode,
            "baseline_name": baseline_name,
            "segment_id": seg.segment_id,
            "segment_name": seg.segment_name,
            **_flatten_metrics("train", train_metrics),
            **_flatten_metrics("eval", eval_metrics),
            **matrix_columns_from_eval(eval_metrics),
            **_routing_row_metrics(eval_metrics),
            **{f"hook.{k}": v for k, v in (hook_info or {}).items()},
            "active_adapter": lora.get_active_adapter_name(),
        }
        if drift_row is not None:
            row["drift.triggered"] = bool(drift_row.get("triggered", False))
            row["drift.score"] = float(drift_row.get("score", 0.0))
            row["drift.core_mean_nll"] = float(drift_row.get("core_mean_nll", 0.0))
            row["drift.probe_mean_nll"] = float(drift_row.get("probe_mean_nll", 0.0))
            row["drift.core_cusum"] = float(drift_row.get("core_cusum", 0.0))
            row["drift.probe_cusum"] = float(drift_row.get("probe_cusum", 0.0))
        segment_metrics_rows.append(row)

        # Save per-segment artifact
        seg_dir = ensure_dir(str(Path(run_paths.run_dir) / f"segment_{seg.segment_id:03d}"))
        save_json(str(Path(seg_dir) / "train_metrics.json"), train_metrics)
        save_json(str(Path(seg_dir) / "eval_metrics.json"), eval_metrics)
        if baseline_name == "bank_no_router":
            save_json(str(Path(seg_dir) / "drift_state.json"), drift.state_dict())
            save_json(str(Path(seg_dir) / "bank_state.json"), lora_bank.state_dict())
            save_json(str(Path(run_paths.run_dir) / "branch_registry.json"), lora_bank.state_dict())
        append_jsonl(str(Path(run_paths.run_dir) / "metrics.jsonl"), row)
        if tracker is not None:
            tracker.log_segment_row(row)

    final = segment_metrics_rows[-1] if segment_metrics_rows else {}
    return {
        "run_id": run_paths.run_id,
        "mode": mode,
        "baseline_name": baseline_name,
        "final": final,
        "drift_quality": _summarize_drift_proxy_quality(stream, drift if baseline_name == "bank_no_router" else None),
        "routing_quality": (
            (last_eval_metrics.get("extra", {}) if isinstance(last_eval_metrics.get("extra", {}), dict) else {}).get("routing", {})
            if last_eval_metrics
            else {}
        ),
    }


def run_ours(
    *,
    cfg: Dict[str, Any],
    stream: ContinualStream,
    backbone: Any,
    lora: Any,
    run_paths: RunPaths,
    logger: SimpleLogger,
    segment_metrics_rows: List[Dict[str, Any]],
    tracker: Optional[WandbTracker] = None,
    process_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    modules = cfg.get("modules", {}) if isinstance(cfg.get("modules", {}), dict) else {}
    use_drift = bool(modules.get("use_drift_detector", True))
    use_bank = bool(modules.get("use_lora_bank", True))
    use_router = bool(modules.get("use_router", True))
    use_overlap = bool(modules.get("use_overlap_loss", True))

    logger.log(
        "Ours modules: "
        + json.dumps(
            {
                "use_drift_detector": use_drift,
                "use_lora_bank": use_bank,
                "use_router": use_router,
                "use_overlap_loss": use_overlap,
            },
            ensure_ascii=False,
        )
    )

    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    lr = float(train_cfg.get("lr", 1e-4))
    epochs = int(train_cfg.get("epochs_per_segment", 1))
    batch_size = int(train_cfg.get("batch_size", 1))
    eval_max_new_tokens = int(model_cfg.get("gen_max_new_tokens", 64))

    drift = DriftDetector(cfg.get("drift", {}) if isinstance(cfg.get("drift", {}), dict) else {}) if use_drift else None
    bank_cfg = cfg.get("bank", {}) if isinstance(cfg.get("bank", {}), dict) else {}
    lora_bank = LoRABank(max_branches=int(bank_cfg.get("max_branches", 8))) if use_bank else None
    router = Router(cfg.get("router", {}) if isinstance(cfg.get("router", {}), dict) else {}) if use_router else None
    overlap_cfg = cfg.get("overlap", {}) if isinstance(cfg.get("overlap", {}), dict) else {}
    beta = float(overlap_cfg.get("beta", 0.1))
    normalization_cfg = cfg.get("eval_normalization", {}) if isinstance(cfg.get("eval_normalization", {}), dict) else {}
    drift_anchor_set: Optional[AnchorSet] = None
    anchor_refresh_segments = _drift_anchor_refresh_segment_count(cfg)
    ssrg_cfg = cfg.get("spectral_replay", {}) if isinstance(cfg.get("spectral_replay", {}), dict) else {}
    spectral_replay = (
        SpectralSparseReplayGate(ssrg_cfg) if bool(ssrg_cfg.get("enabled", False)) else None
    )

    print("Initializing LoRABank...", file=sys.stderr)
    # Initialize bank with first branch if enabled
    if lora_bank is not None:
        lora_bank.initialize(lora_wrapper=lora, initial_branch="b0", segment_id=0)
    print("Done initializing LoRABank.", file=sys.stderr)

    seen_segments: List[Segment] = []
    historical_best_per_segment: Dict[int, float] = {}
    historical_best_task_aware_per_segment: Dict[int, float] = {}
    last_eval_metrics: Dict[str, Any] = {}
    print("Starting segment loop...", file=sys.stderr)
    for seg in stream.stream:
        if process_context is not None:
            process_context.update(
                {
                    "phase": "segment_train",
                    "current_segment_id": int(seg.segment_id),
                    "current_segment_name": str(seg.segment_name),
                }
            )
        print(f"Inside loop for segment {seg.segment_id}...", file=sys.stderr)
        logger.log(f"=== Segment {seg.segment_id}: {seg.segment_name} ===")

        seg_train = seg
        if spectral_replay is not None:
            tok = getattr(backbone, "tokenizer", None)
            if tok is not None:
                prompt_fn = lambda ins, inp: format_for_infer(tok, ins, inp)  # noqa: E731
            else:
                prompt_fn = lambda ins, inp: f"{ins}\n\n{inp}"  # noqa: E731
            seg_train, ssrg_metrics = spectral_replay.augment_segment(
                seg, model=backbone, prompt_fn=prompt_fn
            )
            if ssrg_metrics.get("ssrg_replay_added", 0) > 0:
                logger.log(f"SSRG replay: {json.dumps(ssrg_metrics, ensure_ascii=False)}")

        if drift is not None and drift_anchor_set is None:
            print("Building drift anchor set...", file=sys.stderr)
            drift_anchor_set = _maybe_build_drift_anchor_set(
                stream=stream,
                source_segments=[seg],
                drift=drift,
                cfg=cfg,
                run_paths=run_paths,
                logger=logger,
                reason=f"bootstrap_segment_{seg.segment_id}",
                model=backbone,
            )
            print("Done building drift anchor set.", file=sys.stderr)

        # Decide which branch to use for training
        if lora_bank is not None and router is not None:
            # V8: Assess-then-Update Subspace Decomposition
            transient_name = "b_transient"
            if transient_name in lora.list_adapters():
                if hasattr(lora, "delete_adapter"):
                    lora.delete_adapter(transient_name)
            lora.create_adapter(transient_name)
            lora.set_active_adapter(transient_name)
            
            logger.log(f"Training transient probe expert for segment {seg.segment_id}...")
            _train_on_active_branch(
                segment=seg_train,
                model=backbone,
                lora=lora,
                lora_bank=lora_bank,
                lr=lr,
                epochs=1,
                batch_size=batch_size,
                use_overlap=False,
                beta=0.0,
                overlap_cfg={},
                train_cfg=train_cfg,
            )
            
            from core.methods.assess_update import assess_transient_branch, set_adapter_vector
            existing_branches = lora_bank.list_branches()
            assess_cfg = cfg.get("assess_update", {}) if isinstance(cfg.get("assess_update", {}), dict) else {}
            assessment_threshold = float(assess_cfg.get("isolated_energy_threshold", 0.2))
            assessment = assess_transient_branch(
                lora,
                transient_name,
                existing_branches,
                threshold=assessment_threshold,
            )
            logger.log(f"Transient assessment: {assessment['action']}, isolated_energy_ratio: {assessment['isolated_energy_ratio']:.4f}")
            
            if hasattr(lora, "delete_adapter"):
                lora.delete_adapter(transient_name)
                
            # Always merge the shared subspace into the best matching existing branch
            target = assessment["target"]
            if target is not None:
                # Unfreeze the target branch so it can be updated
                lora_bank.unfreeze_branch(target)
                target_vec = lora.get_adapter_vector(target, detach=True)
                proj_v = assessment.get("proj_vec", None)
                if proj_v is None:
                    proj_v = assessment.get("residual_vec", torch.zeros_like(target_vec)) # fallback
                set_adapter_vector(lora, target, target_vec + proj_v.to(target_vec.device))
                logger.log(f"Merged shared subspace into {target} and unfroze it.")
                
            if assessment["action"] == "spawn":
                if bool(bank_cfg.get("freeze_old_branches", True)):
                    # Freeze all branches except the target we just unfroze
                    for b in lora_bank.list_branches():
                        if b != target:
                            if hasattr(lora_bank, "_branches") and b in lora_bank._branches:
                                lora_bank._branches[b].frozen = True
                            if hasattr(lora, "freeze_adapter"):
                                lora.freeze_adapter(b)
                                
                new_b = lora_bank.spawn_new_branch(lora_wrapper=lora, segment_id=seg.segment_id)
                logger.log(f"Spawned new branch {new_b} based on subspace energy.")
                
                # Transfer the isolated subspace to the new branch
                set_adapter_vector(lora, new_b, assessment["residual_vec"].to(lora.get_adapter_vector(new_b, detach=True).device))
                logger.log(f"Transferred isolated subspace to {new_b}.")
                
                # Initialize router prototype for the new branch
                if drift_anchor_set is not None:
                    spawn_proto = _maybe_init_spawn_prototype_from_anchors(
                        router=router,
                        model=backbone,
                        lora=lora,
                        lora_bank=lora_bank,
                        new_branch=new_b,
                        drift_anchor_set=drift_anchor_set,
                    )
                    if spawn_proto:
                        logger.log(f"Spawn-sync prototype init: {json.dumps(spawn_proto, ensure_ascii=False)}")
                
                # Refresh anchor set after spawning
                drift_anchor_set = _maybe_build_drift_anchor_set(
                    stream=stream,
                    source_segments=seen_segments[-anchor_refresh_segments:] if len(seen_segments) > 0 else [seg],
                    drift=drift,
                    cfg=cfg,
                    run_paths=run_paths,
                    logger=logger,
                    reason=f"refresh_after_spawn_segment_{seg.segment_id}",
                    model=backbone,
                )
            else:
                logger.log(f"No new branch spawned. Subspace energy shared with {target}.")
                if target is not None:
                    lora.set_active_adapter(target)
                    lora_bank._active = target
                    
                    # Merge the remaining isolated subspace (residual) into the target branch
                    target_vec = lora.get_adapter_vector(target, detach=True)
                    set_adapter_vector(lora, target, target_vec + assessment["residual_vec"].to(target_vec.device))
                    logger.log(f"Merged remaining isolated subspace into {target}.")

            warmup_metrics = _maybe_train_dialogue_replay_warmup(
                segment=seg_train,
                model=backbone,
                lora=lora,
                lora_bank=lora_bank,
                spectral_replay=spectral_replay,
                train_cfg=train_cfg,
                lr=lr,
                batch_size=batch_size,
                logger=logger,
            )

            # Route per example (hard routing); switch adapter before fit
            train_metrics = _train_with_router(
                segment=seg_train,
                model=backbone,
                lora=lora,
                lora_bank=lora_bank,
                router=router,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                use_overlap=use_overlap,
                beta=beta,
                overlap_cfg=overlap_cfg,
                prev_eval_metrics=last_eval_metrics,
                train_cfg=train_cfg,
            )
            if warmup_metrics:
                train_metrics.update(warmup_metrics)
            refresh_metrics = _maybe_refresh_segment_prototypes_from_anchors(
                router=router,
                model=backbone,
                lora=lora,
                lora_bank=lora_bank,
                drift_anchor_set=drift_anchor_set,
            )
            if refresh_metrics:
                train_metrics.update(refresh_metrics)
        elif lora_bank is not None and router is None:
            # No router: always use the active branch
            train_metrics = _train_on_active_branch(
                segment=seg_train,
                model=backbone,
                lora=lora,
                lora_bank=lora_bank,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                use_overlap=use_overlap,
                beta=beta,
                overlap_cfg=overlap_cfg,
                train_cfg=train_cfg,
            )
        else:
            # No bank: fall back to single default adapter
            if "default" in lora.list_adapters():
                lora.set_active_adapter("default")
            train_metrics = _train_on_active_branch(
                segment=seg_train,
                model=backbone,
                lora=lora,
                lora_bank=lora_bank,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                use_overlap=use_overlap,
                beta=beta,
                overlap_cfg=overlap_cfg,
                train_cfg=train_cfg,
            )

        if spectral_replay is not None:
            train_metrics.update(spectral_replay.state_dict().get("last_metrics", {}))
            spectral_replay.ingest_segment(seg)

        # Optional overlap loss (logged as a scalar proxy)
        overlap_value = 0.0
        overlap_mean_cosine = 0.0
        if use_overlap and lora_bank is not None:
            tok = getattr(backbone, "tokenizer", None)
            overlap_items = drift_anchor_set.probe if drift_anchor_set is not None else []
            if overlap_items:
                if tok is not None:
                    anchor_prompts = [
                        format_for_infer(tok, item.instruction, item.input_text, add_generation_prompt=True)
                        for item in overlap_items
                    ]
                else:
                    anchor_prompts = [f"{item.instruction}\n\n{item.input_text}" for item in overlap_items]
            elif tok is not None:
                anchor_prompts = [format_for_infer(tok, ex.instruction, ex.input) for ex in seg.train[:8]]
            else:
                anchor_prompts = [f"{ex.instruction}\n\n{ex.input}" for ex in seg.train[:8]]
            activations_by_branch: Dict[str, List[List[float]]] = {}
            for b in lora_bank.list_branches():
                lora.set_active_adapter(b)
                activations_by_branch[b] = backbone.get_activations(anchor_prompts)
            overlap_mean_cosine = compute_overlap_loss(activations_by_branch=activations_by_branch, beta=1.0)
            overlap_value = float(overlap_mean_cosine * beta)
            train_metrics["overlap_loss_proxy"] = float(overlap_value)
            train_metrics["overlap_mean_cosine"] = float(overlap_mean_cosine)

        logger.log(f"Train metrics: {json.dumps(train_metrics, ensure_ascii=False)}")

        # Evaluate
        seen_segments.append(seg)
        active_adapter_before_eval = lora.get_active_adapter_name()
        if process_context is not None:
            process_context["phase"] = "segment_eval"
        eval_metrics = _run_eval_with_diagnostics(
            cfg=cfg,
            run_paths=run_paths,
            logger=logger,
            tracker=tracker,
            seg=seg,
            seen_segments=seen_segments,
            model=backbone,
            max_new_tokens=eval_max_new_tokens,
            router=router if (router is not None and lora_bank is not None) else None,
            lora_bank=lora_bank if (router is not None and lora_bank is not None) else None,
            normalization_cfg=normalization_cfg,
            historical_best_per_segment=historical_best_per_segment,
            historical_best_task_aware_per_segment=historical_best_task_aware_per_segment,
        )
        last_eval_metrics = eval_metrics
        if active_adapter_before_eval in lora.list_adapters():
            lora.set_active_adapter(active_adapter_before_eval)
        logger.log(f"Eval metrics: {json.dumps(eval_metrics, ensure_ascii=False)}")

        # Drift update after evaluation using fixed anchor monitoring
        drift_event = None
        if drift is not None and drift_anchor_set is not None:
            active_branch = lora_bank.get_active_branch() if lora_bank is not None else lora.get_active_adapter_name()
            if active_branch in lora.list_adapters():
                lora.set_active_adapter(active_branch)
            monitor_metrics = _compute_anchor_monitor_metrics(backbone, drift_anchor_set)
            drift_event = drift.update(
                core_mean_nll=float(monitor_metrics["core_mean_nll"]),
                probe_mean_nll=float(monitor_metrics["probe_mean_nll"]),
                segment_id=seg.segment_id,
                monitor_step=seg.segment_id + 1,
            )
            _record_drift_monitor(
                run_paths=run_paths,
                seg=seg,
                active_adapter=active_branch,
                monitor_metrics=monitor_metrics,
                event=drift_event,
            )
            logger.log(
                "Drift detector: "
                f"triggered={drift_event.triggered} calibrated={drift_event.calibrated} "
                f"probe_mean_nll={drift_event.probe_mean_nll:.4f} core_mean_nll={drift_event.core_mean_nll:.4f} "
                f"probe_cusum={drift_event.probe_cusum:.4f} threshold={drift_event.threshold:.4f} "
                f"reason={drift_event.reason}"
            )

        # Spawn branch on drift (if enabled) - DISABLED FOR V8 (Assess-then-Update Subspace Decomposition)
        # if drift is not None and drift_event is not None and drift_event.triggered:
        #     if lora_bank is not None and bool(bank_cfg.get("spawn_on_drift", True)):
        #         if bool(bank_cfg.get("freeze_old_branches", True)):
        #             lora_bank.freeze_current_branch()
        #         new_b = lora_bank.spawn_new_branch(lora_wrapper=lora, segment_id=seg.segment_id)
        #         logger.log(f"Spawned new branch due to drift: {new_b}")
        #         spawn_proto = _maybe_init_spawn_prototype_from_anchors(
        #             router=router,
        #             model=backbone,
        #             lora=lora,
        #             lora_bank=lora_bank,
        #             new_branch=new_b,
        #             drift_anchor_set=drift_anchor_set,
        #         )
        #         if spawn_proto:
        #             logger.log(f"Spawn-sync prototype init: {json.dumps(spawn_proto, ensure_ascii=False)}")
        #     drift.reset(keep_history=True)
        #     drift_anchor_set = _maybe_build_drift_anchor_set(
        #         stream=stream,
        #         source_segments=seen_segments[-anchor_refresh_segments:] or [seg],
        #         drift=drift,
        #         cfg=cfg,
        #         run_paths=run_paths,
        #         logger=logger,
        #         reason=f"refresh_after_drift_segment_{seg.segment_id}",
        #         model=backbone,
        #     )

        row = {
            "run_id": run_paths.run_id,
            "mode": "ours",
            "segment_id": seg.segment_id,
            "segment_name": seg.segment_name,
            **_flatten_metrics("train", train_metrics),
            **_flatten_metrics("eval", eval_metrics),
            **matrix_columns_from_eval(eval_metrics),
            **_routing_row_metrics(eval_metrics),
            "active_adapter": lora.get_active_adapter_name(),
            "overlap_loss_proxy": float(overlap_value),
            "overlap_mean_cosine": float(overlap_mean_cosine),
            "num_branches": len(lora_bank.list_branches()) if lora_bank is not None else 1,
        }
        if drift_event is not None:
            row["drift.triggered"] = bool(drift_event.triggered)
            row["drift.score"] = float(drift_event.score)
            row["drift.core_mean_nll"] = float(drift_event.core_mean_nll)
            row["drift.probe_mean_nll"] = float(drift_event.probe_mean_nll)
            row["drift.core_cusum"] = float(drift_event.core_cusum)
            row["drift.probe_cusum"] = float(drift_event.probe_cusum)
        segment_metrics_rows.append(row)

        seg_dir = ensure_dir(str(Path(run_paths.run_dir) / f"segment_{seg.segment_id:03d}"))
        save_json(str(Path(seg_dir) / "train_metrics.json"), train_metrics)
        save_json(str(Path(seg_dir) / "eval_metrics.json"), eval_metrics)
        if drift is not None:
            save_json(str(Path(seg_dir) / "drift_state.json"), drift.state_dict())
        if lora_bank is not None:
            save_json(str(Path(seg_dir) / "bank_state.json"), lora_bank.state_dict())
            save_json(str(Path(run_paths.run_dir) / "branch_registry.json"), lora_bank.state_dict())
        if router is not None:
            save_json(str(Path(seg_dir) / "router_state.json"), router.state_dict())
        append_jsonl(str(Path(run_paths.run_dir) / "metrics.jsonl"), row)
        if tracker is not None:
            tracker.log_segment_row(row)

        # Early stopping defaults preserve legacy SOTA-chase behavior, but
        # strict benchmark runs should leave an auditable stop_and_diagnose file.
        train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
        min_segments_before_stop = int(train_cfg.get("early_stop_min_segment", 3))
        if seg.segment_id >= min_segments_before_stop:
            seen_avg = float(eval_metrics.get("seen_avg_score", 0))
            hard_stop_enabled = bool(train_cfg.get("hard_early_stop_enabled", True))
            hard_stop_min_seen_avg = float(train_cfg.get("hard_early_stop_min_seen_avg", 0.2))
            if hard_stop_enabled and seen_avg < hard_stop_min_seen_avg:
                logger.log(
                    f"Early stopping triggered: seen_avg_score {seen_avg} < {hard_stop_min_seen_avg} "
                    f"at segment {seg.segment_id}"
                )
                _write_stop_and_diagnose(
                    run_paths=run_paths,
                    cfg=cfg,
                    segment_id=seg.segment_id,
                    reason="hard_early_stop_min_seen_avg",
                    eval_metrics=eval_metrics,
                    extra={"threshold": hard_stop_min_seen_avg},
                )
                sys.exit(1)
            min_task_aware = train_cfg.get("early_stop_min_task_aware_score")
            if min_task_aware is not None:
                task_seen = float(eval_metrics.get("seen_avg_task_aware_score", 0.0))
                min_task_aware_f = float(min_task_aware)
                if task_seen < min_task_aware_f:
                    logger.log(
                        f"Early stopping triggered: seen_avg_task_aware_score {task_seen} < "
                        f"{min_task_aware_f} at segment {seg.segment_id}"
                    )
                    _write_stop_and_diagnose(
                        run_paths=run_paths,
                        cfg=cfg,
                        segment_id=seg.segment_id,
                        reason="early_stop_min_task_aware_score",
                        eval_metrics=eval_metrics,
                        extra={"threshold": min_task_aware_f},
                    )
                    sys.exit(42)
            traj_floor = _trajectory_early_stop_floor(cfg, seg.segment_id)
            if traj_floor is not None and seen_avg < traj_floor:
                logger.log(
                    f"Trajectory early stopping: seen_avg_score {seen_avg} < baseline floor {traj_floor} "
                    f"at segment {seg.segment_id}"
                )
                _write_stop_and_diagnose(
                    run_paths=run_paths,
                    cfg=cfg,
                    segment_id=seg.segment_id,
                    reason="trajectory_early_stop_floor",
                    eval_metrics=eval_metrics,
                    extra={"threshold": traj_floor},
                )
                sys.exit(1)
            if _ours_v10_early_stop_triggered(cfg, seg.segment_id, eval_metrics, logger):
                _write_stop_and_diagnose(
                    run_paths=run_paths,
                    cfg=cfg,
                    segment_id=seg.segment_id,
                    reason="ours_v10_sota_gap",
                    eval_metrics=eval_metrics,
                    extra={},
                )
                sys.exit(42)

    final = segment_metrics_rows[-1] if segment_metrics_rows else {}
    return {
        "run_id": run_paths.run_id,
        "mode": "ours",
        "final": final,
        "drift_quality": _summarize_drift_proxy_quality(stream, drift),
        "routing_quality": (
            (last_eval_metrics.get("extra", {}) if isinstance(last_eval_metrics.get("extra", {}), dict) else {}).get("routing", {})
            if last_eval_metrics
            else {}
        ),
    }


def _build_baseline_method(baseline_name: str, cfg: Dict[str, Any]) -> Any:
    if baseline_name == "sequential_lora":
        from baselines.basic_baselines.sequential_lora.method import SequentialLoRAMethod

        return SequentialLoRAMethod(cfg)
    if baseline_name == "replay_lora":
        from baselines.basic_baselines.replay_lora.method import ReplayLoRAMethod

        return ReplayLoRAMethod(cfg)
    if baseline_name == "periodic_multilora":
        from baselines.basic_baselines.periodic_multilora.method import PeriodicMultiLoRAMethod

        return PeriodicMultiLoRAMethod(cfg)
    if baseline_name == "router_only":
        from baselines.basic_baselines.router_only.method import RouterOnlyMethod

        return RouterOnlyMethod(cfg)
    if baseline_name == "bank_no_router":
        # Implemented directly in core/train.py to avoid creating a new baseline folder.
        return object()
    if baseline_name == "o_lora":
        from baselines.advanced_baselines.o_lora.method import OLoraMethod

        return OLoraMethod(cfg)
    if baseline_name == "lb_cl":
        from baselines.advanced_baselines.lb_cl.method import LBCLMethod

        return LBCLMethod(cfg)
    if baseline_name == "progressive_prompts":
        from baselines.advanced_baselines.progressive_prompts.method import ProgressivePromptsMethod

        return ProgressivePromptsMethod(cfg)
    if baseline_name == "continual_t0":
        from baselines.advanced_baselines.continual_t0.method import ContinualT0Method

        return ContinualT0Method(cfg)
    raise ValueError(
        "Unknown baseline_name. Expected one of: "
        "sequential_lora | replay_lora | periodic_multilora | router_only | bank_no_router | "
        "o_lora | lb_cl | progressive_prompts | continual_t0"
    )


def _train_on_active_branch(
    *,
    segment: Segment,
    model: Any,
    lora: Any,
    lora_bank: Optional[LoRABank],
    lr: float,
    epochs: int,
    batch_size: int,
    use_overlap: bool,
    beta: float,
    overlap_cfg: Optional[Dict[str, Any]] = None,
    train_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from baselines.basic_baselines.sequential_lora.method import _batch

    pairs = [(ex.instruction, ex.input) for ex in segment.train]
    targets = [ex.output for ex in segment.train]

    batch_accs: List[float] = []
    batch_losses: List[float] = []
    batch_ans_accs: List[float] = []
    grad_norms: List[float] = []
    delta_norms: List[float] = []
    overlap_metric_sums: Dict[str, float] = {}
    supervision_metric_sums: Dict[str, float] = {}
    supervision_metric_steps = 0
    overlap_steps = 0
    total_tokens = 0
    supervised_tokens = 0
    batches = 0
    accum_steps = max(1, int((train_cfg or {}).get("gradient_accumulation_steps", 1)))
    pending_accum = 0
    optimizer_steps = 0
    for _ in range(max(1, epochs)):
        for b_pairs, b_targets in _batch(pairs, targets, batch_size):
            out = model.fit_batch(b_pairs, b_targets, lr=lr)
            pending_accum += 1

            # Anti-overlap regularization integrated into the training step.
            if (
                use_overlap
                and lora_bank is not None
                and beta > 0
                and hasattr(model, "get_activations_tensor")
                and len(lora_bank.list_branches()) > 1
            ):
                tok = getattr(model, "tokenizer", None)
                if tok is not None:
                    prompts = [format_for_infer(tok, ins, inp) for (ins, inp) in b_pairs]
                else:
                    prompts = [f"{ins}\n\n{inp}" for (ins, inp) in b_pairs]
                active_adapter = lora.get_active_adapter_name()
                activations_by_branch = {}
                for b in lora_bank.list_branches():
                    lora.set_active_adapter(b)
                    with_grad = b == active_adapter
                    acts = model.get_activations_tensor(prompts, with_grad=with_grad)
                    if not with_grad:
                        acts = acts.detach()
                    activations_by_branch[b] = acts
                lora.set_active_adapter(active_adapter)
                total_loss, overlap_metrics = compute_anti_overlap_training_loss(
                    activations_by_branch=activations_by_branch,
                    lora_wrapper=lora,
                    cfg={**(overlap_cfg or {}), "beta": beta},
                    segment_id=segment.segment_id,
                    active_adapter=active_adapter,
                )
                total_loss.backward()
                for key, value in overlap_metrics.items():
                    overlap_metric_sums[key] = overlap_metric_sums.get(key, 0.0) + float(value)
                overlap_steps += 1

            step_stats: Dict[str, Any] = {}
            if pending_accum >= accum_steps:
                if hasattr(lora, "scale_active_gradients"):
                    lora.scale_active_gradients(1.0 / float(pending_accum))
                step_stats = lora.step_adapter()
                optimizer_steps += 1
                pending_accum = 0
            batch_accs.append(float(out.get("train_batch_acc", 0.0)))
            batch_losses.append(float(out.get("train_loss", 0.0)))
            batch_ans_accs.append(float(out.get("train_answer_token_acc", 0.0)))
            for key, value in out.items():
                if str(key).startswith("train.") and str(key) not in {"train.loss", "train.answer_token_acc"}:
                    try:
                        supervision_metric_sums[str(key)] = supervision_metric_sums.get(str(key), 0.0) + float(value)
                    except (TypeError, ValueError):
                        continue
            supervision_metric_steps += 1
            if step_stats:
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                delta_norms.append(float(step_stats.get("lora_param_delta_l2", 0.0)))
            total_tokens += int(out.get("num_total_tokens", 0))
            supervised_tokens += int(out.get("num_supervised_tokens", 0))
            batches += 1
    if pending_accum > 0:
        if hasattr(lora, "scale_active_gradients"):
            lora.scale_active_gradients(1.0 / float(pending_accum))
        step_stats = lora.step_adapter()
        optimizer_steps += 1
        grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
        delta_norms.append(float(step_stats.get("lora_param_delta_l2", 0.0)))
    metrics = {
        "batches": batches,
        "optimizer_steps": optimizer_steps,
        "gradient_accumulation_steps": accum_steps,
        "mean_batch_acc": sum(batch_accs) / max(1, len(batch_accs)),
        "train.loss": sum(batch_losses) / max(1, len(batch_losses)),
        "train.answer_token_acc": sum(batch_ans_accs) / max(1, len(batch_ans_accs)),
        "num_total_tokens": int(total_tokens),
        "num_supervised_tokens": int(supervised_tokens),
        "grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
        "lora_param_delta_l2": sum(delta_norms) / max(1, len(delta_norms)),
        "lr": float(lr),
    }
    if overlap_steps > 0:
        metrics.update({k: v / float(overlap_steps) for k, v in overlap_metric_sums.items()})
        metrics["anti_overlap_steps"] = int(overlap_steps)
    if supervision_metric_steps > 0:
        metrics.update({k: v / float(supervision_metric_steps) for k, v in supervision_metric_sums.items()})
    return metrics


def _maybe_train_dialogue_replay_warmup(
    *,
    segment: Segment,
    model: Any,
    lora: Any,
    lora_bank: LoRABank,
    spectral_replay: Optional[SpectralSparseReplayGate],
    train_cfg: Dict[str, Any],
    lr: float,
    batch_size: int,
    logger: SimpleLogger,
) -> Dict[str, Any]:
    cfg = train_cfg.get("dialogue_replay_warmup", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return {}
    patterns = cfg.get("segment_name_patterns", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    segment_name = str(segment.segment_name or "")
    if patterns and not any(re.search(str(p), segment_name, flags=re.IGNORECASE) for p in patterns):
        return {}
    if spectral_replay is None or not hasattr(spectral_replay, "retrieve_dialogue_examples"):
        return {}

    replay_patterns = cfg.get("replay_segment_name_patterns", ["task1714", "task565", "dialogue", "generation"])
    if isinstance(replay_patterns, str):
        replay_patterns = [replay_patterns]
    prefixes = cfg.get("speaker_prefixes", ["agent", "customer"])
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    examples, retrieval_metrics = spectral_replay.retrieve_dialogue_examples(
        segment,
        name_patterns=replay_patterns,
        max_examples=max(0, int(cfg.get("max_examples", 96))),
        min_overlap=max(0, int(cfg.get("min_input_overlap", 1))),
        speaker_prefixes=prefixes,
        allow_prefixless_fallback=bool(cfg.get("allow_prefixless_fallback", False)),
        fallback_min_target_words=max(1, int(cfg.get("fallback_min_target_words", 4))),
    )

    active_branch = lora_bank.get_active_branch()
    metrics: Dict[str, Any] = {
        "dialogue_replay_warmup_enabled": True,
        "dialogue_replay_warmup_branch": active_branch,
        **{f"dialogue_replay_warmup.{k}": v for k, v in retrieval_metrics.items()},
    }
    if not examples:
        logger.log(f"Dialogue replay warmup skipped: {json.dumps(metrics, ensure_ascii=False)}")
        return metrics

    if active_branch in lora.list_adapters():
        lora_bank.unfreeze_branch(active_branch)
        lora.set_active_adapter(active_branch)
    warmup_segment = Segment(
        segment_id=segment.segment_id,
        segment_name=f"{segment.segment_name}_dialogue_replay_warmup",
        train=examples,
        eval=[],
    )
    train_metrics = _train_on_active_branch(
        segment=warmup_segment,
        model=model,
        lora=lora,
        lora_bank=lora_bank,
        lr=float(lr) * float(cfg.get("lr_scale", 0.75)),
        epochs=max(1, int(cfg.get("epochs", 1))),
        batch_size=max(1, int(cfg.get("batch_size", batch_size))),
        use_overlap=False,
        beta=0.0,
        overlap_cfg={},
        train_cfg=train_cfg,
    )
    metrics.update(
        {
            "dialogue_replay_warmup_examples": int(len(examples)),
            "dialogue_replay_warmup_batches": int(train_metrics.get("batches", 0)),
            "dialogue_replay_warmup_loss": float(train_metrics.get("train.loss", 0.0)),
            "dialogue_replay_warmup_answer_token_acc": float(train_metrics.get("train.answer_token_acc", 0.0)),
        }
    )
    logger.log(f"Dialogue replay warmup: {json.dumps(metrics, ensure_ascii=False)}")
    return metrics


def _extract_router_features(
    *,
    model: Any,
    lora: Any,
    lora_bank: LoRABank,
    router: Router,
    prompts: List[str],
) -> tuple[Any, str]:
    import torch

    feature_adapter = (
        router.feature_adapter_name
        if hasattr(lora_bank, "has_adapter") and lora_bank.has_adapter(router.feature_adapter_name)
        else lora.get_active_adapter_name()
    )
    active_before = lora.get_active_adapter_name()
    if feature_adapter in lora.list_adapters():
        lora.set_active_adapter(feature_adapter)
    try:
        if hasattr(model, "get_activations_tensor"):
            # Process in mini-batches to avoid OOM
            import torch
            batch_size = 4
            all_features = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_features = model.get_activations_tensor(batch_prompts, with_grad=False)
                all_features.append(batch_features)
            features = torch.cat(all_features, dim=0)
        else:
            features = torch.tensor(model.get_activations(prompts), dtype=torch.float32)
    finally:
        if active_before in lora.list_adapters():
            lora.set_active_adapter(active_before)
    return features, feature_adapter


def _maybe_init_spawn_prototype_from_anchors(
    *,
    router: Optional[Router],
    model: Any,
    lora: Any,
    lora_bank: LoRABank,
    new_branch: str,
    drift_anchor_set: Optional[AnchorSet],
) -> Optional[Dict[str, Any]]:
    if router is None or drift_anchor_set is None or not getattr(router, "spawn_sync_prototype_init", False):
        return None
    anchor_items = list(drift_anchor_set.core) + list(drift_anchor_set.probe)
    if not anchor_items:
        return None
    tok = getattr(model, "tokenizer", None)
    if tok is not None:
        prompts = [
            format_for_infer(tok, item.instruction, item.input_text, add_generation_prompt=True)
            for item in anchor_items
        ]
    else:
        prompts = [f"{item.instruction}\n\n{item.input_text}" for item in anchor_items]
    active_before = lora.get_active_adapter_name()
    if new_branch in lora.list_adapters():
        lora.set_active_adapter(new_branch)
    try:
        features, feature_adapter = _extract_router_features(
            model=model,
            lora=lora,
            lora_bank=lora_bank,
            router=router,
            prompts=prompts,
        )
        ok = router.init_prototype_from_anchor_features(
            new_branch,
            features,
            sample_count=len(anchor_items),
        )
    finally:
        if active_before in lora.list_adapters():
            lora.set_active_adapter(active_before)
    if not ok:
        return None
    return {
        "spawn_sync_prototype_branch": new_branch,
        "spawn_sync_prototype_samples": int(len(anchor_items)),
        "spawn_sync_feature_adapter": feature_adapter,
    }


def _anchor_prompts_from_set(anchor_set: AnchorSet, tok: Any) -> List[str]:
    anchor_items = list(anchor_set.core) + list(anchor_set.probe)
    if tok is not None:
        return [
            format_for_infer(tok, item.instruction, item.input_text, add_generation_prompt=True)
            for item in anchor_items
        ]
    return [f"{item.instruction}\n\n{item.input_text}" for item in anchor_items]


def _maybe_refresh_segment_prototypes_from_anchors(
    *,
    router: Optional[Router],
    model: Any,
    lora: Any,
    lora_bank: LoRABank,
    drift_anchor_set: Optional[AnchorSet],
) -> Optional[Dict[str, Any]]:
    if router is None or drift_anchor_set is None or not getattr(router, "segment_anchor_prototype_refresh", False):
        return None
    anchor_items = list(drift_anchor_set.core) + list(drift_anchor_set.probe)
    if not anchor_items:
        return None
    tok = getattr(model, "tokenizer", None)
    prompts = _anchor_prompts_from_set(drift_anchor_set, tok)
    refreshed = 0
    active_before = lora.get_active_adapter_name()
    feature_adapter = ""
    for branch in lora_bank.list_branches():
        if lora_bank.is_branch_frozen(branch):
            continue
        if branch not in lora.list_adapters():
            continue
        lora.set_active_adapter(branch)
        try:
            features, feature_adapter = _extract_router_features(
                model=model,
                lora=lora,
                lora_bank=lora_bank,
                router=router,
                prompts=prompts,
            )
            if router.refresh_prototype_from_anchor_features(
                branch,
                features,
                sample_count=len(anchor_items),
            ):
                refreshed += 1
        finally:
            pass
    if active_before in lora.list_adapters():
        lora.set_active_adapter(active_before)
    if refreshed == 0:
        return None
    return {
        "segment_anchor_proto_refreshed": int(refreshed),
        "segment_anchor_proto_samples": int(len(anchor_items)),
        "segment_anchor_proto_feature_adapter": feature_adapter,
    }


def _update_router_with_segment_pseudo_labels(
    *,
    segment: Segment,
    model: Any,
    lora: Any,
    lora_bank: LoRABank,
    router: Router,
) -> Dict[str, Any]:
    import torch

    branch_names = lora_bank.list_branches()
    tok = getattr(model, "tokenizer", None)
    pairs = [(ex.instruction, ex.input) for ex in segment.train]
    targets = [ex.output for ex in segment.train]
    prompts = [
        format_for_infer(tok, ins, inp, add_generation_prompt=True) if tok is not None else f"{ins}\n\n{inp}"
        for (ins, inp) in pairs
    ]
    features, feature_adapter = _extract_router_features(
        model=model,
        lora=lora,
        lora_bank=lora_bank,
        router=router,
        prompts=prompts,
    )
    frozen_branches = [b for b in branch_names if lora_bank.is_branch_frozen(b)]
    if len(branch_names) <= 1:
        pseudo_labels = [branch_names[0]] * len(pairs)
        router_metrics = router.update_with_pseudo_labels(
            features=features,
            pseudo_labels=pseudo_labels,
            branch_names=branch_names,
            frozen_branches=frozen_branches,
        )
        if hasattr(router, "record_segment_assignment"):
            router.record_segment_assignment(segment.segment_id, branch_names[0])
        return {
            "router_feature_adapter": feature_adapter,
            "router_num_candidates": int(len(segment.train)),
            "router_num_labels": router_metrics.get("num_router_labels", 0),
            "router_num_skipped_low_margin": 0,
            "router_mean_margin": 0.0,
            "router_loss": router_metrics.get("router_loss", 0.0),
            "router_train_acc": router_metrics.get("router_train_acc", 0.0),
        }

    active_before = lora.get_active_adapter_name()
    loss_by_branch: Dict[str, List[float]] = {}
    try:
        for branch_name in branch_names:
            lora.set_active_adapter(branch_name)
            loss_by_branch[branch_name] = model.score_answer_nlls(pairs, targets)
    finally:
        if active_before in lora.list_adapters():
            lora.set_active_adapter(active_before)

    pseudo_labels: List[str] = []
    kept_indices: List[int] = []
    margins: List[float] = []
    for i in range(len(pairs)):
        ranked = sorted((float(loss_by_branch[b][i]), b) for b in branch_names)
        best_loss, best_branch = ranked[0]
        second_loss = ranked[1][0] if len(ranked) > 1 else best_loss
        margin = float(second_loss - best_loss)
        
        base_margin = float(getattr(router, "margin_filter_min_gap", 0.0))
        dynamic_margin = min(0.02, base_margin + (getattr(router, "_num_updates", 0) / 50.0) * 0.02)
        
        if len(ranked) > 1 and margin < dynamic_margin:
            continue
        pseudo_labels.append(best_branch)
        kept_indices.append(i)
        margins.append(margin)

    if not kept_indices:
        if hasattr(router, "record_segment_assignment"):
            router.record_segment_assignment(segment.segment_id, lora_bank.get_active_branch())
        return {
            "router_feature_adapter": feature_adapter,
            "router_num_candidates": int(len(pairs)),
            "router_num_labels": 0,
            "router_num_skipped_low_margin": int(len(pairs)),
            "router_mean_margin": 0.0,
            "router_loss": 0.0,
            "router_train_acc": 0.0,
        }

    if isinstance(features, torch.Tensor):
        kept_features = features[kept_indices]
    else:
        kept_features = [features[i] for i in kept_indices]

    update_metrics = router.update_with_pseudo_labels(
        features=kept_features,
        pseudo_labels=pseudo_labels,
        branch_names=branch_names,
        frozen_branches=frozen_branches,
    )
    if hasattr(router, "record_segment_assignment") and pseudo_labels:
        counts: Dict[str, int] = {}
        for label in pseudo_labels:
            counts[label] = counts.get(label, 0) + 1
        majority_branch = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        router.record_segment_assignment(segment.segment_id, majority_branch)
    return {
        "router_feature_adapter": feature_adapter,
        "router_num_candidates": int(len(pairs)),
        "router_num_labels": int(len(pseudo_labels)),
        "router_num_skipped_low_margin": int(len(pairs) - len(pseudo_labels)),
        "router_mean_margin": float(sum(margins) / max(1, len(margins))),
        **update_metrics,
    }


def _eval_oracle_agreement(eval_metrics: Optional[Dict[str, Any]]) -> float:
    if not eval_metrics:
        return 1.0
    extra = eval_metrics.get("extra", {}) if isinstance(eval_metrics.get("extra"), dict) else {}
    routing = extra.get("routing", {}) if isinstance(extra.get("routing"), dict) else {}
    return float(routing.get("oracle_agreement_rate", 0.0) or 0.0)


def _train_with_router(
    *,
    segment: Segment,
    model: Any,
    lora: Any,
    lora_bank: LoRABank,
    router: Router,
    lr: float,
    epochs: int,
    batch_size: int,
    use_overlap: bool,
    beta: float,
    overlap_cfg: Optional[Dict[str, Any]] = None,
    prev_eval_metrics: Optional[Dict[str, Any]] = None,
    train_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    strategy = str(getattr(router, "training_strategy", "active_branch") or "active_branch")
    if strategy == "active_branch":
        train_metrics = _train_on_active_branch(
            segment=segment,
            model=model,
            lora=lora,
            lora_bank=lora_bank,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            use_overlap=use_overlap,
            beta=beta,
            overlap_cfg=overlap_cfg,
            train_cfg=train_cfg,
        )
        train_metrics["router_training_strategy"] = strategy
        train_metrics["routed_train_examples"] = 0
    elif strategy in {"oracle_min_nll", "learned_router", "ema"}:
        train_metrics = _train_with_routed_assignments(
            segment=segment,
            model=model,
            lora=lora,
            lora_bank=lora_bank,
            router=router,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            use_overlap=use_overlap,
            beta=beta,
            strategy=strategy,
            overlap_cfg=overlap_cfg,
            train_cfg=train_cfg,
        )
    else:
        raise ValueError(f"Unknown router.training_strategy: {strategy}. Expected one of: active_branch | oracle_min_nll | learned_router")
    proto_steps = max(1, int(getattr(router, "prototype_update_steps", 1)))
    pll_bonus_steps = 0
    pll_prev_oracle = _eval_oracle_agreement(prev_eval_metrics)
    if (
        bool(getattr(router, "oracle_pll_recalibrate", False))
        and segment.segment_id >= 3
        and len(lora_bank.list_branches()) >= 2
        and pll_prev_oracle < float(getattr(router, "oracle_pll_min_agreement", 0.55))
    ):
        pll_bonus_steps = max(0, int(getattr(router, "oracle_pll_bonus_steps", 2)))

    router_metrics: Dict[str, Any] = {}
    total_proto_steps = proto_steps + pll_bonus_steps
    saved_proto_ema = float(getattr(router, "prototype_ema", 0.8))
    if pll_bonus_steps > 0:
        router.prototype_ema = float(getattr(router, "oracle_pll_ema_override", 0.72))
        train_metrics["router_pll_recalibrate"] = True
        train_metrics["router_pll_prev_oracle_agreement"] = float(pll_prev_oracle)
    for proto_step in range(total_proto_steps):
        router_metrics = _update_router_with_segment_pseudo_labels(
            segment=segment,
            model=model,
            lora=lora,
            lora_bank=lora_bank,
            router=router,
        )
    if pll_bonus_steps > 0:
        router.prototype_ema = saved_proto_ema
    training_assignment = str(train_metrics.get("task_aware_fallback_training_branch", "") or "")
    if (
        training_assignment
        and bool(getattr(router, "task_aware_fallback_force_assigned", False))
        and hasattr(router, "record_segment_assignment")
    ):
        # Keep forced task-aware eval aligned to the branch that actually received
        # the segment's training updates; prototype pseudo-label refresh is only
        # a router fit signal and can be noisy on early CITB segments.
        router.record_segment_assignment(segment.segment_id, training_assignment)
        train_metrics["task_aware_fallback_assignment_after_proto"] = training_assignment
        train_metrics["task_aware_fallback_assignment_preserved"] = True
    train_metrics["router_prototype_update_steps"] = int(total_proto_steps)
    train_metrics["router_pll_bonus_steps"] = int(pll_bonus_steps)
    train_metrics.update(router_metrics)
    train_metrics["num_branches"] = len(lora_bank.list_branches())
    train_metrics["num_trainable_branches"] = len(lora_bank.list_trainable_branches())
    return train_metrics


def _train_with_routed_assignments(
    *,
    segment: Segment,
    model: Any,
    lora: Any,
    lora_bank: LoRABank,
    router: Router,
    lr: float,
    epochs: int,
    batch_size: int,
    use_overlap: bool,
    beta: float,
    strategy: str,
    overlap_cfg: Optional[Dict[str, Any]] = None,
    train_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from baselines.basic_baselines.sequential_lora.method import _batch

    print("Entering _train_with_routed_assignments...", file=sys.stderr)
    segment, slot_plan_metrics = _maybe_add_slot_aware_content_planning_examples(
        segment=segment,
        train_cfg=train_cfg or {},
    )
    segment, target_slot_metrics = _maybe_add_target_slot_alignment_examples(
        segment=segment,
        train_cfg=train_cfg or {},
    )
    pairs = [(ex.instruction, ex.input) for ex in segment.train]
    targets = [ex.output for ex in segment.train]
    branch_names = lora_bank.list_branches()
    active_branch = lora_bank.get_active_branch()
    print("Calling _assign_training_branches...", file=sys.stderr)
    assignments, assignment_metrics = _assign_training_branches(
        segment=segment,
        pairs=pairs,
        targets=targets,
        model=model,
        lora=lora,
        lora_bank=lora_bank,
        router=router,
        branch_names=branch_names,
        active_branch=active_branch,
        strategy=strategy,
    )
    print("Done _assign_training_branches...", file=sys.stderr)

    branch_to_examples: Dict[str, List[int]] = {}
    for idx, branch_name in enumerate(assignments):
        branch_to_examples.setdefault(branch_name, []).append(idx)
    balance_metrics = _maybe_balance_generation_training_indices(
        segment=segment,
        branch_to_examples=branch_to_examples,
        train_cfg=train_cfg or {},
    )
    slot_rich_metrics = _maybe_emphasize_slot_rich_generation_indices(
        segment=segment,
        branch_to_examples=branch_to_examples,
        train_cfg=train_cfg or {},
    )
    if hasattr(router, "record_segment_assignment") and branch_to_examples:
        majority_training_branch = max(branch_to_examples.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
        router.record_segment_assignment(segment.segment_id, majority_training_branch)

    batch_accs: List[float] = []
    batch_losses: List[float] = []
    batch_ans_accs: List[float] = []
    grad_norms: List[float] = []
    delta_norms: List[float] = []
    overlap_metric_sums: Dict[str, float] = {}
    supervision_metric_sums: Dict[str, float] = {}
    supervision_metric_steps = 0
    overlap_steps = 0
    total_tokens = 0
    supervised_tokens = 0
    batches = 0
    accum_steps = max(1, int((train_cfg or {}).get("gradient_accumulation_steps", 1)))
    optimizer_steps = 0
    active_before = lora.get_active_adapter_name()
    try:
        for _ in range(max(1, epochs)):
            for branch_name in sorted(branch_to_examples):
                if branch_name not in lora.list_adapters():
                    continue
                lora.set_active_adapter(branch_name)
                idxs = branch_to_examples[branch_name]
                routed_pairs = [pairs[i] for i in idxs]
                routed_targets = [targets[i] for i in idxs]
                pending_accum = 0
                for b_pairs, b_targets in _batch(routed_pairs, routed_targets, batch_size):
                    print("Calling model.fit_batch...", file=sys.stderr)
                    out = model.fit_batch(b_pairs, b_targets, lr=lr)
                    print("Done model.fit_batch.", file=sys.stderr)
                    pending_accum += 1
                    if (
                        use_overlap
                        and beta > 0
                        and hasattr(model, "get_activations_tensor")
                        and len(lora_bank.list_branches()) > 1
                    ):
                        tok = getattr(model, "tokenizer", None)
                        if tok is not None:
                            prompts = [format_for_infer(tok, ins, inp) for (ins, inp) in b_pairs]
                        else:
                            prompts = [f"{ins}\n\n{inp}" for (ins, inp) in b_pairs]
                        active_adapter = lora.get_active_adapter_name()
                        activations_by_branch = {}
                        for b in lora_bank.list_branches():
                            lora.set_active_adapter(b)
                            with_grad = b == active_adapter
                            acts = model.get_activations_tensor(prompts, with_grad=with_grad)
                            if not with_grad:
                                acts = acts.detach()
                            activations_by_branch[b] = acts
                        lora.set_active_adapter(active_adapter)
                        total_routed = sum(len(idxs) for idxs in branch_to_examples.values())
                        branch_route_fractions = {
                            b: len(branch_to_examples.get(b, [])) / max(1, total_routed)
                            for b in lora_bank.list_branches()
                        }
                        total_loss, overlap_metrics = compute_anti_overlap_training_loss(
                            activations_by_branch=activations_by_branch,
                            lora_wrapper=lora,
                            cfg={
                                **(overlap_cfg or {}),
                                "beta": beta,
                                "branch_route_fractions": branch_route_fractions,
                            },
                            segment_id=segment.segment_id,
                            active_adapter=active_adapter,
                        )
                        total_loss.backward()
                        for key, value in overlap_metrics.items():
                            overlap_metric_sums[key] = overlap_metric_sums.get(key, 0.0) + float(value)
                        overlap_steps += 1

                    step_stats: Dict[str, Any] = {}
                    if pending_accum >= accum_steps:
                        if hasattr(lora, "scale_active_gradients"):
                            lora.scale_active_gradients(1.0 / float(pending_accum))
                        step_stats = lora.step_adapter()
                        optimizer_steps += 1
                        pending_accum = 0
                    batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                    batch_losses.append(float(out.get("train_loss", 0.0)))
                    batch_ans_accs.append(float(out.get("train_answer_token_acc", 0.0)))
                    for key, value in out.items():
                        if str(key).startswith("train.") and str(key) not in {"train.loss", "train.answer_token_acc"}:
                            try:
                                supervision_metric_sums[str(key)] = supervision_metric_sums.get(str(key), 0.0) + float(value)
                            except (TypeError, ValueError):
                                continue
                    supervision_metric_steps += 1
                    if step_stats:
                        grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                        delta_norms.append(float(step_stats.get("lora_param_delta_l2", 0.0)))
                    total_tokens += int(out.get("num_total_tokens", 0))
                    supervised_tokens += int(out.get("num_supervised_tokens", 0))
                    batches += 1
                if pending_accum > 0:
                    if hasattr(lora, "scale_active_gradients"):
                        lora.scale_active_gradients(1.0 / float(pending_accum))
                    step_stats = lora.step_adapter()
                    optimizer_steps += 1
                    grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                    delta_norms.append(float(step_stats.get("lora_param_delta_l2", 0.0)))
    finally:
        restore_branch = active_branch if active_branch in lora.list_adapters() else active_before
        if restore_branch in lora.list_adapters():
            lora.set_active_adapter(restore_branch)

    metrics = {
        "batches": batches,
        "optimizer_steps": optimizer_steps,
        "gradient_accumulation_steps": accum_steps,
        "mean_batch_acc": sum(batch_accs) / max(1, len(batch_accs)),
        "train.loss": sum(batch_losses) / max(1, len(batch_losses)),
        "train.answer_token_acc": sum(batch_ans_accs) / max(1, len(batch_ans_accs)),
        "num_total_tokens": int(total_tokens),
        "num_supervised_tokens": int(supervised_tokens),
        "grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
        "lora_param_delta_l2": sum(delta_norms) / max(1, len(delta_norms)),
        "lr": float(lr),
        "router_training_strategy": strategy,
        "routed_train_examples": int(len(assignments)),
        "task_aware_fallback_training_branch": max(branch_to_examples.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
        if branch_to_examples
        else "",
        **assignment_metrics,
        **balance_metrics,
        **slot_rich_metrics,
        **slot_plan_metrics,
        **target_slot_metrics,
    }
    if overlap_steps > 0:
        metrics.update({k: v / float(overlap_steps) for k, v in overlap_metric_sums.items()})
        metrics["anti_overlap_steps"] = int(overlap_steps)
    if supervision_metric_steps > 0:
        metrics.update({k: v / float(supervision_metric_steps) for k, v in supervision_metric_sums.items()})
    return metrics


def _maybe_balance_generation_training_indices(
    *,
    segment: Segment,
    branch_to_examples: Dict[str, List[int]],
    train_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = train_cfg.get("balanced_generation_sampling", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return {}

    patterns = cfg.get("segment_name_patterns", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    segment_name = str(segment.segment_name or "")
    if patterns and not any(re.search(str(p), segment_name, flags=re.IGNORECASE) for p in patterns):
        return {}
    if not segment.train:
        return {}

    current_instruction = segment.train[0].instruction
    bucket_names = [_normalize_generation_bucket_name(x) for x in cfg.get("buckets", ["no", "yes", "i", "other"])]
    bucket_names = [name for name in bucket_names if name]
    if "other" not in bucket_names:
        bucket_names.append("other")
    max_multiplier = max(1.0, float(cfg.get("max_total_multiplier", 1.5)))
    downsample_majority = bool(cfg.get("downsample_majority", True))

    original_counts: Dict[str, int] = {}
    balanced_counts: Dict[str, int] = {}
    original_total = 0
    balanced_total = 0

    for branch_name, idxs in list(branch_to_examples.items()):
        current_idxs = [i for i in idxs if segment.train[i].instruction == current_instruction]
        passthrough = [i for i in idxs if segment.train[i].instruction != current_instruction]
        if len(current_idxs) < 2:
            continue

        buckets: Dict[str, List[int]] = {name: [] for name in bucket_names}
        for idx in current_idxs:
            bucket = _generation_target_bucket(segment.train[idx].output, bucket_names)
            buckets.setdefault(bucket, []).append(idx)
        non_empty = {k: v for k, v in buckets.items() if v}
        if len(non_empty) < 2:
            continue

        branch_orig_counts = {k: len(v) for k, v in non_empty.items()}
        for k, v in branch_orig_counts.items():
            original_counts[k] = original_counts.get(k, 0) + int(v)
        original_total += len(current_idxs)

        target_per_bucket = max(len(v) for v in non_empty.values())
        capped_target = int(math.ceil(len(current_idxs) * max_multiplier / max(1, len(non_empty))))
        target_per_bucket = max(1, min(target_per_bucket, capped_target))

        balanced_by_bucket: Dict[str, List[int]] = {}
        for bucket in bucket_names:
            vals = list(non_empty.get(bucket, []))
            if not vals:
                continue
            if len(vals) >= target_per_bucket:
                balanced_vals = vals[:target_per_bucket] if downsample_majority else vals
            else:
                repeats = [vals[i % len(vals)] for i in range(target_per_bucket)]
                balanced_vals = repeats
            balanced_by_bucket[bucket] = balanced_vals
            balanced_counts[bucket] = balanced_counts.get(bucket, 0) + len(balanced_vals)

        interleaved: List[int] = []
        max_len = max((len(v) for v in balanced_by_bucket.values()), default=0)
        for pos in range(max_len):
            for bucket in bucket_names:
                vals = balanced_by_bucket.get(bucket, [])
                if pos < len(vals):
                    interleaved.append(vals[pos])
        balanced_total += len(interleaved)
        branch_to_examples[branch_name] = interleaved + passthrough

    if original_total == 0:
        return {}
    return {
        "balanced_generation_sampling_enabled": True,
        "balanced_generation_sampling_segment": segment_name,
        "balanced_generation_sampling_original_examples": int(original_total),
        "balanced_generation_sampling_examples": int(balanced_total),
        "balanced_generation_sampling_original_bucket_counts_json": json.dumps(original_counts, sort_keys=True),
        "balanced_generation_sampling_bucket_counts_json": json.dumps(balanced_counts, sort_keys=True),
    }


def _generation_target_bucket(target: str, bucket_names: List[str]) -> str:
    normalized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", str(target or "").strip().lower())
    first = normalized.split()[0] if normalized.split() else "other"
    return first if first in set(bucket_names) and first != "other" else "other"


def _maybe_emphasize_slot_rich_generation_indices(
    *,
    segment: Segment,
    branch_to_examples: Dict[str, List[int]],
    train_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = train_cfg.get("slot_rich_generation_sampling", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return {}

    patterns = cfg.get("segment_name_patterns", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    segment_name = str(segment.segment_name or "")
    if patterns and not any(re.search(str(p), segment_name, flags=re.IGNORECASE) for p in patterns):
        return {}
    if not segment.train:
        return {}

    current_instruction = segment.train[0].instruction
    min_score = max(1, int(cfg.get("min_slot_score", 2)))
    repeat_factor = max(1, int(cfg.get("repeat_factor", 3)))
    max_multiplier = max(1.0, float(cfg.get("max_total_multiplier", 2.0)))

    original_total = 0
    emphasized_total = 0
    rich_total = 0
    score_hist: Dict[str, int] = {}

    for branch_name, idxs in list(branch_to_examples.items()):
        current_idxs = [i for i in idxs if segment.train[i].instruction == current_instruction]
        passthrough = [i for i in idxs if segment.train[i].instruction != current_instruction]
        if not current_idxs:
            continue

        scored = [(i, _generation_slot_rich_score(segment.train[i].output)) for i in current_idxs]
        rich_idxs = [i for i, score in scored if score >= min_score]
        for _idx, score in scored:
            key = str(min(score, 5))
            score_hist[key] = score_hist.get(key, 0) + 1
        if not rich_idxs:
            continue

        original_total += len(current_idxs)
        rich_total += len(rich_idxs)
        target_total = min(
            int(math.ceil(len(current_idxs) * max_multiplier)),
            len(current_idxs) + len(rich_idxs) * (repeat_factor - 1),
        )
        emphasized = list(current_idxs)
        cursor = 0
        while len(emphasized) < target_total:
            emphasized.append(rich_idxs[cursor % len(rich_idxs)])
            cursor += 1
        branch_to_examples[branch_name] = emphasized + passthrough
        emphasized_total += len(emphasized)

    if original_total == 0:
        return {}
    return {
        "slot_rich_generation_sampling_enabled": True,
        "slot_rich_generation_sampling_segment": segment_name,
        "slot_rich_generation_sampling_original_examples": int(original_total),
        "slot_rich_generation_sampling_slot_rich_examples": int(rich_total),
        "slot_rich_generation_sampling_examples": int(emphasized_total),
        "slot_rich_generation_sampling_score_hist_json": json.dumps(score_hist, sort_keys=True),
    }


def _generation_slot_rich_score(target: str) -> int:
    text = str(target or "").lower()
    words = re.findall(r"[a-z0-9']+", text)
    if len(words) < 4:
        return 0
    score = 0
    if any(ch.isdigit() for ch in text):
        score += 1
    if re.search(r"\b\d{1,2}/\d{1,2}\b|\b\d{3,4}\b", text):
        score += 1
    slot_terms = {
        "airline",
        "airlines",
        "american",
        "delta",
        "frontier",
        "jetblue",
        "southwest",
        "ua",
        "united",
        "flight",
        "fare",
        "price",
        "class",
        "economy",
        "business",
        "connection",
        "connecting",
        "direct",
        "reservation",
        "booking",
        "booked",
        "cancel",
        "cancelled",
        "confirmed",
        "proceed",
        "name",
    }
    score += min(3, len(set(words) & slot_terms))
    if re.search(r"\b[A-Z]{3}\b", str(target or "")):
        score += 1
    return score


def _maybe_add_slot_aware_content_planning_examples(
    *,
    segment: Segment,
    train_cfg: Dict[str, Any],
) -> tuple[Segment, Dict[str, Any]]:
    cfg = train_cfg.get("slot_aware_content_planning", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return segment, {}

    patterns = cfg.get("segment_name_patterns", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    segment_name = str(segment.segment_name or "")
    if patterns and not any(re.search(str(p), segment_name, flags=re.IGNORECASE) for p in patterns):
        return segment, {}
    if not segment.train:
        return segment, {}

    min_score = max(1, int(cfg.get("min_slot_score", 2)))
    repeat_factor = max(1, int(cfg.get("repeat_factor", 1)))
    max_examples = max(0, int(cfg.get("max_examples", 160)))
    include_original_gold = bool(cfg.get("include_original_gold", False))

    added: List[Example] = []
    slot_hist: Dict[str, int] = {}
    for ex in segment.train:
        if len(added) >= max_examples:
            break
        score = _generation_slot_rich_score(ex.output)
        if score < min_score:
            continue
        planned, slots = _build_slot_aware_planned_response(ex.input, ex.output)
        if not planned:
            continue
        for key in slots:
            slot_hist[key] = slot_hist.get(key, 0) + 1
        targets = [planned]
        if include_original_gold and _normalize_planning_text(planned) != _normalize_planning_text(ex.output):
            targets.append(str(ex.output))
        for target in targets:
            for _ in range(repeat_factor):
                if len(added) >= max_examples:
                    break
                added.append(
                    Example(
                        instruction=ex.instruction,
                        input=ex.input,
                        output=target,
                        output_references=(target,),
                    )
                )
            if len(added) >= max_examples:
                break

    if not added:
        return segment, {}

    expanded = Segment(
        segment_id=segment.segment_id,
        segment_name=segment.segment_name,
        train=list(segment.train) + added,
        eval=segment.eval,
    )
    return expanded, {
        "slot_aware_content_planning_enabled": True,
        "slot_aware_content_planning_segment": segment_name,
        "slot_aware_content_planning_original_examples": int(len(segment.train)),
        "slot_aware_content_planning_added_examples": int(len(added)),
        "slot_aware_content_planning_slot_hist_json": json.dumps(slot_hist, sort_keys=True),
    }


def _normalize_planning_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _build_slot_aware_planned_response(input_text: str, target: str) -> tuple[str, List[str]]:
    source = str(input_text or "")
    gold = str(target or "").strip()
    speaker_match = re.match(r"\s*(agent|customer)\s*:\s*", gold, flags=re.IGNORECASE)
    if not speaker_match:
        return "", []
    speaker = speaker_match.group(1).lower()
    combined = f"{source}\n{gold}"

    slots: Dict[str, List[str]] = {}

    def add(key: str, values: List[Any]) -> None:
        cleaned: List[str] = []
        seen = set()
        for value in values:
            if isinstance(value, tuple):
                value = next((part for part in value if part), "")
            val = str(value or "").strip(" .,;:")
            if not val:
                continue
            low = val.lower()
            if low in seen:
                continue
            seen.add(low)
            cleaned.append(val)
        if cleaned:
            slots[key] = cleaned

    add("name", re.findall(r"\b(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", source))
    add("airport", re.findall(r"\b[A-Z]{3}\b", combined))
    add("date", re.findall(r"\b\d{1,2}/\d{1,2}\b", combined))
    add("flight", re.findall(r"\b(?:flight\s*)?(\d{3,4})\b", gold, flags=re.IGNORECASE))
    add(
        "airline",
        re.findall(r"\b(American Airlines|American|Delta|Frontier|JetBlue|Southwest|United|UA)\b", gold, flags=re.IGNORECASE),
    )
    add("fare", re.findall(r"\b(?:fare|price)\s+(?:is|of)?\s*(\d+)\b", gold, flags=re.IGNORECASE))
    add("class", re.findall(r"\b(economy|business|first)\s+class\b|\b(economy|business)\b", gold, flags=re.IGNORECASE))
    if re.search(r"\bcancel|cancellation|reservation\b", combined, flags=re.IGNORECASE):
        slots["intent"] = ["reservation"]
    if re.search(r"\bbook|booking|flight ticket|proceed\b", combined, flags=re.IGNORECASE):
        slots["intent"] = list(dict.fromkeys(slots.get("intent", []) + ["booking"]))

    slot_keys = [k for k, v in slots.items() if v]
    if len(slot_keys) < 2:
        return "", []

    parts: List[str] = []
    if slots.get("name"):
        parts.append(f"for {slots['name'][0]}")
    airports = slots.get("airport", [])
    if len(airports) >= 2:
        parts.append(f"from {airports[0]} to {airports[1]}")
    if slots.get("date"):
        parts.append(f"on {' and '.join(slots['date'][:2])}")
    if slots.get("airline"):
        parts.append(f"with {slots['airline'][0]}")
    if slots.get("flight"):
        parts.append(f"flight {slots['flight'][0]}")
    if slots.get("fare"):
        parts.append(f"fare {slots['fare'][0]}")
    if slots.get("class"):
        parts.append(f"{slots['class'][0]} class")

    if not parts:
        return "", []

    if speaker == "agent":
        prefix = "agent: "
        if "booking" in slots.get("intent", []):
            body = "I can help with the booking " + ", ".join(parts) + "."
        elif "reservation" in slots.get("intent", []):
            body = "I checked the reservation details " + ", ".join(parts) + "."
        else:
            body = "I will use these flight details " + ", ".join(parts) + "."
    else:
        prefix = "customer: "
        body = "Please keep these travel details " + ", ".join(parts) + "."
    return prefix + body, slot_keys


def _maybe_add_target_slot_alignment_examples(
    *,
    segment: Segment,
    train_cfg: Dict[str, Any],
) -> tuple[Segment, Dict[str, Any]]:
    cfg = train_cfg.get("target_slot_alignment", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return segment, {}

    patterns = cfg.get("segment_name_patterns", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    segment_name = str(segment.segment_name or "")
    if patterns and not any(re.search(str(p), segment_name, flags=re.IGNORECASE) for p in patterns):
        return segment, {}
    if not segment.train:
        return segment, {}

    min_slots = max(1, int(cfg.get("min_slot_count", 1)))
    repeat_factor = max(1, int(cfg.get("repeat_factor", 1)))
    max_examples = max(0, int(cfg.get("max_examples", 240)))
    max_values_per_slot = max(1, int(cfg.get("max_values_per_slot", 3)))

    added: List[Example] = []
    slot_hist: Dict[str, int] = {}
    target_only_hist: Dict[str, int] = {}
    source_overlap_hist: Dict[str, int] = {}
    for ex in segment.train:
        if len(added) >= max_examples:
            break
        aligned, slots, target_only_slots, source_overlap_slots = _build_target_slot_alignment_response(
            ex.input,
            ex.output,
            max_values_per_slot=max_values_per_slot,
        )
        if not aligned or len(slots) < min_slots:
            continue
        for key in slots:
            slot_hist[key] = slot_hist.get(key, 0) + 1
        for key in target_only_slots:
            target_only_hist[key] = target_only_hist.get(key, 0) + 1
        for key in source_overlap_slots:
            source_overlap_hist[key] = source_overlap_hist.get(key, 0) + 1
        for _ in range(repeat_factor):
            if len(added) >= max_examples:
                break
            added.append(
                Example(
                    instruction=ex.instruction,
                    input=ex.input,
                    output=aligned,
                    output_references=(aligned,),
                )
            )

    if not added:
        return segment, {}

    expanded = Segment(
        segment_id=segment.segment_id,
        segment_name=segment.segment_name,
        train=list(segment.train) + added,
        eval=segment.eval,
    )
    return expanded, {
        "target_slot_alignment_enabled": True,
        "target_slot_alignment_segment": segment_name,
        "target_slot_alignment_original_examples": int(len(segment.train)),
        "target_slot_alignment_added_examples": int(len(added)),
        "target_slot_alignment_slot_hist_json": json.dumps(slot_hist, sort_keys=True),
        "target_slot_alignment_target_only_hist_json": json.dumps(target_only_hist, sort_keys=True),
        "target_slot_alignment_source_overlap_hist_json": json.dumps(source_overlap_hist, sort_keys=True),
    }


def _build_target_slot_alignment_response(
    input_text: str,
    target: str,
    *,
    max_values_per_slot: int,
) -> tuple[str, List[str], List[str], List[str]]:
    source_slots = _extract_dialogue_slot_values(str(input_text or ""))
    target_slots = _extract_dialogue_slot_values(str(target or ""))
    speaker_match = re.match(r"\s*(agent|customer)\s*:\s*", str(target or ""), flags=re.IGNORECASE)
    speaker = speaker_match.group(1).lower() if speaker_match else "response"

    ordered_keys = [
        "name",
        "airport",
        "date",
        "flight",
        "airline",
        "fare",
        "class",
        "booking",
        "connect",
    ]
    parts: List[str] = []
    slot_keys: List[str] = []
    target_only_keys: List[str] = []
    source_overlap_keys: List[str] = []
    for key in ordered_keys:
        values = target_slots.get(key, [])
        if not values:
            continue
        limited = values[:max_values_per_slot]
        source_canon = {_canonical_slot_value(v) for v in source_slots.get(key, [])}
        target_only = [v for v in limited if _canonical_slot_value(v) not in source_canon]
        overlapping = [v for v in limited if _canonical_slot_value(v) in source_canon]
        slot_keys.append(key)
        if target_only:
            target_only_keys.append(key)
        if overlapping:
            source_overlap_keys.append(key)
        parts.append(f"{key}=" + "/".join(str(v) for v in limited))

    if not parts:
        return "", [], [], []
    prefix = f"{speaker}: " if speaker in {"agent", "customer"} else ""
    aligned = prefix + "slot alignment " + "; ".join(parts) + "."
    return aligned, slot_keys, target_only_keys, source_overlap_keys


def _extract_dialogue_slot_values(text: str) -> Dict[str, List[str]]:
    raw = str(text or "")

    def unique(values: List[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            if isinstance(value, tuple):
                value = next((part for part in value if part), "")
            cleaned = str(value or "").strip(" .,;:")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out

    month = (
        "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec|"
        "January|February|March|April|May|June|July|August|September|October|November|December"
    )
    slots: Dict[str, List[str]] = {
        "airport": unique(re.findall(r"\b[A-Z]{3}\b", raw)),
        "date": unique(
            re.findall(
                rf"\b(?:\d{{1,2}}/\d{{1,2}}|(?:{month})\s*,?\s*\d{{1,2}}(?:st|nd|rd|th)?)\b",
                raw,
            )
        ),
        "flight": unique(
            [m.group(1) for m in re.finditer(r"\bflight\s*(?:number\s*)?(\d{3,4})\b", raw, re.IGNORECASE)]
        ),
        "fare": unique(
            [m.group(1) for m in re.finditer(r"\b(?:fare|price)\s*(?:is|of)?\s*(\d{2,5})\b", raw, re.IGNORECASE)]
        ),
        "airline": unique(
            re.findall(
                r"\b(?:American Airlines|American|Delta|Frontier|JetBlue|Southwest|United|UA)\b",
                raw,
                re.IGNORECASE,
            )
        ),
        "class": unique(
            [a or b for a, b in re.findall(r"\b(economy|business|first)\s+class\b|\b(economy|business)\b", raw, re.IGNORECASE)]
        ),
        "name": unique(
            [
                m.group(1)
                for m in re.finditer(
                    r"\b(?:my name is|i am|myself|name of|with the name of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
                    raw,
                )
            ]
        ),
        "booking": unique(
            re.findall(
                r"\b(?:booking|booked|reservation|reserved|confirmation|confirmed|ticket|cancelled|canceled|cancel)\b",
                raw,
                re.IGNORECASE,
            )
        ),
        "connect": unique(re.findall(r"\b(?:connecting|connection|direct|halt|break)\b", raw, re.IGNORECASE)),
    }
    return {key: values for key, values in slots.items() if values}


def _canonical_slot_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _normalize_generation_bucket_name(raw: Any) -> str:
    if isinstance(raw, bool):
        return "yes" if raw else "no"
    return str(raw).strip().lower()


def _resolve_trainable_training_branch(
    *,
    raw_branch: str,
    branch_names: List[str],
    lora_bank: LoRABank,
    active_branch: str,
    fallback_to_active: bool,
    ranked_branches: Optional[List[tuple[float, str]]] = None,
) -> str:
    """When oracle/router picks a frozen branch, train on the best trainable alternative."""
    if ranked_branches:
        for _loss, branch_name in ranked_branches:
            if branch_name in branch_names and not lora_bank.is_branch_frozen(branch_name):
                return branch_name
    trainable = [b for b in branch_names if not lora_bank.is_branch_frozen(b)]
    if fallback_to_active and active_branch in trainable:
        return active_branch
    if trainable:
        return trainable[0]
    return active_branch


def _assign_training_branches(
    *,
    segment: Segment,
    pairs: List[tuple[str, str]],
    targets: List[str],
    model: Any,
    lora: Any,
    lora_bank: LoRABank,
    router: Router,
    branch_names: List[str],
    active_branch: str,
    strategy: str,
) -> tuple[List[str], Dict[str, Any]]:
    print(f"Entering _assign_training_branches with strategy={strategy}...", file=sys.stderr)
    force_active_on_spawn = bool(getattr(router, "force_active_branch_on_spawn_segment", False))
    active_created_at_segment = None
    if active_branch in getattr(lora_bank, "_branches", {}):
        active_created_at_segment = int(lora_bank._branches[active_branch].created_at_segment)
    if (
        force_active_on_spawn
        and len(branch_names) > 1
        and active_branch in branch_names
        and not lora_bank.is_branch_frozen(active_branch)
        and active_created_at_segment == int(segment.segment_id)
    ):
        return [active_branch for _ in pairs], {
            "routed_train_num_branches": int(len(branch_names)),
            "routed_train_fallback_to_active": int(len(pairs)),
            "routed_train_branch_counts_json": json.dumps({active_branch: len(pairs)}, sort_keys=True),
            "routed_train_raw_branch_counts_json": json.dumps({active_branch: len(pairs)}, sort_keys=True),
            "routed_train_mean_margin": 0.0,
            "routed_train_force_active_on_spawn_segment": True,
            "routed_train_forced_active_branch": active_branch,
        }
    if len(branch_names) <= 1:
        print("Returning early because len(branch_names) <= 1", file=sys.stderr)
        return [active_branch for _ in pairs], {
            "routed_train_num_branches": int(len(branch_names)),
            "routed_train_fallback_to_active": int(len(pairs)),
            "routed_train_branch_counts_json": json.dumps({active_branch: len(pairs)}, sort_keys=True),
            "routed_train_mean_margin": 0.0,
        }

    raw_assignments: List[str] = []
    margins: List[float] = []
    loss_by_branch: Dict[str, List[float]] = {}
    if strategy == "oracle_min_nll":
        active_before = lora.get_active_adapter_name()
        try:
            for branch_name in branch_names:
                lora.set_active_adapter(branch_name)
                loss_by_branch[branch_name] = model.score_answer_nlls(pairs, targets)
        finally:
            if active_before in lora.list_adapters():
                lora.set_active_adapter(active_before)
        for i in range(len(pairs)):
            ranked = sorted((float(loss_by_branch[b][i]), b) for b in branch_names)
            best_loss, best_branch = ranked[0]
            second_loss = ranked[1][0] if len(ranked) > 1 else best_loss
            raw_assignments.append(best_branch)
            margins.append(float(second_loss - best_loss))
    elif strategy in {"learned_router", "ema"}:
        print("Extracting features...", file=sys.stderr)
        tok = getattr(model, "tokenizer", None)
        prompts = [
            format_for_infer(tok, ins, inp, add_generation_prompt=True) if tok is not None else f"{ins}\n\n{inp}"
            for (ins, inp) in pairs
        ]
        features, _feature_adapter = _extract_router_features(
            model=model,
            lora=lora,
            lora_bank=lora_bank,
            router=router,
            prompts=prompts,
        )
        print("Done extracting features. Now predicting...", file=sys.stderr)
        branch_meta = lora_bank.state_dict()
        for i, prompt in enumerate(prompts):
            feat_i = features[i : i + 1] if hasattr(features, "__getitem__") else [features[i]]
            decision = router.predict_branch(
                prompt=prompt,
                branch_names=branch_names,
                branch_meta=branch_meta,
                segment_id=segment.segment_id,
                features=feat_i,
            )
            raw_assignments.append(decision.branch_name)
            prob_scores = sorted(((float(v), k) for k, v in decision.scores.items()), reverse=True)
            if len(prob_scores) > 1:
                margins.append(float(prob_scores[0][0] - prob_scores[1][0]))
            else:
                margins.append(0.0)
    else:
        raise ValueError(f"Unsupported routed training strategy: {strategy}")

    assignments: List[str] = []
    fallback_count = 0
    train_frozen = bool(getattr(router, "train_frozen_branches", False))
    fallback_to_active = bool(getattr(router, "training_fallback_to_active", True))
    for idx, branch_name in enumerate(raw_assignments):
        target_branch = branch_name
        if (
            not train_frozen
            and branch_name in branch_names
            and lora_bank.is_branch_frozen(branch_name)
        ):
            resolved = _resolve_trainable_training_branch(
                raw_branch=branch_name,
                branch_names=branch_names,
                lora_bank=lora_bank,
                active_branch=active_branch,
                fallback_to_active=fallback_to_active,
                ranked_branches=(
                    sorted((float(loss_by_branch[b][idx]), b) for b in branch_names)
                    if strategy == "oracle_min_nll" and loss_by_branch
                    else None
                ),
            )
            target_branch = resolved
            fallback_count += 1
        if target_branch not in branch_names:
            target_branch = active_branch
            fallback_count += 1
        assignments.append(target_branch)

    branch_counts: Dict[str, int] = {}
    raw_branch_counts: Dict[str, int] = {}
    for branch_name in assignments:
        branch_counts[branch_name] = branch_counts.get(branch_name, 0) + 1
    for branch_name in raw_assignments:
        raw_branch_counts[branch_name] = raw_branch_counts.get(branch_name, 0) + 1

    return assignments, {
        "routed_train_num_branches": int(len(branch_names)),
        "routed_train_fallback_to_active": int(fallback_count),
        "routed_train_branch_counts_json": json.dumps(branch_counts, sort_keys=True),
        "routed_train_raw_branch_counts_json": json.dumps(raw_branch_counts, sort_keys=True),
        "routed_train_mean_margin": float(sum(margins) / max(1, len(margins))),
    }


def _train_bank_no_router(
    *,
    segment: Segment,
    model: Any,
    lora: Any,
    lora_bank: LoRABank,
    lr: float,
    epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Baseline mode: bank + no router.

    - Does NOT route per-example; always trains on the active branch
    - Drift monitoring / spawning is handled in `run_baseline(...)` so it can share
      the same anchor-based logging path as `run_ours(...)`
    - This baseline exists as baseline_name in config/code, without a new top-level folder.
    """

    if not lora_bank.list_branches():
        lora_bank.initialize(lora_wrapper=lora, initial_branch="b0", segment_id=segment.segment_id)

    # Train on current active branch
    train_metrics = _train_on_active_branch(
        segment=segment,
        model=model,
        lora=lora,
        lora_bank=lora_bank,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        use_overlap=False,
        beta=0.0,
    )

    train_metrics["num_branches"] = len(lora_bank.list_branches())
    train_metrics["active_branch"] = lora_bank.get_active_branch() if lora_bank.list_branches() else ""
    return train_metrics


def _write_stop_and_diagnose(
    *,
    run_paths: RunPaths,
    cfg: Dict[str, Any],
    segment_id: int,
    reason: str,
    eval_metrics: Dict[str, Any],
    extra: Dict[str, Any],
) -> None:
    """Persist early-stop context before exiting an incomplete strict run."""
    extra_metrics = eval_metrics.get("extra", {}) if isinstance(eval_metrics.get("extra"), dict) else {}
    routing = extra_metrics.get("routing", {}) if isinstance(extra_metrics.get("routing"), dict) else {}
    payload = {
        "recommendation": "stop_and_diagnose",
        "reason": reason,
        "segment_id": int(segment_id),
        "config_path": str(cfg.get("__config_path__", "")),
        "run_dir": str(run_paths.run_dir),
        "metrics": {
            "seen_avg_score": float(eval_metrics.get("seen_avg_score", 0.0)),
            "seen_avg_task_aware_score": float(eval_metrics.get("seen_avg_task_aware_score", 0.0)),
            "rouge_l_mean": float(eval_metrics.get("rouge_l_mean", 0.0)),
            "bleu_mean": float(eval_metrics.get("bleu_mean", 0.0)),
            "token_f1_mean": float(eval_metrics.get("token_f1_mean", 0.0)),
            "slot_error_rate": float(eval_metrics.get("slot_error_rate", 0.0)),
            "forgetting": float(eval_metrics.get("forgetting", 0.0)),
        },
        "routing": {
            "num_routed": int(routing.get("num_routed", 0) or 0),
            "oracle_agreement_rate": float(routing.get("oracle_agreement_rate", 0.0) or 0.0),
            "decision_confidence_mean": float(routing.get("decision_confidence_mean", 0.0) or 0.0),
            "decision_entropy_mean": float(routing.get("decision_entropy_mean", 0.0) or 0.0),
            "branch_utilization": routing.get("branch_utilization", {}),
        },
        "extra": extra,
    }
    out = Path(run_paths.run_dir) / "stop_and_diagnose.json"
    save_json(str(out), payload)


# v6_sota_2 seen_avg_score trajectory (s123) — reference for SOTA chase early stop.
_V6_SOTA_2_SEEN_AVG_TRAJECTORY: Dict[int, float] = {
    0: 0.0,
    1: 0.05,
    2: 0.0,
    3: 0.25,
    4: 0.34,
    5: 0.325,
    6: 0.426,
    7: 0.373,
    8: 0.378,
    9: 0.322,
    10: 0.274,
    11: 0.311,
    12: 0.295,
    13: 0.275,
    14: 0.297,
    15: 0.328,
    16: 0.35,
    17: 0.381,
    18: 0.35,
}


def _ours_v10_early_stop_triggered(
    cfg: Dict[str, Any],
    segment_id: int,
    eval_metrics: Dict[str, Any],
    logger: SimpleLogger,
) -> bool:
    """Exit 42 when OURS_V10_EARLY_STOP=1 and trajectory cannot reach SOTA margin."""
    import os

    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    env_on = os.environ.get("OURS_V10_EARLY_STOP", "").strip() in {"1", "true", "yes"}
    cfg_on = bool(train_cfg.get("ours_v10_early_stop", False))
    if not (env_on or cfg_on):
        return False

    check_segments = train_cfg.get("ours_v10_check_segments", [3, 4, 5])
    if segment_id not in {int(x) for x in check_segments}:
        return False

    margin = float(train_cfg.get("ours_v10_margin", 1.10))
    forget_margin = float(train_cfg.get("ours_v10_forget_margin", 0.90))
    baseline_path = str(train_cfg.get("ours_v10_baseline_csv", "")).strip()
    if not baseline_path:
        return False
    bl_row = _ours_v10_baseline_row_at_segment(Path(baseline_path), segment_id)
    if bl_row is None:
        return False

    ours_vals = {
        "seen_avg_score": float(eval_metrics.get("seen_avg_score", 0)),
        "seen_avg_task_aware_score": float(eval_metrics.get("seen_avg_task_aware_score", 0)),
        "token_f1_mean": float(eval_metrics.get("token_f1_mean", 0)),
        "forgetting": float(eval_metrics.get("forgetting", 0)),
    }
    bl_vals = {
        "seen_avg_score": float(bl_row.get("eval.seen_avg_score", 0)),
        "seen_avg_task_aware_score": float(bl_row.get("eval.seen_avg_task_aware_score", 0)),
        "token_f1_mean": float(bl_row.get("eval.token_f1_mean", 0)),
        "forgetting": float(bl_row.get("eval.forgetting", 0)),
    }
    failures: List[str] = []
    for key in ("seen_avg_score", "seen_avg_task_aware_score", "token_f1_mean"):
        target = bl_vals[key] * margin
        if ours_vals[key] < target:
            failures.append(f"{key} {ours_vals[key]:.4f} < {target:.4f}")
    forget_target = bl_vals["forgetting"] * forget_margin
    if ours_vals["forgetting"] > forget_target:
        failures.append(f"forgetting {ours_vals['forgetting']:.4f} > {forget_target:.4f}")

    if failures:
        logger.log(
            "OURS_V10 early stop: SOTA trajectory gap at segment "
            f"{segment_id}: " + "; ".join(failures)
        )
        return True
    return False


def _ours_v10_baseline_row_at_segment(csv_path: Path, segment_id: int) -> Optional[Dict[str, Any]]:
    if not csv_path.is_file():
        return None
    best: Optional[Dict[str, Any]] = None
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sid = int(row.get("segment_id", -1))
            except (TypeError, ValueError):
                continue
            if sid == segment_id:
                best = row
    return best


def _trajectory_early_stop_floor(cfg: Dict[str, Any], segment_id: int) -> Optional[float]:
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    if not bool(train_cfg.get("trajectory_early_stop", False)):
        return None
    margin = float(train_cfg.get("trajectory_margin", 0.05))
    baseline = str(train_cfg.get("trajectory_baseline", "v6_sota_2")).strip()
    traj_path = str(train_cfg.get("trajectory_metrics_path", "")).strip()
    ref: Dict[int, float] = {}
    if traj_path:
        p = Path(traj_path)
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row.get("segment_id")
                seen = row.get("eval.seen_avg_score", row.get("seen_avg_score"))
                if sid is not None and seen is not None:
                    ref[int(sid)] = float(seen)
    if not ref and baseline == "v6_sota_2":
        ref = _V6_SOTA_2_SEEN_AVG_TRAJECTORY
    if segment_id not in ref:
        return None
    return float(ref[segment_id]) - margin


def _flatten_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            flat[f"{prefix}.{k}"] = v
    return flat


def _load_overfit_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Reload partial overfit8_steps.csv for resume (typed columns)."""
    if not csv_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: Dict[str, Any] = {}
            for k, v in raw.items():
                if v is None or v == "":
                    continue
                try:
                    if k == "step" or k.endswith("_count"):
                        row[k] = int(float(v))
                    else:
                        row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def _run_overfit_8_mode(
    *,
    cfg: Dict[str, Any],
    stream: ContinualStream,
    backbone: Any,
    lora: Any,
    run_paths: RunPaths,
    logger: SimpleLogger,
) -> None:
    from baselines.basic_baselines.sequential_lora.method import _batch

    debug_tools = cfg.get("debug_tools", {}) if isinstance(cfg.get("debug_tools", {}), dict) else {}
    if not stream.stream:
        logger.log("overfit_8_mode skipped: empty stream.")
        return
    seg = stream.stream[int(debug_tools.get("overfit_segment_index", 0))]
    train_subset = seg.train[:8]
    if len(train_subset) < 1:
        logger.log("overfit_8_mode skipped: no train examples in selected segment.")
        return
    steps = int(debug_tools.get("overfit_steps", 200))
    log_every = int(debug_tools.get("overfit_log_every", 20))
    overfit_batch_size = int(debug_tools.get("overfit_batch_size", 1))
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    overfit_gen_max_new_tokens = int(debug_tools.get("overfit_gen_max_new_tokens", model_cfg.get("gen_max_new_tokens", 64)))
    lr = float((cfg.get("train", {}) or {}).get("lr", 2e-4))
    out_dir = Path(run_paths.run_dir) / "debug" / "overfit8"
    ensure_dir(str(out_dir))
    config_snapshot_path = Path(run_paths.run_dir) / "config_snapshot.yaml"
    manifest_path = out_dir / "run_manifest.json"
    run_manifest = init_overfit_run_manifest(
        cfg=cfg,
        run_id=run_paths.run_id,
        config_path=str(cfg.get("__config_path__", "")),
        config_snapshot_path=str(config_snapshot_path),
        run_dir=Path(run_paths.run_dir),
        debug_tools=debug_tools,
    )
    manifest_add_artifact(run_manifest, manifest_path)
    manifest_add_artifact(run_manifest, config_snapshot_path)

    def _flush_overfit_manifest() -> None:
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ckpt_root = out_dir / "lora_ckpt"
    state_path = out_dir / "overfit_state.json"
    csv_path = out_dir / "overfit8_steps.csv"
    resume = bool(debug_tools.get("overfit_resume", False))
    fresh = bool(debug_tools.get("overfit_fresh_start", False))
    decode_only = bool(debug_tools.get("overfit_decode_ablation_only", False))
    if fresh and not resume and not decode_only:
        stale = collect_overfit_stale_artifacts(out_dir)
        logger.log(f"[overfit8][fresh_start][dry_run] will remove {len(stale)} artifact(s):")
        for p in stale:
            logger.log(f"[overfit8][fresh_start][dry_run] - {p}")
        for p in stale:
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink()
            except Exception as e:
                logger.log(f"[overfit8][fresh_start] failed to remove {p}: {e}")

    snapshot_every = int(debug_tools.get("overfit_checkpoint_every", 0))
    if snapshot_every <= 0:
        snapshot_every = max(1, log_every)

    generation_ablation_out_csv = Path(run_paths.run_dir) / "debug" / "generation_prompt_ablation.csv"
    generation_ablation_out_cont_json = Path(run_paths.run_dir) / "debug" / "generation_prompt_ablation_cont_heads.json"

    normalization_cfg = cfg.get("eval_normalization", {}) if isinstance(cfg.get("eval_normalization", {}), dict) else {}

    start_step = 1
    per_step_rows: List[Dict[str, Any]] = []
    if resume and state_path.is_file():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            last_done = int(st.get("last_completed_step", 0))
            start_step = last_done + 1
            latest = ckpt_root / "latest"
            snap = ckpt_root / f"step_{last_done:06d}"
            load_path = latest if latest.is_dir() else snap
            if load_path.is_dir() and hasattr(lora, "load_adapter_checkpoint"):
                lora.load_adapter_checkpoint(str(load_path))
                logger.log(
                    f"[overfit8] Resume: loaded LoRA from {load_path} "
                    f"(last_completed_step={last_done}), continue from step {start_step}."
                )
            else:
                logger.log(
                    f"[overfit8] Resume requested but no adapter at {latest} or {snap}; "
                    "starting from step 1."
                )
                start_step = 1
        except Exception as e:
            logger.log(f"[overfit8] Resume parse/load failed ({e}); starting from step 1.")
            start_step = 1

    if start_step > 1 and csv_path.is_file():
        per_step_rows = [r for r in _load_overfit_csv_rows(csv_path) if int(r.get("step", 0)) < start_step]
        logger.log(f"[overfit8] Restored {len(per_step_rows)} CSV rows for steps < {start_step}.")
    elif start_step > 1 and not csv_path.is_file():
        logger.log(
            "[overfit8] Warning: resume without overfit8_steps.csv; metrics history will only "
            f"contain steps >= {start_step}."
        )

    if start_step > steps:
        logger.log(f"[overfit8] Nothing to do: resume start_step={start_step} > overfit_steps={steps}.")
        return

    hf_ok = getattr(backbone, "tokenizer", None) is not None and getattr(backbone, "model", None) is not None
    diag_enabled = hf_ok and bool(debug_tools.get("enable_sequence_behavior_diagnosis", False))
    # Staged diagnosis (YAML toggles). Steps 1→5: failure+first_token → prefix → decode → ladder.
    run_failure_decomposition = diag_enabled and bool(debug_tools.get("run_failure_decomposition", True))
    run_first_token_audit = diag_enabled and bool(debug_tools.get("run_first_token_audit", True))
    run_prefix_rollout = diag_enabled and bool(debug_tools.get("run_prefix_rollout", False))
    if run_prefix_rollout and not hasattr(backbone, "generate_with_forced_answer_prefix"):
        logger.log("[overfit8] run_prefix_rollout requested but backbone has no generate_with_forced_answer_prefix; skipping.")
        run_prefix_rollout = False
    run_decode_ablation = diag_enabled and bool(debug_tools.get("run_decode_ablation", False))
    run_overfit_ladder = diag_enabled and bool(debug_tools.get("run_overfit_ladder", False))
    ladder_steps = int(debug_tools.get("overfit_ladder_steps", steps))
    ladder_init_path = str(ckpt_root / "ladder_init_adapter")

    if diag_enabled:
        logger.log(
            "[overfit8] Sequence behavior diagnosis: "
            f"failure={run_failure_decomposition} first_token={run_first_token_audit} "
            f"prefix_rollout={run_prefix_rollout} decode_ablation={run_decode_ablation} "
            f"overfit_ladder={run_overfit_ladder}. "
            "(Open-loop vs teacher-forced gap; not mask/shift debugging.)"
        )

    if decode_only:
        if not diag_enabled:
            logger.log("[overfit8] decode_ablation_only: requires HF tokenizer+model and enable_sequence_behavior_diagnosis.")
            return
        if not run_decode_ablation:
            logger.log("[overfit8] decode_ablation_only: set run_decode_ablation: true in debug_tools.")
            return
        ckpt_rel = str(debug_tools.get("overfit_decode_ablation_ckpt", "lora_ckpt/step_000020"))
        ckpt_abs = (out_dir / ckpt_rel).resolve()
        if not ckpt_abs.is_dir():
            logger.log(f"[overfit8] decode_ablation_only: missing adapter directory {ckpt_abs}")
            return
        if not hasattr(lora, "load_adapter_checkpoint"):
            logger.log("[overfit8] decode_ablation_only: LoRA wrapper has no load_adapter_checkpoint.")
            return
        pairs = [(ex.instruction, ex.input) for ex in train_subset]
        lora.load_adapter_checkpoint(str(ckpt_abs))
        logger.log(
            f"[overfit8] decode_ablation_only: loaded {ckpt_abs}; "
            f"eval greedy/beam2/beam4 (do_sample=false), max_new_tokens={overfit_gen_max_new_tokens}"
        )
        from core.overfit_sequence_diagnostics import run_decode_ablation

        infer_prompts_final = [
            format_for_infer(backbone.tokenizer, ins, inp, add_generation_prompt=True) for (ins, inp) in pairs
        ]
        run_decode_ablation(
            out_dir=out_dir,
            examples=list(train_subset),
            infer_prompts=infer_prompts_final,
            backbone=backbone,
            max_new_tokens=overfit_gen_max_new_tokens,
            normalization_cfg=normalization_cfg,
        )
        manifest_add_artifact(run_manifest, out_dir / "decode_ablation.json")
        manifest_add_artifact(run_manifest, out_dir / "decode_ablation.csv")
        _flush_overfit_manifest()
        logger.log("[overfit8] Wrote decode_ablation.json / decode_ablation.csv (decode_ablation_only).")
        return

    logger.log(
        f"[overfit8] segment={seg.segment_id}:{seg.segment_name} examples={len(train_subset)} "
        f"steps={steps} log_every={log_every} start_step={start_step} resume={resume}"
    )

    pairs = [(ex.instruction, ex.input) for ex in train_subset]
    targets = [ex.output for ex in train_subset]

    if start_step == 1 and not resume and hasattr(lora, "save_adapter_checkpoint"):
        ensure_dir(str(ckpt_root))
        lora.save_adapter_checkpoint(ladder_init_path)
        manifest_add_artifact(run_manifest, Path(ladder_init_path))
        logger.log(f"[overfit8] Saved ladder_init adapter (pre-training) to {ladder_init_path}")

    if (
        getattr(backbone, "tokenizer", None) is not None
        and getattr(backbone, "model", None) is not None
        and start_step <= 1
    ):
        from core.debug_teacher_audit import dump_overfit_teacher_audit

        if bool(debug_tools.get("dump_teacher_forced_audit", True)):
            dump_overfit_teacher_audit(
                backbone=backbone,
                examples=list(train_subset),
                out_path=str(out_dir / "teacher_forced_audit_initial.json"),
                strict_assertions=bool(debug_tools.get("strict_teacher_metric_assertions", False)),
            )

    from core.debug_teacher_audit import mean_teacher_forced_over_examples
    from core.evaluate import _lcs_overlap, _normalize, _token_f1
    for step in range(start_step, steps + 1):
        # Train repeatedly on the same 8 examples in tiny batches to avoid OOM.
        step_loss_sum = 0.0
        step_ans_acc_sum = 0.0
        step_grad_sum = 0.0
        step_delta_sum = 0.0
        step_total_tokens = 0
        step_supervised_tokens = 0
        step_loss_tokens = 0
        step_inner_batches = 0
        for b_pairs, b_targets in _batch(pairs, targets, overfit_batch_size):
            out = backbone.fit_batch(b_pairs, b_targets, lr=lr)
            step_stats = lora.step_adapter()
            step_loss_sum += float(out.get("train_loss", 0.0))
            step_ans_acc_sum += float(out.get("train_answer_token_acc", 0.0))
            step_grad_sum += float(step_stats.get("grad_norm", 0.0))
            step_delta_sum += float(step_stats.get("lora_param_delta_l2", 0.0))
            step_total_tokens += int(out.get("num_total_tokens", 0))
            step_supervised_tokens += int(out.get("num_supervised_tokens", 0))
            step_loss_tokens += int(out.get("num_loss_tokens", 0))
            step_inner_batches += 1
        row = {
            "step": step,
            "train_loss": step_loss_sum / max(1, step_inner_batches),
            "train_answer_token_acc": step_ans_acc_sum / max(1, step_inner_batches),
            "num_total_tokens": int(step_total_tokens),
            "num_supervised_tokens": int(step_supervised_tokens),
            "num_loss_tokens": int(step_loss_tokens),
            "grad_norm": step_grad_sum / max(1, step_inner_batches),
            "lora_param_delta_l2": step_delta_sum / max(1, step_inner_batches),
            "lr": float(lr),
        }
        per_step_rows.append(row)

        if step % max(1, log_every) == 0 or step == 1 or step == steps:
            tok = getattr(backbone, "tokenizer", None)
            infer_prompts = (
                [format_for_infer(tok, ins, inp, add_generation_prompt=True) for (ins, inp) in pairs]
                if tok is not None
                else [f"{ins}\n\n{inp}" for (ins, inp) in pairs]
            )
            preds = backbone.generate(infer_prompts, max_new_tokens=overfit_gen_max_new_tokens)
            side_by_side = []
            exact_match_cnt = 0
            prefix1_match_cnt = 0
            prefix3_match_cnt = 0
            prefix5_match_cnt = 0
            token_f1_list: List[float] = []
            lcs_list: List[float] = []
            for i, (ex, pred, infer_prompt) in enumerate(zip(train_subset, preds, infer_prompts)):
                norm_pred = _normalize(pred or "", prompt=infer_prompt, cfg=normalization_cfg)
                norm_gold = _normalize(ex.output or "", prompt=infer_prompt, cfg=normalization_cfg)
                is_match = bool(norm_pred == norm_gold)
                if is_match:
                    exact_match_cnt += 1
                pred_tokens = [t for t in norm_pred.split() if t]
                gold_tokens = [t for t in norm_gold.split() if t]
                prefix1_match_cnt += int(pred_tokens[:1] == gold_tokens[:1])
                prefix3_match_cnt += int(pred_tokens[:3] == gold_tokens[:3])
                prefix5_match_cnt += int(pred_tokens[:5] == gold_tokens[:5])
                token_f1_list.append(float(_token_f1(norm_pred, norm_gold)))
                lcs_list.append(float(_lcs_overlap(norm_pred, norm_gold)))
                side_by_side.append(
                    {
                        "idx": i,
                        "instruction": ex.instruction,
                        "input": ex.input,
                        "gold_output": ex.output,
                        "formatted_infer_prompt": infer_prompt,
                        "generated_output": pred,
                        "normalized_prediction": norm_pred,
                        "normalized_gold": norm_gold,
                        "exact_match": is_match,
                        "prefix_1_match": bool(pred_tokens[:1] == gold_tokens[:1]),
                        "prefix_3_match": bool(pred_tokens[:3] == gold_tokens[:3]),
                        "prefix_5_match": bool(pred_tokens[:5] == gold_tokens[:5]),
                        "token_f1": float(token_f1_list[-1]),
                        "lcs_overlap": float(lcs_list[-1]),
                    }
                )
            tf_loss_m, tf_acc_m, _n_tf = mean_teacher_forced_over_examples(
                backbone=backbone, pairs=pairs, targets=targets
            )
            row["teacher_forced_loss_mean"] = float(tf_loss_m)
            row["teacher_forced_shifted_token_acc_mean"] = float(tf_acc_m)
            row["token_f1_mean"] = float(sum(token_f1_list) / max(1, len(token_f1_list)))
            row["lcs_overlap_mean"] = float(sum(lcs_list) / max(1, len(lcs_list)))
            save_json(str(out_dir / f"step_{step:04d}_predictions.json"), side_by_side)
            manifest_add_artifact(run_manifest, out_dir / f"step_{step:04d}_predictions.json")
            row["exact_match_count"] = int(exact_match_cnt)
            row["prefix1_acc"] = float(prefix1_match_cnt / max(1, len(side_by_side)))
            row["prefix3_acc"] = float(prefix3_match_cnt / max(1, len(side_by_side)))
            row["prefix5_acc"] = float(prefix5_match_cnt / max(1, len(side_by_side)))
            logger.log(
                f"[overfit8] step={step} loss={row['train_loss']:.6f} "
                f"answer_token_acc={row['train_answer_token_acc']:.4f} "
                f"tf_shifted_acc={row['teacher_forced_shifted_token_acc_mean']:.4f} "
                f"grad_norm={row['grad_norm']:.6f} delta={row['lora_param_delta_l2']:.6f} "
                f"exact_match={exact_match_cnt}/{len(side_by_side)} "
                f"prefix1={prefix1_match_cnt}/{len(side_by_side)} "
                f"prefix3={prefix3_match_cnt}/{len(side_by_side)} "
                f"prefix5={prefix5_match_cnt}/{len(side_by_side)}"
            )
            if run_failure_decomposition or run_first_token_audit or run_prefix_rollout:
                from core.overfit_sequence_diagnostics import (
                    run_failure_decomposition_step,
                    run_first_token_audit_step,
                    run_prefix_rollout_step,
                )

                if run_failure_decomposition:
                    run_failure_decomposition_step(
                        step=step,
                        out_dir=out_dir,
                        examples=train_subset,
                        preds=preds,
                        infer_prompts=infer_prompts,
                        normalization_cfg=normalization_cfg,
                    )
                if run_first_token_audit:
                    run_first_token_audit_step(
                        step=step,
                        out_dir=out_dir,
                        examples=train_subset,
                        infer_prompts=infer_prompts,
                        backbone=backbone,
                        pairs=pairs,
                        first_token_gold_source=str(debug_tools.get("first_token_gold_source", "supervised_span")),
                    )
                if run_prefix_rollout:
                    run_prefix_rollout_step(
                        step=step,
                        out_dir=out_dir,
                        examples=train_subset,
                        infer_prompts=infer_prompts,
                        backbone=backbone,
                        max_new_tokens=overfit_gen_max_new_tokens,
                        normalization_cfg=normalization_cfg,
                    )
                manifest_add_artifact(run_manifest, out_dir / f"step_{step:04d}_failure_analysis.json")
                manifest_add_artifact(run_manifest, out_dir / f"step_{step:04d}_failure_summary.csv")
                manifest_add_artifact(run_manifest, out_dir / f"step_{step:04d}_first_token_audit.json")
                manifest_add_artifact(run_manifest, out_dir / f"step_{step:04d}_first_token_gold_source_compare.json")
                manifest_add_artifact(run_manifest, out_dir / "first_token_gold_source_compare.csv")
                manifest_add_artifact(run_manifest, out_dir / "first_token_margin_over_time.csv")
                manifest_add_artifact(run_manifest, out_dir / f"step_{step:04d}_prefix_rollout.json")
                manifest_add_artifact(run_manifest, out_dir / "prefix_rollout_summary.csv")

        state_path.write_text(
            json.dumps({"last_completed_step": step, "overfit_steps": steps}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        save_csv(str(csv_path), per_step_rows)
        manifest_add_artifact(run_manifest, state_path)
        manifest_add_artifact(run_manifest, csv_path)
        # Always refresh latest/ each step so resume matches overfit_state.json.
        if hasattr(lora, "save_adapter_checkpoint"):
            ensure_dir(str(ckpt_root))
            lora.save_adapter_checkpoint(str(ckpt_root / "latest"))
            manifest_add_artifact(run_manifest, ckpt_root / "latest")
        if snapshot_every > 0 and (step % snapshot_every == 0 or step == steps or step == 1):
            ensure_dir(str(ckpt_root))
            if hasattr(lora, "save_adapter_checkpoint"):
                lora.save_adapter_checkpoint(str(ckpt_root / f"step_{step:06d}"))
                manifest_add_artifact(run_manifest, ckpt_root / f"step_{step:06d}")
        _flush_overfit_manifest()

    save_csv(str(csv_path), per_step_rows)

    post_overfit_adapter = str(ckpt_root / "post_overfit_adapter")
    if diag_enabled and (run_decode_ablation or run_overfit_ladder) and hasattr(lora, "save_adapter_checkpoint"):
        ensure_dir(str(ckpt_root))
        lora.save_adapter_checkpoint(post_overfit_adapter)
        manifest_add_artifact(run_manifest, Path(post_overfit_adapter))
        logger.log(f"[overfit8] Saved post-overfit adapter for decode/ladder restore: {post_overfit_adapter}")

    if diag_enabled and run_decode_ablation:
        from core.overfit_sequence_diagnostics import run_decode_ablation

        infer_prompts_final = [
            format_for_infer(backbone.tokenizer, ins, inp, add_generation_prompt=True) for (ins, inp) in pairs
        ]
        run_decode_ablation(
            out_dir=out_dir,
            examples=list(train_subset),
            infer_prompts=infer_prompts_final,
            backbone=backbone,
            max_new_tokens=overfit_gen_max_new_tokens,
            normalization_cfg=normalization_cfg,
        )
        manifest_add_artifact(run_manifest, out_dir / "decode_ablation.json")
        manifest_add_artifact(run_manifest, out_dir / "decode_ablation.csv")
        logger.log("[overfit8] Wrote decode_ablation.json / decode_ablation.csv")

    if diag_enabled and run_overfit_ladder:
        from core.overfit_sequence_diagnostics import run_overfit_ladder

        infer_prompts_final = [
            format_for_infer(backbone.tokenizer, ins, inp, add_generation_prompt=True) for (ins, inp) in pairs
        ]
        run_overfit_ladder(
            ladder_sizes=(1, 2, 4, 8),
            ladder_steps=ladder_steps,
            pairs=pairs,
            targets=targets,
            train_subset=list(train_subset),
            infer_prompts=infer_prompts_final,
            backbone=backbone,
            lora=lora,
            lr=lr,
            overfit_batch_size=overfit_batch_size,
            overfit_gen_max_new_tokens=overfit_gen_max_new_tokens,
            normalization_cfg=normalization_cfg,
            ladder_init_adapter_path=ladder_init_path,
            out_csv=out_dir / "overfit_ladder.csv",
            logger=logger,
        )
        manifest_add_artifact(run_manifest, out_dir / "overfit_ladder.csv")
        logger.log("[overfit8] Wrote overfit_ladder.csv")

    if diag_enabled and Path(post_overfit_adapter).is_dir() and hasattr(lora, "load_adapter_checkpoint"):
        lora.load_adapter_checkpoint(post_overfit_adapter)
        logger.log("[overfit8] Restored post-overfit adapter after decode/ladder.")

    if diag_enabled:
        from core.overfit_sequence_diagnostics import write_sequence_behavior_report

        results_dir = str(cfg.get("paths", {}).get("results_dir", "results"))
        experiment_name = str(cfg.get("experiment_name", "experiment"))
        _flush_overfit_manifest()
        write_sequence_behavior_report(
            run_manifest_path=manifest_path,
            results_dir=Path(results_dir),
            experiment_name=experiment_name,
            run_id=run_paths.run_id,
        )
        manifest_add_artifact(run_manifest, Path(results_dir) / "debug_report_sequence_behavior_diagnosis.md")
        _flush_overfit_manifest()
        logger.log(
            f"[overfit8] Wrote sequence behavior report: "
            f"{Path(results_dir) / 'debug_report_sequence_behavior_diagnosis.md'}"
        )

    if getattr(backbone, "tokenizer", None) is not None and getattr(backbone, "model", None) is not None:
        from core.debug_teacher_audit import dump_overfit_teacher_audit

        if bool(debug_tools.get("dump_teacher_forced_audit", True)):
            dump_overfit_teacher_audit(
                backbone=backbone,
                examples=list(train_subset),
                out_path=str(out_dir / "teacher_forced_audit_final.json"),
                strict_assertions=bool(debug_tools.get("strict_teacher_metric_assertions", False)),
            )
            manifest_add_artifact(run_manifest, out_dir / "teacher_forced_audit_final.json")

    # Final A/B generation prompt ablation on the same 8 overfit samples.
    # Variant A: add_generation_prompt=True
    # Variant B: add_generation_prompt=False (may have no effect depending on the chat template).
    tok = getattr(backbone, "tokenizer", None)
    if tok is not None and hasattr(backbone, "generate_with_ids"):
        infer_prompts_a = [format_for_infer(tok, ins, inp, add_generation_prompt=True) for (ins, inp) in pairs]
        infer_prompts_b = [format_for_infer(tok, ins, inp, add_generation_prompt=False) for (ins, inp) in pairs]

        audit_a = backbone.generate_with_ids(infer_prompts_a, max_new_tokens=overfit_gen_max_new_tokens)
        audit_b = backbone.generate_with_ids(infer_prompts_b, max_new_tokens=overfit_gen_max_new_tokens)
        preds_a = [x.get("raw_generated_text", "") for x in audit_a]
        preds_b = [x.get("raw_generated_text", "") for x in audit_b]

        def _compute_metrics(preds: List[str], prompts_local: List[str]) -> Dict[str, Any]:
            exact = 0
            p1 = 0
            p3 = 0
            p5 = 0
            for ex, pred, pr in zip(train_subset, preds, prompts_local):
                n_pred = _normalize(pred or "", prompt=pr, cfg=normalization_cfg)
                n_gold = _normalize(ex.output or "", prompt=pr, cfg=normalization_cfg)
                if n_pred == n_gold:
                    exact += 1
                pred_tokens = [t for t in n_pred.split() if t]
                gold_tokens = [t for t in n_gold.split() if t]
                p1 += int(pred_tokens[:1] == gold_tokens[:1])
                p3 += int(pred_tokens[:3] == gold_tokens[:3])
                p5 += int(pred_tokens[:5] == gold_tokens[:5])
            return {
                "exact_match_count": int(exact),
                "prefix1_acc": float(p1 / max(1, len(train_subset))),
                "prefix3_acc": float(p3 / max(1, len(train_subset))),
                "prefix5_acc": float(p5 / max(1, len(train_subset))),
            }

        m_a = _compute_metrics(preds_a, infer_prompts_a)
        m_b = _compute_metrics(preds_b, infer_prompts_b)

        save_csv(
            str(generation_ablation_out_csv),
            [
                {"variant": "A_add_generation_prompt_true", **m_a},
                {"variant": "B_add_generation_prompt_false", **m_b},
            ],
        )
        manifest_add_artifact(run_manifest, generation_ablation_out_csv)

        cont_head_rows: List[Dict[str, Any]] = []
        head_match_cnt = 0
        for i, ex in enumerate(train_subset):
            a_head = (audit_a[i].get("decoded_continuation_head", "") or "")
            b_head = (audit_b[i].get("decoded_continuation_head", "") or "")
            a_tokens = [t for t in _normalize(a_head, prompt="", cfg=normalization_cfg).split() if t]
            b_tokens = [t for t in _normalize(b_head, prompt="", cfg=normalization_cfg).split() if t]
            match5 = bool(a_tokens[:5] == b_tokens[:5])
            head_match_cnt += int(match5)
            cont_head_rows.append(
                {
                    "idx": i,
                    "gold_output": ex.output,
                    "A_decoded_continuation_head": a_head,
                    "B_decoded_continuation_head": b_head,
                    "head_prefix5_match": match5,
                }
            )
        generation_ablation_out_cont_json.write_text(
            json.dumps(
                {
                    "continuation_head_prefix5_match_rate": float(head_match_cnt / max(1, len(train_subset))),
                    "rows": cont_head_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_add_artifact(run_manifest, generation_ablation_out_cont_json)
    _flush_overfit_manifest()




if __name__ == "__main__":
    main()

