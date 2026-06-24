"""Official CITB T5 command planner for strict preflight.

The published CITB scripts run full Seq2Seq fine-tuning, not LoRA. This module
bridges local YAML configs to the official runner command and reports missing
inputs. It deliberately avoids importing Transformers or launching training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .data_splits import REPO, get_stream_spec, resolve_official_repo, task_order_dir
from .replay_memory import replay_examples_from_policy


DEFAULT_INIT_MODEL = "output/initial_multitask_model/base_epoch15_lr1e-05_seed50/checkpoint-14000"


@dataclass(frozen=True)
class CITBRunPlan:
    official_workdir: str
    entry: str
    cl_method: str
    paper_variant: str
    model_name_or_path: str
    command: List[str]
    missing_inputs: List[str]
    notes: List[str]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"


def _resolve_model_for_method(
    *,
    official_repo: Path,
    method_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> tuple[str, List[str], List[str]]:
    method_name = str(method_cfg.get("name", ""))
    paper_variant = str(method_cfg.get("paper_variant", ""))
    configured_model = str(model_cfg.get("model_name_or_path", "")).strip()
    notes: List[str] = []
    missing: List[str] = []

    if method_name == "citb_t5_sequential" and paper_variant.upper().replace("_", "-") in {"FT-NO-INSTR", "FT-NO-INIT"}:
        model = configured_model or "google/t5-small-lm-adapt"
        notes.append("FT-no-init starts from LM-adapted T5-small.")
    elif method_name in {"citb_t5_sequential", "citb_t5_replay"}:
        model = DEFAULT_INIT_MODEL
        init_checkpoint = official_repo / model
        if not init_checkpoint.exists():
            missing.append(str(init_checkpoint))
        notes.append("FT-init and Replay use the stage-1 initial multitask checkpoint from official CITB.")
    else:
        model = configured_model
        notes.append(f"Unknown method {method_name!r}; command planner only validates common arguments.")

    if configured_model:
        local_model = Path(configured_model)
        if not local_model.is_absolute():
            local_model = REPO / local_model
        if local_model.exists():
            notes.append(f"Local configured model asset is present: {local_model}")

    return model, missing, notes


def build_official_continual_command(
    cfg: Mapping[str, Any],
    *,
    config_path: Optional[Path] = None,
    official_repo: Optional[Path | str] = None,
) -> CITBRunPlan:
    """Create the official CITB command that would be used for a strict run."""

    official = resolve_official_repo(official_repo)
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    method_cfg = cfg.get("method") or {}
    output_cfg = cfg.get("output") or {}

    benchmark = str(data_cfg.get("benchmark", "InstrDialog"))
    spec = get_stream_spec(benchmark)
    method_name = str(method_cfg.get("name", ""))
    paper_variant = str(method_cfg.get("paper_variant", ""))
    run_name = str(output_cfg.get("run_name") or (config_path.stem if config_path else "citb_t5_run"))
    seed = str(cfg.get("seed", 123))
    order = int(data_cfg.get("order", 1))

    cl_method = "FT_INSTR"
    replay_num: Optional[int] = None
    if method_name == "citb_t5_replay":
        cl_method = "REPLAY"
        replay_num = replay_examples_from_policy(str(method_cfg.get("memory_policy", "replay50")))
    elif paper_variant.upper().replace("_", "-") in {"FT-NO-INSTR", "FT-NO-INIT"}:
        cl_method = "FT_NO_INSTR"

    model_name, missing_inputs, notes = _resolve_model_for_method(
        official_repo=official,
        method_cfg=method_cfg,
        model_cfg=model_cfg,
    )

    entry = "continual_learning/run_continual_instruct_tuning.py"
    command = [
        "python",
        entry,
        "--do_train",
        "--do_eval",
        "--do_predict",
        "--predict_with_generate",
        "--model_name_or_path",
        model_name,
        "--cl_method",
        cl_method,
        "--order",
        str(order),
        "--data_dir_for_task_order",
        str(task_order_dir(official, spec).relative_to(official)),
        "--task_split_file_name",
        spec.task_split_file_name,
        "--data_dir_for_official_test",
        "data/CIT_data/official_test_data",
        "--data_dir_for_initial_training_dir",
        "data/CIT_data/initial_multitask_learning/defintion_pos_2",
        "--max_source_length",
        "1024",
        "--max_target_length",
        "128",
        "--generation_max_length",
        "128",
        "--max_num_instances_per_task",
        str(spec.max_train_instances_per_task),
        "--max_num_instances_per_eval_task",
        str(spec.max_test_instances_per_task),
        "--add_task_name",
        _bool_arg(False),
        "--add_task_definition",
        _bool_arg(True),
        "--num_pos_examples",
        "2",
        "--num_neg_examples",
        "0",
        "--add_explanation",
        _bool_arg(False),
        "--tk_instruct",
        _bool_arg(False),
        "--data_dir",
        "data/splits/CIT_splits/",
        "--task_dir",
        "data/tasks/",
        "--output_dir",
        f"output/continual_instruction_tuning/stream={spec.task_order_stream}/CL={cl_method}/{run_name}",
        "--overwrite_output_dir",
        "--cache_dir",
        "./cache/",
        "--overwrite_cache",
        "--per_device_train_batch_size",
        "8",
        "--per_device_eval_batch_size",
        "32",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "1e-05",
        "--num_train_epochs",
        "15",
        "--lr_scheduler_type",
        "constant",
        "--warmup_steps",
        "0",
        "--logging_strategy",
        "steps",
        "--logging_steps",
        "50",
        "--evaluation_strategy",
        "epoch",
        "--save_strategy",
        "epoch",
        "--save_total_limit",
        "1",
        "--load_best_model_at_end",
        "--metric_for_best_model",
        "rougeL",
        "--run_name",
        run_name,
        "--seed",
        seed,
    ]
    if replay_num is not None:
        command.extend(["--replay_num_instance_per_task", str(replay_num)])
        notes.append(f"Replay policy {method_cfg.get('memory_policy', 'replay50')} maps to {replay_num} examples per task.")

    for required in [
        official / entry,
        official / "data/splits/CIT_splits" / f"{spec.task_split_file_name}.txt",
        task_order_dir(official, spec) / f"order{order}.txt",
        official / "data/CIT_data/official_test_data/state.json",
        official / "data/CIT_data/initial_multitask_learning/defintion_pos_2/train/state.json",
    ]:
        if not required.exists():
            missing_inputs.append(str(required))

    return CITBRunPlan(
        official_workdir=str(official),
        entry=entry,
        cl_method=cl_method,
        paper_variant=paper_variant,
        model_name_or_path=model_name,
        command=command,
        missing_inputs=sorted(set(missing_inputs)),
        notes=notes,
    )

