"""CITB official split and task-order inspection helpers.

Official CITB builds CL data from Super-NaturalInstructions tasks with:
- InstrDialog: ``cl_dialogue_tasks`` (19 tasks)
- InstrDialog++: ``cl_dialogue_long_tasks`` task orders over ``cl_38_random_tasks``
- max train instances per CL task: 500
- max eval instances per task: 50 test + 50 dev, selected before train sampling

This module only validates and describes local official resources. It does not
materialize HuggingFace datasets or mutate the official checkout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_REPO = REPO / "external_baselines/citb_official"


@dataclass(frozen=True)
class CITBStreamSpec:
    benchmark: str
    task_split_file_name: str
    task_order_stream: str
    expected_task_count: int
    max_train_instances_per_task: int = 500
    max_dev_instances_per_task: int = 50
    max_test_instances_per_task: int = 50

    @property
    def eval_instances_per_task_total(self) -> int:
        return self.max_dev_instances_per_task + self.max_test_instances_per_task

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["eval_instances_per_task_total"] = self.eval_instances_per_task_total
        return payload


STREAM_SPECS: Mapping[str, CITBStreamSpec] = {
    "InstrDialog": CITBStreamSpec(
        benchmark="InstrDialog",
        task_split_file_name="cl_dialogue_tasks",
        task_order_stream="cl_dialogue_tasks",
        expected_task_count=19,
    ),
    "InstrDialog++": CITBStreamSpec(
        benchmark="InstrDialog++",
        task_split_file_name="cl_38_random_tasks",
        task_order_stream="cl_dialogue_long_tasks",
        expected_task_count=38,
    ),
}


def resolve_official_repo(path: Optional[Path | str] = None) -> Path:
    """Return the official CITB checkout path without requiring imports there."""

    candidate = Path(path) if path is not None else DEFAULT_OFFICIAL_REPO
    if not candidate.is_absolute():
        candidate = REPO / candidate
    return candidate


def get_stream_spec(benchmark: str) -> CITBStreamSpec:
    try:
        return STREAM_SPECS[benchmark]
    except KeyError as exc:
        known = ", ".join(sorted(STREAM_SPECS))
        raise ValueError(f"unsupported CITB benchmark {benchmark!r}; expected one of: {known}") from exc


def _read_nonempty_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_split_path(official_repo: Path, spec: CITBStreamSpec) -> Path:
    return official_repo / "data/splits/CIT_splits" / f"{spec.task_split_file_name}.txt"


def task_order_dir(official_repo: Path, spec: CITBStreamSpec) -> Path:
    return official_repo / "data/CIT_data/task_orders" / f"stream={spec.task_order_stream}"


def _referenced_dataset_files(dataset_dir: Path) -> List[Path]:
    state_path = dataset_dir / "state.json"
    if not state_path.is_file():
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return [dataset_dir / item["filename"] for item in state.get("_data_files", []) if item.get("filename")]


def load_task_split(official_repo: Path | str, benchmark: str) -> List[str]:
    repo = resolve_official_repo(official_repo)
    spec = get_stream_spec(benchmark)
    return _read_nonempty_lines(task_split_path(repo, spec))


def load_task_order(official_repo: Path | str, benchmark: str, order: int) -> List[str]:
    repo = resolve_official_repo(official_repo)
    spec = get_stream_spec(benchmark)
    return _read_nonempty_lines(task_order_dir(repo, spec) / f"order{order}.txt")


def inspect_official_split(official_repo: Path | str, benchmark: str) -> Dict[str, object]:
    """Validate official files needed before a strict CITB run can be launched."""

    repo = resolve_official_repo(official_repo)
    spec = get_stream_spec(benchmark)
    required_paths = {
        "readme": repo / "README.md",
        "continual_runner": repo / "continual_learning/run_continual_instruct_tuning.py",
        "data_utils": repo / "continual_learning/utils.py",
        "score_collector": repo / "collect_results.py",
        "task_split": task_split_path(repo, spec),
        "task_order_dir": task_order_dir(repo, spec),
        "official_test_data": repo / "data/CIT_data/official_test_data",
        "initial_multitask_train": repo / "data/CIT_data/initial_multitask_learning/defintion_pos_2/train",
        "initial_multitask_test": repo / "data/CIT_data/initial_multitask_learning/defintion_pos_2/test",
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    dataset_file_errors: List[str] = []
    for name in ["official_test_data", "initial_multitask_train", "initial_multitask_test"]:
        dataset_dir = required_paths[name]
        for data_file in _referenced_dataset_files(dataset_dir):
            if not data_file.is_file():
                dataset_file_errors.append(f"{name} references missing data file {data_file.name}")

    tasks: List[str] = []
    orders: Dict[str, List[str]] = {}
    order_errors: List[str] = []
    if required_paths["task_split"].is_file():
        tasks = _read_nonempty_lines(required_paths["task_split"])
    if required_paths["task_order_dir"].is_dir():
        for order in (1, 2, 3):
            order_path = required_paths["task_order_dir"] / f"order{order}.txt"
            if not order_path.is_file():
                order_errors.append(f"missing order{order}.txt")
                continue
            order_tasks = _read_nonempty_lines(order_path)
            orders[str(order)] = order_tasks
            if len(order_tasks) != spec.expected_task_count:
                order_errors.append(f"order{order} has {len(order_tasks)} tasks, expected {spec.expected_task_count}")

    if tasks and len(tasks) != spec.expected_task_count:
        order_errors.append(f"task split has {len(tasks)} tasks, expected {spec.expected_task_count}")
    task_set = set(tasks)
    for order, order_tasks in orders.items():
        if task_set and set(order_tasks) != task_set:
            order_errors.append(f"order{order} task set differs from {spec.task_split_file_name}.txt")

    return {
        "official_repo": str(repo),
        "benchmark": benchmark,
        "spec": spec.as_dict(),
        "required_paths": {name: str(path) for name, path in required_paths.items()},
        "missing": missing,
        "dataset_file_errors": dataset_file_errors,
        "task_count": len(tasks),
        "available_orders": sorted(orders),
        "order_errors": order_errors,
        "strict_split_available": not missing and not order_errors and not dataset_file_errors,
    }


def is_official_split_name(split_name: str) -> bool:
    return split_name in {"citb_official", "official_citb", "paper_official"}


def describe_split_mismatch(split_name: str) -> Optional[str]:
    if is_official_split_name(split_name):
        return None
    if split_name == "train50_eval10":
        return (
            "local train50/eval10 stream is not the CITB paper split; official CITB uses "
            "up to 500 train instances plus 50 dev and 50 test instances per CL task"
        )
    return f"unrecognized split {split_name!r}; strict CITB requires the official paper split"


def all_missing_modules(paths: Iterable[Path | str]) -> List[str]:
    return [str(path) for path in paths if not Path(path).is_file()]

