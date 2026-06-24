"""Replay memory helpers matching the official CITB implementation.

Official CITB builds replay memory from the initial multitask train dataset by
grouping examples by ``Task`` and taking the first N examples per task. The
dataset itself is produced after seeded task-level shuffling in the official
data preparation stage, so this selector intentionally preserves input order.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


SUPPORTED_REPLAY_POLICIES = {"replay10": 10, "replay50": 50}


def replay_examples_from_policy(policy: str) -> int:
    try:
        return SUPPORTED_REPLAY_POLICIES[policy]
    except KeyError as exc:
        expected = ", ".join(sorted(SUPPORTED_REPLAY_POLICIES))
        raise ValueError(f"unsupported CITB replay policy {policy!r}; expected one of: {expected}") from exc


def group_instances_by_task(instances: Iterable[Mapping[str, Any]], task_key: str = "Task") -> Dict[str, List[Mapping[str, Any]]]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for instance in instances:
        if task_key not in instance:
            raise KeyError(f"instance is missing CITB task key {task_key!r}: {instance}")
        grouped[str(instance[task_key])].append(instance)
    return dict(grouped)


def select_replay_instances_by_task(
    instances: Iterable[Mapping[str, Any]],
    replay_num_instance_per_task: int,
    *,
    task_key: str = "Task",
) -> List[Mapping[str, Any]]:
    """Select replay examples using the official first-N-per-task policy."""

    if replay_num_instance_per_task <= 0:
        raise ValueError("replay_num_instance_per_task must be positive")
    grouped = group_instances_by_task(instances, task_key=task_key)
    replay_instances: List[Mapping[str, Any]] = []
    for task_instances in grouped.values():
        replay_instances.extend(task_instances[:replay_num_instance_per_task])
    return replay_instances


def replay_memory_summary(instances: Sequence[Mapping[str, Any]], replay_num_instance_per_task: int) -> Dict[str, Any]:
    grouped = group_instances_by_task(instances)
    selected = select_replay_instances_by_task(instances, replay_num_instance_per_task)
    return {
        "source_task_count": len(grouped),
        "source_instance_count": len(instances),
        "replay_num_instance_per_task": replay_num_instance_per_task,
        "selected_instance_count": len(selected),
        "selector": "official_first_n_per_task_after_seeded_data_preparation",
    }

