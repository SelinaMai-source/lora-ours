#!/usr/bin/env python3
"""
Convert MultiWOZ 2.2 NLG (semantic-act -> system response) into unified continual stream JSON.

Logic aligned with external_baselines/adaptercl_dialogue:
  - data/MWOZ.py preprocessMWOZ
  - utils/dataloader.py get_NLG_from_dial

Domain continual order (5 segments, seed=123):
  restaurant -> hotel -> attraction -> train -> taxi

Raw layout (after adaptercl download.sh or manual setup):
  {raw_root}/multiwoz_repo/data/MultiWOZ_2.2/data.json
  {raw_root}/multiwoz_repo/data/MultiWOZ_2.1/valListFile.txt
  {raw_root}/multiwoz_repo/data/MultiWOZ_2.1/testListFile.txt

Output matches CITB/TRACE processed streams:
  {"benchmark": "MultiWOZ-NLG", "version": "...", "stream": [{segment_id, segment_name, train, eval}, ...]}
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

MULTIWOZ_DOMAIN_ORDER: Tuple[str, ...] = (
    "restaurant",
    "hotel",
    "attraction",
    "train",
    "taxi",
)

NLG_INSTRUCTION = (
    "Generate a natural language system response for the given semantic dialogue acts."
)

DUPLICATE_DIAL_ID = "dlg-ff2b8de2-467d-4917-be13-1529765752e9"

DEFAULT_OUT = "data/processed/multiwoz_nlg_cl_domains_train50_eval10.json"
DEFAULT_RAW_ROOT = "data/raw/multiwoz"
DEFAULT_VERSION = "multiwoz_nlg_domains_train50_eval10_v1"


def _service_key(domain: str) -> str:
    return f"MWOZ_{domain}"


def get_value_dst(dst: Dict[str, Any]) -> Dict[str, List[List[str]]]:
    active_dst: Dict[str, List[List[str]]] = defaultdict(list)
    for k, v in dst.items():
        for k_s, v_s in v["semi"].items():
            if len(v_s) != 0:
                active_dst[k].append([k_s, v_s])
        for k_s, v_s in v["book"].items():
            if len(v_s) != 0 and k_s != "booked":
                active_dst[k].append([k_s, v_s])
    return active_dst


def get_domains(goal: Dict[str, Any]) -> List[str]:
    dom: List[str] = []
    for d, g in goal.items():
        if len(g) != 0 and d not in ("message", "topic"):
            dom.append(f"MWOZ_{d}")
    return dom


def load_split_ids(split_list_path: Path) -> List[str]:
    ids: List[str] = []
    with split_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(line.replace("\n", ""))
    return ids


def preprocess_mwoz_dialogues(data_json_path: Path) -> List[Dict[str, Any]]:
    with data_json_path.open("r", encoding="utf-8") as f:
        dialogue = json.load(f)

    data: List[Dict[str, Any]] = []
    for d_idx, d in dialogue.items():
        services = get_domains(d["goal"])
        if any(x in services for x in ("MWOZ_police", "MWOZ_hospital", "MWOZ_bus")):
            continue

        turns: List[Dict[str, Any]] = []
        intents_act: set[str] = set()
        str_api_act = ""

        for t_idx, t in enumerate(d["log"]):
            if t_idx % 2 == 0:
                turns.append(
                    {
                        "dataset": "MWOZ",
                        "id": d_idx,
                        "turn_id": t_idx,
                        "spk": "USER",
                        "utt": t["text"],
                    }
                )
                str_api_act = ""
                if "dialog_act" in t:
                    intents_act = set()
                    for k, slt in t["dialog_act"].items():
                        if "Inform" in k or "Request" in k:
                            str_api_act += f"{k.lower().replace('-', '_')}("
                            for s, v in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"', "'")
                                    str_api_act += f"{s.lower()}=\"{v}\","
                                    intents_act.add(k.lower().replace("-", "_"))
                            if str_api_act.endswith(","):
                                str_api_act = str_api_act[:-1]
                            str_api_act += ") "
            else:
                dst_api = get_value_dst(t["metadata"])
                str_api = ""
                intents: set[str] = set()
                for k, slt in dst_api.items():
                    str_api += f"{k.lower().replace('-', '_')}("
                    for s, v in slt:
                        if len(v) != 0:
                            v = v[0].replace('"', "'")
                            str_api += f"{s.lower()}=\"{v}\","
                            intents.add(k.lower().replace("-", "_"))
                    if len(str_api) > 0 and str_api.endswith(","):
                        str_api = str_api[:-1]
                    str_api += ") "

                if str_api == "":
                    turns.append(
                        {
                            "dataset": "MWOZ",
                            "id": d_idx,
                            "turn_id": t_idx,
                            "spk": "API",
                            "utt": str_api_act,
                            "service": list(intents_act),
                        }
                    )
                else:
                    turns.append(
                        {
                            "dataset": "MWOZ",
                            "id": d_idx,
                            "turn_id": t_idx,
                            "spk": "API",
                            "utt": str_api,
                            "service": list(intents),
                        }
                    )

                str_act = ""
                if "dialog_act" in t:
                    for k, slt in t["dialog_act"].items():
                        if (
                            "Inform" in k
                            or "Recommend" in k
                            or "Booking-Book" in k
                            or "-Select" in k
                        ):
                            str_act += f"{k.lower().replace('-', '_')}("
                            for s, v in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"', "'")
                                    str_act += f"{s.lower()}=\"{v}\","
                            if str_act.endswith(","):
                                str_act = str_act[:-1]
                            str_act += ") "
                        if "Booking-NoBook" in k:
                            str_act += f"{k.lower().replace('-', '_')}() "

                turns.append(
                    {
                        "dataset": "MWOZ",
                        "id": d_idx,
                        "turn_id": t_idx,
                        "spk": "API-OUT",
                        "utt": str_act,
                        "service": None,
                    }
                )
                turns.append(
                    {
                        "dataset": "MWOZ",
                        "id": d_idx,
                        "turn_id": t_idx,
                        "spk": "SYSTEM",
                        "utt": t["text"],
                    }
                )

        data.append({"id": d_idx, "services": services, "dataset": "MWOZ", "dialogue": turns})
    return data


def split_dialogues(
    dialogues: Sequence[Dict[str, Any]],
    *,
    dev_ids: Sequence[str],
    test_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    dev_set = set(dev_ids)
    test_set = set(test_ids)
    train_data: List[Dict[str, Any]] = []
    dev_data: List[Dict[str, Any]] = []
    test_data: List[Dict[str, Any]] = []

    for dial in dialogues:
        if dial["id"] in dev_set:
            dev_data.append(dial)
        elif dial["id"] in test_set:
            test_data.append(dial)
        else:
            train_data.append(dial)
    return train_data, dev_data, test_data


def filter_single_domain(dialogues: Sequence[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    key = _service_key(domain)
    out: List[Dict[str, Any]] = []
    for dial in dialogues:
        if dial["services"] == [key]:
            out.append(dial)
    return out


def extract_nlg_examples(dialogues: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Mirror adaptercl get_NLG_from_dial (history=API-OUT acts, reply=SYSTEM utterance)."""
    examples: List[Dict[str, str]] = []

    for dial in dialogues:
        if dial["id"] == DUPLICATE_DIAL_ID:
            continue

        latest_api_out = ""
        for idx_t, t in enumerate(dial["dialogue"]):
            if t["id"] == DUPLICATE_DIAL_ID:
                continue
            if t["spk"] == "API-OUT":
                latest_api_out = str(t["utt"]).strip()
            elif t["spk"] == "SYSTEM" and idx_t != 0 and str(t["utt"]).strip() != "":
                if latest_api_out:
                    examples.append(
                        {
                            "instruction": NLG_INSTRUCTION,
                            "input": latest_api_out,
                            "output": str(t["utt"]).strip(),
                        }
                    )
                latest_api_out = ""

    return examples


def _select_split(
    samples: Sequence[Dict[str, str]],
    *,
    max_count: int,
    rng: random.Random,
    shuffle: bool,
) -> List[Dict[str, str]]:
    items = list(samples)
    if shuffle:
        rng.shuffle(items)
    if max_count > 0:
        items = items[:max_count]
    return items


def convert_domain_segment(
    *,
    domain: str,
    segment_id: int,
    train_dialogues: Sequence[Dict[str, Any]],
    eval_dialogues: Sequence[Dict[str, Any]],
    max_train: int,
    max_eval: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(int(seed) + segment_id)
    train_raw = extract_nlg_examples(train_dialogues)
    eval_raw = extract_nlg_examples(eval_dialogues)

    train_selected = _select_split(train_raw, max_count=max_train, rng=rng, shuffle=True)
    eval_selected = _select_split(eval_raw, max_count=max_eval, rng=rng, shuffle=False)

    if not train_selected or not eval_selected:
        raise ValueError(
            f"Domain {domain} produced empty train/eval after selection "
            f"(train={len(train_selected)}, eval={len(eval_selected)}, "
            f"raw train={len(train_raw)}, raw eval={len(eval_raw)})"
        )

    print(
        f"[{segment_id:03d}] {domain}: "
        f"train={len(train_selected)} eval={len(eval_selected)} "
        f"(raw train turns={len(train_raw)} raw eval turns={len(eval_raw)}, "
        f"dials train={len(train_dialogues)} eval={len(eval_dialogues)})"
    )

    return {
        "segment_id": segment_id,
        "segment_name": domain,
        "train": train_selected,
        "eval": eval_selected,
    }


def build_toy_segments(*, max_train: int, max_eval: int) -> List[Dict[str, Any]]:
    toy_specs = [
        (
            "restaurant",
            [
                (
                    "restaurant_inform(name=\"The Golden Curry\", price=\"moderate\") ",
                    "The Golden Curry is a moderately priced restaurant.",
                ),
                (
                    "restaurant_request(area=\"centre\") ",
                    "Which area of town are you looking for?",
                ),
            ],
        ),
        (
            "hotel",
            [
                (
                    "hotel_inform(name=\"Alexander Bed and Breakfast\", type=\"guesthouse\") ",
                    "Alexander Bed and Breakfast is a guesthouse.",
                ),
                (
                    "hotel_request(parking=\"yes\") ",
                    "Do you need parking?",
                ),
            ],
        ),
    ]

    segments: List[Dict[str, Any]] = []
    for seg_id, (domain, pairs) in enumerate(toy_specs):
        examples = [
            {"instruction": NLG_INSTRUCTION, "input": inp, "output": out} for inp, out in pairs
        ]
        train_n = min(max_train, len(examples))
        eval_n = min(max_eval, len(examples))
        segments.append(
            {
                "segment_id": seg_id,
                "segment_name": domain,
                "train": examples[:train_n],
                "eval": examples[:eval_n],
            }
        )
        print(f"[{seg_id:03d}] {domain} (toy): train={train_n} eval={eval_n}")
    return segments


def resolve_raw_paths(raw_root: Path) -> Tuple[Path, Path, Path]:
    data_json = raw_root / "multiwoz_repo" / "data" / "MultiWOZ_2.2" / "data.json"
    val_list = raw_root / "multiwoz_repo" / "data" / "MultiWOZ_2.1" / "valListFile.txt"
    test_list = raw_root / "multiwoz_repo" / "data" / "MultiWOZ_2.1" / "testListFile.txt"
    return data_json, val_list, test_list


def convert_multiwoz_to_stream(
    *,
    raw_root: Path,
    out_path: Path,
    max_train: int,
    max_eval: int,
    seed: int,
    limit_domains: int,
    domain_order: Sequence[str],
    mode: str,
    version: str,
) -> Dict[str, Any]:
    if mode == "toy":
        segments = build_toy_segments(max_train=max_train, max_eval=max_eval)
    else:
        data_json, val_list, test_list = resolve_raw_paths(raw_root)
        missing = [p for p in (data_json, val_list, test_list) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"MultiWOZ raw files missing under {raw_root}: {[str(p) for p in missing]}. "
                "See data/raw/multiwoz/README.md or external_baselines/adaptercl_dialogue/data/download.sh."
            )

        all_dialogues = preprocess_mwoz_dialogues(data_json)
        train_all, _dev_all, test_all = split_dialogues(
            all_dialogues,
            dev_ids=load_split_ids(val_list),
            test_ids=load_split_ids(test_list),
        )

        domains = list(domain_order)
        if limit_domains > 0:
            domains = domains[:limit_domains]

        segments = []
        for seg_id, domain in enumerate(domains):
            train_dials = filter_single_domain(train_all, domain)
            eval_dials = filter_single_domain(test_all, domain)
            segments.append(
                convert_domain_segment(
                    domain=domain,
                    segment_id=seg_id,
                    train_dialogues=train_dials,
                    eval_dialogues=eval_dials,
                    max_train=max_train,
                    max_eval=max_eval,
                    seed=seed,
                )
            )

    payload = {
        "benchmark": "MultiWOZ-NLG",
        "version": version if mode == "full" else f"{version}_toy",
        "stream": segments,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path} ({len(segments)} segments)")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MultiWOZ NLG raw dialogues to unified continual stream JSON."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "toy"),
        default="full",
        help="full=read raw MultiWOZ; toy=smoke data without raw files",
    )
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help="Root dir (contains multiwoz_repo/)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output processed JSON path")
    parser.add_argument("--seed", type=int, default=123, help="Shuffle seed for train subsampling")
    parser.add_argument("--max-train", type=int, default=50, help="Max train examples per domain segment")
    parser.add_argument("--max-eval", type=int, default=10, help="Max eval examples per domain segment")
    parser.add_argument("--limit-domains", type=int, default=-1, help="Limit domains (-1 = all 5)")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Version string stored in output JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = repo_root / raw_root
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    convert_multiwoz_to_stream(
        raw_root=raw_root,
        out_path=out_path,
        max_train=int(args.max_train),
        max_eval=int(args.max_eval),
        seed=int(args.seed),
        limit_domains=int(args.limit_domains),
        domain_order=MULTIWOZ_DOMAIN_ORDER,
        mode=str(args.mode),
        version=str(args.version),
    )


if __name__ == "__main__":
    main()
