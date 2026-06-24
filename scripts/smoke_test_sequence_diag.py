from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.models.base_model import apply_prefix_mixing_to_supervised_tokens
from core.overfit_sequence_diagnostics import write_sequence_behavior_report
from core.run_artifacts import collect_overfit_stale_artifacts, init_overfit_run_manifest
from core.train_labels import build_supervised_labels, supervised_span_from_labels


class TinyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 99
        self.all_special_ids = [0, 99]
        self._id2tok = {
            0: "<pad>",
            1: "<sp>",
            2: "hello",
            3: "world",
            4: "A",
            5: "B",
            6: "C",
            7: "prefix",
            8: "answer",
            99: "<eos>",
        }
        self._tok2id = {v: k for k, v in self._id2tok.items()}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        user = messages[0]["content"]
        if len(messages) == 1:
            return f"USER:{user}\nASSISTANT: "
        return f"USER:{user}\nASSISTANT: {messages[1]['content']}"

    def _encode(self, text: str):
        t = str(text or "")
        ids = []
        if t.startswith(" "):
            ids.append(1)
            t = t[1:]
        for p in [x for x in t.replace("\n", " ").split(" ") if x]:
            ids.append(self._tok2id.get(p, 7))
        return ids

    def encode(self, text: str, add_special_tokens=False):
        return self._encode(text)

    def __call__(self, text: str, add_special_tokens=False, truncation=False):
        return {"input_ids": self._encode(text)}

    def decode(self, ids, skip_special_tokens=False):
        toks = [self._id2tok.get(int(i), "?") for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in {"<pad>", "<eos>"}]
        return " ".join(toks)

    def convert_ids_to_tokens(self, ids):
        return [self._id2tok.get(int(i), "?") for i in ids]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_first_token_gold_source() -> None:
    tok = TinyTokenizer()
    enc = build_supervised_labels(
        tok,
        instruction="hello",
        input_text="world",
        target=" A B",
        max_len=128,
        min_target_tokens=1,
        mask_eos_token_in_labels=False,
        mask_all_special_tokens_in_labels=False,
        labeling_mode="manual",
        completion_only_response_template="ASSISTANT: ",
    )
    sup = supervised_span_from_labels(tokenizer=tok, full_ids=enc.full_ids, labels=enc.labels, preview_tokens=3)
    first_pos = next(i for i, v in enumerate(enc.labels) if v != -100)
    _assert(sup["first_supervised_token_position"] == first_pos, "supervised_span position mismatch")
    _assert(sup["first_supervised_token_id"] == enc.full_ids[first_pos], "supervised_span token mismatch")
    raw_first = tok.encode("A B", add_special_tokens=False)[0]
    raw_space_first = tok.encode(" A B", add_special_tokens=False)[0]
    _assert(raw_first != raw_space_first, "tokenizer should distinguish leading-space boundary")
    _assert(
        sup["first_supervised_token_id"] in (raw_first, raw_space_first),
        "supervised token should align to one tokenization boundary variant",
    )


def test_fresh_start_cleanup_and_manifest() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "debug" / "overfit8"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "lora_ckpt").mkdir()
        for name in [
            "overfit_state.json",
            "overfit8_steps.csv",
            "first_token_margin_over_time.csv",
            "first_token_margin_aggregate_by_step.csv",
            "prefix_rollout_summary.csv",
            "decode_ablation.csv",
            "decode_ablation.json",
            "step_0001_failure_analysis.json",
            "step_0001_failure_summary.csv",
            "step_0001_first_token_audit.json",
            "step_0001_prefix_rollout.json",
            "step_0001_predictions.json",
        ]:
            (out_dir / name).write_text("x", encoding="utf-8")
        stale = collect_overfit_stale_artifacts(out_dir)
        _assert(len(stale) >= 10, "stale artifacts not fully discovered")

        cfg = {"mode": "debug", "paths": {"sample_stream_path": "data/sample.json"}, "model": {}, "lora": {}}
        manifest = init_overfit_run_manifest(
            cfg=cfg,
            run_id="smoke",
            config_path="configs/x.yaml",
            config_snapshot_path=str(Path(td) / "config_snapshot.yaml"),
            run_dir=Path(td),
            debug_tools={},
        )
        _assert("run_id" in manifest and manifest["run_id"] == "smoke", "manifest run_id missing")
        _assert("artifacts_generated_in_run" in manifest, "manifest artifacts list missing")


def test_scheduled_sampling_mixing_alignment() -> None:
    full_ids = [10, 11, 12, 13, 14]
    labels = [-100, -100, 12, 13, 14]
    mixed = apply_prefix_mixing_to_supervised_tokens(
        full_ids=full_ids,
        labels=labels,
        self_prefix_ids=[21, 22],
        prefix_k=2,
    )
    _assert(mixed["mixed_full_ids"][:2] == [10, 11], "prompt tokens should not change")
    _assert(mixed["mixed_labels"][:2] == [-100, -100], "prompt labels should remain masked")
    _assert(mixed["mixed_full_ids"][2:4] == [21, 22], "first supervised prefix should be mixed")
    _assert(mixed["mixed_labels"][2:4] == [21, 22], "mixed labels should align with mixed ids")
    _assert(mixed["mixed_labels"][4] == 14, "gold continuation should remain supervised")


def test_report_reads_manifest_only() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        overfit_dir = root / "debug" / "overfit8"
        overfit_dir.mkdir(parents=True, exist_ok=True)
        good = overfit_dir / "step_0001_failure_summary.csv"
        bad = overfit_dir / "step_9999_failure_summary.csv"
        with good.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["step", "num_samples", "raw_em_count"])
            w.writeheader()
            w.writerow({"step": 1, "num_samples": 8, "raw_em_count": 1})
        with bad.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["step", "num_samples", "raw_em_count"])
            w.writeheader()
            w.writerow({"step": 9999, "num_samples": 8, "raw_em_count": 8})
        manifest_path = overfit_dir / "run_manifest.json"
        manifest_path.write_text(
            json.dumps({"artifacts_generated_in_run": [str(good)]}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_sequence_behavior_report(
            run_manifest_path=manifest_path,
            results_dir=root / "results",
            experiment_name="smoke",
            run_id="smoke",
        )
        report = (root / "results" / "debug_report_sequence_behavior_diagnosis.md").read_text(encoding="utf-8")
        _assert("step_0001_failure_summary.csv" in report, "manifest-listed summary should be used")
        _assert("step_9999_failure_summary.csv" not in report, "stale summary should not be scanned")


def main() -> None:
    test_first_token_gold_source()
    test_fresh_start_cleanup_and_manifest()
    test_scheduled_sampling_mixing_alignment()
    test_report_reads_manifest_only()
    out_dir = Path("results/smoke_test_sequence_diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_test_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "checks": [
                    "first_token_gold_source",
                    "fresh_start_cleanup_and_manifest",
                    "scheduled_sampling_prefix_mixing_alignment",
                    "report_reads_manifest_only",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
