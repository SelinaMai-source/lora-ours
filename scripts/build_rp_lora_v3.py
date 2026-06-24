from __future__ import annotations

import argparse
import csv
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _fmt4(value: Any) -> str:
    return f"{_safe_float(value):.4f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _extract_missing_runs(summary_text: str) -> List[str]:
    lines = summary_text.splitlines()
    out: List[str] = []
    in_missing = False
    for raw in lines:
        line = raw.strip()
        if line == "## Missing Runs":
            in_missing = True
            continue
        if in_missing and line.startswith("## "):
            break
        if in_missing and line.startswith("- `"):
            out.append(line)
    return out


def _extract_summary_stat(summary_text: str, label: str) -> str:
    pattern = rf"- {re.escape(label)}:\s*`([^`]+)`"
    match = re.search(pattern, summary_text)
    return match.group(1).strip() if match else ""


def _extract_diag_value(report_text: str, key: str) -> str:
    pattern = rf"\|\s*{re.escape(key)}\s*\|\s*([^\|]+)\|"
    match = re.search(pattern, report_text)
    return match.group(1).strip() if match else ""


def _extract_diag_conclusion(report_text: str) -> str:
    match = re.search(
        r"### Q6 — Direct conclusion \(single primary bottleneck\)\s*\n\s*\n\*\*(.+?)\*\*",
        report_text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _extract_behavior_gate_line(progress_text: str) -> str:
    for line in progress_text.splitlines():
        if line.startswith("- Overfit exact match improved"):
            return line[2:].strip()
    return ""


def _lookup_row(rows: Sequence[Dict[str, str]], *, benchmark: str, method_label: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("benchmark")) == benchmark and str(row.get("method_label")) == method_label:
            return dict(row)
    return {}


def _main_results_table(main_rows: Sequence[Dict[str, str]]) -> str:
    headers = [
        "benchmark",
        "method",
        "final_step",
        "seen_avg",
        "forgetting",
        "token_f1",
        "oracle_agreement",
    ]
    
    expected_runs = [
        ("instrdialog", "Sequential"),
        ("instrdialog", "Replay(10)"),
        ("instrdialog", "Replay(50)"),
        ("instrdialog", "PeriodicLatest"),
        ("instrdialog", "BankNoRouter"),
        ("instrdialog", "RouterOnly"),
        ("instrdialog", "Ours"),
        ("instrdialog++", "Sequential"),
        ("instrdialog++", "Replay(10)"),
        ("instrdialog++", "Replay(50)"),
        ("instrdialog++", "PeriodicLatest"),
        ("instrdialog++", "BankNoRouter"),
        ("instrdialog++", "RouterOnly"),
        ("instrdialog++", "Ours"),
    ]
    
    rows = []
    for bench, method in expected_runs:
        # Find the row in main_rows
        found_row = None
        for r in main_rows:
            if str(r.get("benchmark")) == bench and str(r.get("method_label")) == method:
                found_row = r
                break
        
        if found_row:
            rows.append(
                [
                    bench,
                    method,
                    _fmt4(found_row.get("eval.current_score")),
                    _fmt4(found_row.get("eval.seen_avg_score")),
                    _fmt4(found_row.get("eval.forgetting")),
                    _fmt4(found_row.get("eval.token_f1_mean")),
                    _fmt4(found_row.get("routing.oracle_agreement_rate")),
                ]
            )
        else:
            rows.append(
                [
                    bench,
                    method,
                    "[TBD]",
                    "[TBD]",
                    "[TBD]",
                    "[TBD]",
                    "[TBD]",
                ]
            )
            
    return _markdown_table(headers, rows)


def _ablation_table(ablation_rows: Sequence[Dict[str, str]]) -> str:
    headers = [
        "method",
        "final_step",
        "seen_avg",
        "forgetting",
        "token_f1",
        "oracle_agreement",
    ]
    rows = []
    for row in ablation_rows:
        rows.append(
            [
                str(row.get("method_label", "")),
                _fmt4(row.get("eval.current_score")),
                _fmt4(row.get("eval.seen_avg_score")),
                _fmt4(row.get("eval.forgetting")),
                _fmt4(row.get("eval.token_f1_mean")),
                _fmt4(row.get("routing.oracle_agreement_rate")),
            ]
        )
    return _markdown_table(headers, rows)


def _build_markdown(repo_root: Path) -> str:
    results_dir = repo_root / "results"
    main_rows = _read_csv(results_dir / "tables" / "paper_main_results.csv")
    ablation_rows = _read_csv(results_dir / "tables" / "paper_ablation_results.csv")
    summary_text = _read_text(results_dir / "paper_results_summary.md")
    diag_text = _read_text(results_dir / "debug_report_sequence_behavior_diagnosis.md")
    progress_text = _read_text(results_dir / "rp_alignment_progress_report.md")

    ours_row = _lookup_row(main_rows, benchmark="instrdialog", method_label="Ours")
    bank_pp_row = _lookup_row(main_rows, benchmark="instrdialog++", method_label="BankNoRouter")
    baseline_rows_instr = [r for r in main_rows if str(r.get("benchmark")) == "instrdialog" and str(r.get("family")) == "baseline"]
    instrdialogpp_rows = [r for r in main_rows if str(r.get("benchmark")) == "instrdialog++"]
    best_seen_baseline = max(baseline_rows_instr, key=lambda r: _safe_float(r.get("eval.seen_avg_score"))) if baseline_rows_instr else {}
    best_token_baseline = max(baseline_rows_instr, key=lambda r: _safe_float(r.get("eval.token_f1_mean"))) if baseline_rows_instr else {}
    best_forgetting_baseline = min(baseline_rows_instr, key=lambda r: _safe_float(r.get("eval.forgetting", 1e9))) if baseline_rows_instr else {}
    router_only_row = _lookup_row(main_rows, benchmark="instrdialog", method_label="RouterOnly")
    no_router_row = _lookup_row(ablation_rows, benchmark="instrdialog", method_label="OursNoRouter")
    no_drift_row = _lookup_row(ablation_rows, benchmark="instrdialog", method_label="OursNoDrift")
    no_bank_row = _lookup_row(ablation_rows, benchmark="instrdialog", method_label="OursNoBank")

    missing_runs = _extract_missing_runs(summary_text)
    available_runs = _extract_summary_stat(summary_text, "available runs") or str(len(main_rows) + len(ablation_rows))
    missing_run_count = _extract_summary_stat(summary_text, "missing runs") or str(len(missing_runs))
    first_token_wrong = _extract_diag_value(diag_text, "count_first_token_wrong") or "n/a"
    extra_preamble = _extract_diag_value(diag_text, "count_extra_preamble") or "n/a"
    prefix_drift = _extract_diag_value(diag_text, "count_prefix_drift_1to5") or "n/a"
    behavior_conclusion = _extract_diag_conclusion(diag_text) or "sequence-level behavior mismatch"
    behavior_gate_line = _extract_behavior_gate_line(progress_text)
    ours_variant = str(ours_row.get("variant_id") or ours_row.get("method_variant") or "unknown")
    ours_descriptor = (
        "the current main-table `Ours` configuration (anti-overlap main method)"
        if "ours_full" in ours_variant
        else "the current main-table `Ours` configuration"
    )
    completed_instrdialogpp_methods = ", ".join(f"`{str(row.get('method_label'))}`" for row in instrdialogpp_rows) if instrdialogpp_rows else "none"

    lines: List[str] = [
        "<!-- Auto-generated by scripts/build_rp_lora_v3.py. -->",
        "# 1. Title",
        "TITLE: Task-Free Online LoRA with Drift Detection and Inference-Time Routing for Continual Instruction Tuning",
        "",
        "# 2. Abstract",
        (
            "Large language models deployed on non-stationary instruction streams need continual updates without task "
            "boundaries, replay-heavy storage, or full-model retraining. The current lora-code repository now implements "
            "a task-free online LoRA system built around curriculum-aware anchor drift detection, a growing LoRA bank, closed-loop "
            "routing, and anti-overlap regularization over activation and LoRA-weight space. Relative to RP(Lora)_v2, this v3 report updates the "
            "proposal into a code-grounded paper-style status document: drift is monitored with teacher-forced answer NLL on "
            "core/probe anchors under a calibrated one-sided CUSUM rule; branch allocation is handled by a LoRA bank that "
            "freezes prior adapters and spawns new branches on drift; router pseudo-labels are produced from per-branch answer "
            "NLL with margin filtering; and the current implementation supports routed per-example training assignments as well as "
            "per-example selection at evaluation time. On the currently completed single-seed InstrDialog runs, "
            f"{ours_descriptor} reaches seen_avg {_fmt4(ours_row.get('eval.seen_avg_score'))}, token_f1 {_fmt4(ours_row.get('eval.token_f1_mean'))}, "
            f"and oracle_agreement {_fmt4(ours_row.get('routing.oracle_agreement_rate'))}. The strongest completed InstrDialog baseline "
            f"on seen_avg is {best_seen_baseline.get('method_label', 'n/a')} ({_fmt4(best_seen_baseline.get('eval.seen_avg_score'))}); "
            f"the current overlap-enabled row must be re-run after the new curriculum anti-overlap fix before final claims. It also does not yet beat the strongest baseline on token_f1 ({best_token_baseline.get('method_label', 'n/a')}: "
            f"{_fmt4(best_token_baseline.get('eval.token_f1_mean'))}) or forgetting ({best_forgetting_baseline.get('method_label', 'n/a')}: "
            f"{_fmt4(best_forgetting_baseline.get('eval.forgetting'))}). Cross-benchmark claims should wait for refreshed InstrDialog++ runs under the "
            "ACL/ARR matrix. The updated report therefore presents both the "
            "implemented method and the most important remaining risks before a paper-ready claim set is possible."
        ),
        "",
        "# 3. Background and Motivation",
        "## 3.1 Real-world problem: non-stationary instruction streams",
        (
            "The repository is organized around continual instruction tuning on segmented CITB streams. New instruction data "
            "arrives sequentially, domains and response patterns change over time, and explicit task boundaries are not available "
            "at inference time. This makes the setting closer to online adaptation than to classical multi-task fine-tuning."
        ),
        "",
        "## 3.2 Why naive fine-tuning fails: catastrophic forgetting",
        (
            "Sequential LoRA remains a meaningful baseline in this codebase, but the current results still show that purely "
            "single-branch adaptation can preserve neither seen-average performance nor generation quality when the stream shifts. "
            "Replay helps forgetting, but it does so under explicit memory budgets and does not solve the task-free inference problem."
        ),
        "",
        "## 3.3 Why parameter-efficient adaptation (LoRA) is a good fit",
        (
            "The implementation keeps the backbone frozen and manipulates only LoRA adapters, which makes branch creation, freezing, "
            "and selective activation operationally cheap. This is what allows the bank-based design to stay practical inside a single "
            "training loop in `core/train.py`."
        ),
        "",
        "# 4. Related Work",
        (
            "The code and RP remain anchored in four literatures: LoRA / PEFT for efficient adaptation, continual instruction tuning "
            "benchmarks such as CITB, LoRA-bank and routing approaches for expert specialization, and task-free online continual "
            "learning based on change-point or dynamics-driven drift detection. The updated code path is closest in spirit to a "
            "task-free LoRA bank with explicit anchor monitoring, but it is more conservative than the original RP in what is actually "
            "implemented today."
        ),
        "",
        "# 5. Research Gap and Updated Questions",
        "## 5.1 Gap",
        (
            "Compared with RP(Lora)_v2, the main gap has shifted. The repository no longer asks whether drift detection, branch "
            "allocation, routing, and overlap control can be wired together at all; that integration now exists. The remaining gap is "
            "whether the current implementation delivers a robust trade-off across seen-average performance, forgetting, routing quality, "
            "and generation behavior, while also generalizing beyond a partially completed single-seed result matrix."
        ),
        "",
        "## 5.2 Updated questions",
        "- Can anchor-based drift monitoring improve seen-average performance without making the detector too conservative or too noisy?",
        "- Does a bank of frozen / active LoRA branches help enough to justify the extra routing and monitoring complexity?",
        "- Does the current router improve downstream performance beyond standalone routing agreement statistics?",
        "- How much of the remaining failure profile is caused by continual adaptation, and how much by open-loop sequence behavior?",
        "",
        "# 6. Proposed Method",
        "## 6.1 Overview",
        (
            "The current implementation uses a single frozen Llama-3.1-8B-Instruct backbone and a configurable set of modules: "
            "drift detector, LoRA bank, router, and overlap loss. The bank is initialized with branch `b0`; subsequent branches are "
            "created only through the bank manager. Evaluation becomes task-free by selecting a branch for each prompt when both the "
            "router and bank are enabled."
        ),
        "",
        "## 6.2 LoRA parameterization (core formula)",
        (
            "The code follows standard LoRA parameterization: DeltaW = BA with rank r, frozen base weights W, and trainable low-rank "
            "matrices A and B only. The wrapper in `core/models/lora_wrapper.py` exposes multi-adapter creation, freezing, and adapter-wise "
            "optimizer stepping so that the rest of the training loop can stay unified across baselines and our method."
        ),
        "",
        "## 6.3 Drift detection via anchor monitoring + curriculum-aware sequential change statistics",
        (
            "Anchors are constructed from eval examples across stream segments and split into core and probe subsets using "
            "K-Center Greedy feature sampling based on model activations, ensuring a diverse and representative anchor set. "
            "The current ACL/ARR version makes this split curriculum-aware: easier anchors form the stable core by default, "
            "while harder anchors are used as the probe side for change-point sensitivity. "
            "For each monitoring event, the model computes teacher-forced answer NLL on core and probe anchors. "
            "The detector smooths both streams with EMA, calibrates baselines and standard deviations over an initial window, and then applies "
            "a one-sided CUSUM-style update. A meta-threshold option learns the effective threshold from anchor volatility during calibration. "
            "A drift event is triggered only when probe degradation is sustained and the core subset also passes a guard condition. "
            "When the trigger fires, the active branch can be frozen, a new branch is spawned, and the anchor set is refreshed."
        ),
        (
            "Implementation note: the current code path performs anchor monitoring once per segment after evaluation, not every K optimization "
            "steps. The `monitor_interval` field is part of the detector configuration, but the operational monitoring schedule is segment-level "
            "in `run_ours(...)`. The reported drift quality is also a proxy metric because every inter-segment boundary is treated as a candidate shift."
        ),
        "",
        "## 6.4 Unified LoRA bank management + inference-time routing (closed-loop)",
        (
            "The LoRA bank tracks branch metadata, the active branch, frozen branches, and the maximum branch budget. "
            "When the capacity limit is reached, a LoRA Branch Merging mechanism is triggered, which performs weight averaging "
            "of the two most similar frozen branches to free up capacity. The router is a lightweight "
            "linear classifier over pooled prompt activations. During routed online training, the code can assign each training example to a branch "
            "using either the learned router or an oracle-min-NLL diagnostic policy. Router supervision scores each training example under every branch, "
            "uses the minimum answer NLL as a pseudo-label, and applies Dynamic Margin Filtering (where the margin linearly increases "
            "from 0.0 to 0.02 based on the number of updates) to ensure stable pseudo-label training. The router head also supports balance and "
            "orthogonality regularization so bank management and inference-time routing form one closed loop. During evaluation, the router extracts "
            "prompt features, predicts a branch, switches the active adapter, and then generates with that branch."
        ),
        "",
        "## 6.5 Anti-overlap regularization for LoRA specialization (main method)",
        (
            "The ACL/ARR version treats anti-overlap as part of the main method rather than an optional add-on. The loss combines "
            "activation-space diversity with LoRA weight-space orthogonality, uses a curriculum warmup over early segments, and scales "
            "with the number of branches so the regularizer does not overwhelm supervised adaptation immediately after a branch split. "
            "The same run records activation overlap, weight overlap, effective beta, and router-head orthogonality metrics."
        ),
        "",
        "# 7. Experimental Evaluation",
        "## 7.1 Benchmark and setup",
        (
            "The current matrix is built around CITB `instrdialog` and `instrdialog++` with Llama-3.1-8B-Instruct as the backbone. Replay budgets "
            "follow the planned 0/10/50 pattern, routing is hard top-1, and the main summary is still single-seed on the completed runs. At the "
            f"time of this v3 build, the refreshed artifact summary reports {available_runs} available runs and {missing_run_count} missing run "
            "across the current main+ablation package."
        ),
        "",
        "## 7.2 Baselines",
        "- Sequential LoRA (single branch, no drift, no router)",
        "- Replay LoRA with memory budgets 10 and 50",
        "- Periodic Multi-LoRA (latest-only inference)",
        "- Bank + no router (drift + bank, latest-only inference)",
        "- Router-only (fixed branches with routing, no drift detector)",
        "- Ours-family variants (drift + bank + routed online training + anti-overlap; `OursFull` is the ACL/ARR main method, while `OursNoOverlap` is the key ablation)",
        "",
        "## 7.3.1 Metrics",
        "- Final-step performance: `eval.current_score`",
        "- Seen-average performance / anytime score: `eval.seen_avg_score`, `eval.anytime_score`",
        "- Forgetting: `eval.forgetting`",
        "- Open-loop quality diagnostics: `eval.token_f1_mean`, `eval.lcs_overlap_mean`, prefix matches",
        "- Routing quality: oracle agreement, decision confidence, entropy, oracle margin, branch utilization",
        "- Drift proxy quality: false-alarm rate, miss rate, detection delay mean",
        "- Specialization proxy: `overlap_mean_cosine`",
        "",
        "## 7.3.2 Figures",
        (
            "The current figure set is generated by `scripts/build_paper_artifacts.py` and reuses the RP(Lora)_v2 numbering. To keep this draft "
            "focused on textual and numerical alignment, the figure assets are not embedded below; this section records the intended figure content only. "
            "Figure 3 currently emphasizes memory-budget performance; branch-budget sweeps remain available in the paper matrix configs but are not yet "
            "separated into a standalone panel in the generated figure set."
        ),
        "",
        "Figure 1 would show anytime performance over seen segments for the currently available main runs.",
        "",
        "Figure 2 would summarize mean forgetting across the currently available main runs.",
        "",
        "Figure 3 would compare performance against memory budget on the currently available matrix rows.",
        "",
        "Figure 4 would visualize drift diagnostics from the current main-table `Ours` row on InstrDialog.",
        "",
        "Figure 5 would compare router-oracle agreement and branch utilization for the currently available routed runs.",
        "",
        "Figure 6 would depict the current closed-loop pipeline schematic used in the artifact builder.",
        "",
        "## 7.4 Main Results",
        (
            f"On `instrdialog`, {ours_descriptor} is currently the best completed method on seen-average performance "
            f"({_fmt4(ours_row.get('eval.seen_avg_score'))}), ahead of {best_seen_baseline.get('method_label', 'n/a')} "
            f"at {_fmt4(best_seen_baseline.get('eval.seen_avg_score'))}. "
            f"It does not beat the strongest baseline on token F1 ({best_token_baseline.get('method_label', 'n/a')}: "
            f"{_fmt4(best_token_baseline.get('eval.token_f1_mean'))}) or forgetting "
            f"({best_forgetting_baseline.get('method_label', 'n/a')}: {_fmt4(best_forgetting_baseline.get('eval.forgetting'))}). "
            f"The current `Ours` row is reported from variant `{ours_variant}`; for ACL/ARR, this should be the overlap-enabled `ours_full` configuration. "
            f"RouterOnly reaches the highest completed oracle agreement ({_fmt4(router_only_row.get('routing.oracle_agreement_rate'))}) "
            "but does not translate that into downstream accuracy, which suggests that branch selection quality alone is not sufficient."
        ),
        (
            f"On `instrdialog++`, the currently completed main-table rows are {completed_instrdialogpp_methods}. "
            f"Among those available rows, BankNoRouter reaches final-step {_fmt4(bank_pp_row.get('eval.current_score'))}, "
            f"seen_avg {_fmt4(bank_pp_row.get('eval.seen_avg_score'))}, and forgetting {_fmt4(bank_pp_row.get('eval.forgetting'))}. "
            "The ACL/ARR matrix now keeps the overlap-enabled Ours row as the declared main method on both benchmarks; cross-benchmark claims should "
            "still wait for refreshed runs under the curriculum anti-overlap implementation."
        ),
        "",
        "Table 1. Main comparison on the currently completed matrix rows.",
        _main_results_table(main_rows),
        "",
        "## 7.5 Ablations and diagnostics",
        (
            f"Compared with the current main-table `Ours` row ({ours_variant}), removing the router lowers seen_avg to "
            f"{_fmt4(no_router_row.get('eval.seen_avg_score'))}, while turning off drift or bank lowers it to "
            f"{_fmt4(no_drift_row.get('eval.seen_avg_score'))} and {_fmt4(no_bank_row.get('eval.seen_avg_score'))}, respectively. "
            "The key anti-overlap ablation is now `OursNoOverlap`, which removes the specialization loss while keeping drift, bank, and routing active. "
            f"The overlap-enabled row reaches seen_avg {_fmt4(ours_row.get('eval.seen_avg_score'))} and token_f1 {_fmt4(ours_row.get('eval.token_f1_mean'))} "
            "under the current available matrix; this row must be re-run after the curriculum anti-overlap fix before making final paper claims."
        ),
        "",
        "Table 2. Key ablations on InstrDialog.",
        _ablation_table(ablation_rows),
        "",
        (
            f"The current main-table `Ours` row on InstrDialog reports false_alarm_rate {_fmt4(ours_row.get('drift.false_alarm_rate'))}, "
            f"miss_rate {_fmt4(ours_row.get('drift.miss_rate'))}, and detection_delay_mean {_fmt4(ours_row.get('drift.detection_delay_mean'))}. "
            "This indicates a conservative detector that is not firing often enough under the present proxy definition of shifts."
        ),
        (
            f"Sequence-behavior diagnostics remain important. The latest diagnosis report attributes the dominant bottleneck to "
            f"'{behavior_conclusion}', with failure counts first_token_wrong={first_token_wrong}, extra_preamble={extra_preamble}, "
            f"and prefix_drift_1to5={prefix_drift}. {behavior_gate_line if behavior_gate_line else ''}"
        ).strip(),
        "",
        "## 7.6 Current coverage gaps",
    ]

    if missing_runs:
        lines.extend(missing_runs)
    else:
        lines.append("- No missing runs were reported in the current paper summary.")

    lines.extend(
        [
            "",
            "# 8. Current Contributions",
            "- A unified, config-driven continual instruction tuning pipeline that covers baselines and the full bank/routing method inside one code path.",
            "- A concrete implementation of curriculum-aware core/probe anchor monitoring, LoRA bank management, routed online training, and curriculum anti-overlap regularization over both activation and weight space.",
            "- Script-generated tables, figures, and packaging outputs that can be refreshed without interrupting ongoing training runs.",
            "- Additional behavior diagnostics that help separate continual-learning failures from open-loop generation failures.",
            "",
            "# 9. Risks and Mitigations",
            "- Risk 1: detector conservativeness. Current miss rate is high under the proxy shift definition. Mitigation: complete the planned threshold / anchor-size / monitor-interval sweeps and refine the proxy labeling scheme before making strong drift claims.",
            "- Risk 2: router training stability. Routed online training is now configurable, but learned-router assignments can still collapse. Mitigation: report branch utilization, router balance loss, and oracle-min-NLL diagnostic runs.",
            "- Risk 3: forgetting remains weaker than replay-heavy baselines. Mitigation: treat seen-average and specialization gains as the current strength, not forgetting superiority, until more runs are complete.",
            "- Risk 4: sequence-level behavior bottlenecks can contaminate continual-learning conclusions. Mitigation: keep the behavior gate and overfit diagnostics in the evaluation loop before promoting paper-grade claims.",
            "- Risk 5: incomplete benchmark coverage and single-seed evidence. Mitigation: finish the missing InstrDialog++ runs and expand the validated winner to multi-seed evaluation before final packaging.",
            "",
            "# 10. References",
            "- Hu, E. J., et al. LoRA: Low-Rank Adaptation of Large Language Models.",
            "- Wei, X., Li, G., and Marculescu, R. Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation.",
            "- Zhang, Z., Fang, M., Chen, L., and Namazi-Rad, M.-R. CITB: A Benchmark for Continual Instruction Tuning.",
            "- Kirkpatrick, J., et al. Overcoming catastrophic forgetting in neural networks.",
            "- Liang, Y.-S., and Li, W.-J. GainLoRA: Gated Integration of Low-Rank Adaptation for Continual Learning of Language Models.",
            "- Han, J., et al. SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture.",
            "- Page, E. S. CUSUM and sequential change-point monitoring.",
            "- Soviany, P., Ionescu, R. T., Rota, P., and Sebe, N. Curriculum Learning: A Survey.",
            "- Lopes, J. F., et al. Online Meta-Recommendation of CUSUM Hyperparameters.",
            "- Kim, H. Geometric Regularization in Mixture-of-Experts: The Disconnect Between Weights and Activations.",
            "- Mu, S., and Lin, S. A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications.",
            "",
        ]
    )
    return "\n".join(lines)


def _strip_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "")
    text = text.replace("*", "")
    return text


def _wrap_text(text: str, width: int) -> List[str]:
    cleaned = _strip_inline_markdown(" ".join(text.split()))
    return textwrap.wrap(cleaned, width=width) or [""]


def _parse_table(block_lines: Sequence[str]) -> tuple[List[str], List[List[str]]]:
    rows: List[List[str]] = []
    for line in block_lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2 and all(set(cell) <= {"-", ":"} for cell in rows[1]):
        rows = [rows[0]] + rows[2:]
    headers = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []
    return headers, body


def _parse_markdown(md_text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        line = raw.strip()
        if not line or line.startswith("<!--"):
            i += 1
            continue
        if line.startswith("TITLE:"):
            blocks.append({"type": "title", "text": line.split(":", 1)[1].strip()})
            i += 1
            continue
        if line.startswith("![") and "](" in line and line.endswith(")"):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                blocks.append({"type": "image", "caption": match.group(1).strip(), "path": match.group(2).strip()})
                i += 1
                continue
        if re.match(r"^#{1,6}\s+", line):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            blocks.append({"type": "heading", "level": level, "text": text})
            i += 1
            continue
        if line.startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            headers, rows = _parse_table(table_lines)
            blocks.append({"type": "table", "headers": headers, "rows": rows})
            continue
        if line.startswith("- "):
            items = [line[2:].strip()]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append({"type": "bullets", "items": items})
            continue
        paragraph_lines = [line]
        i += 1
        while i < len(lines):
            peek = lines[i].strip()
            if (
                not peek
                or peek.startswith("<!--")
                or peek.startswith("TITLE:")
                or peek.startswith("![")
                or peek.startswith("|")
                or peek.startswith("- ")
                or re.match(r"^#{1,6}\s+", peek)
            ):
                break
            paragraph_lines.append(peek)
            i += 1
        blocks.append({"type": "paragraph", "text": " ".join(paragraph_lines)})
    return blocks


class _PdfRenderer:
    def __init__(self, out_path: Path, repo_root: Path):
        self.out_path = out_path
        self.repo_root = repo_root
        self.pdf = PdfPages(str(out_path))
        self.page_num = 0
        self.fig = None
        self.ax = None
        self.y = 0.94
        self._new_page()

    def close(self) -> None:
        self._save_page()
        self.pdf.close()
        plt.close("all")

    def _new_page(self) -> None:
        if self.fig is not None:
            self._save_page()
        self.page_num += 1
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.y = 0.94
        self.ax.text(0.5, 0.975, "RP(Lora)_v3", ha="center", va="top", fontsize=10, fontweight="bold")

    def _save_page(self) -> None:
        if self.fig is None or self.ax is None:
            return
        self.ax.text(
            0.5,
            0.02,
            f"RP(Lora)_v3    {self.page_num}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
        self.pdf.savefig(self.fig)

    def _ensure_space(self, needed: float) -> None:
        if self.y - needed < 0.08:
            self._new_page()

    def draw_heading(self, text: str, level: int) -> None:
        if level <= 1:
            fontsize = 14
            gap = 0.032
        else:
            fontsize = 12
            gap = 0.028
        lines = _wrap_text(text, 70 if level <= 1 else 85)
        needed = gap * len(lines) + 0.01
        self._ensure_space(needed)
        for line in lines:
            self.ax.text(0.08, self.y, _strip_inline_markdown(line), ha="left", va="top", fontsize=fontsize, fontweight="bold")
            self.y -= gap
        self.y -= 0.006

    def draw_title(self, text: str) -> None:
        lines = _wrap_text(text, 58)
        needed = 0.04 * len(lines) + 0.02
        self._ensure_space(needed)
        for line in lines:
            self.ax.text(0.08, self.y, line, ha="left", va="top", fontsize=18, fontweight="bold")
            self.y -= 0.04
        self.y -= 0.01

    def draw_paragraph(self, text: str, *, fontsize: int = 10) -> None:
        lines = _wrap_text(text, 105)
        line_gap = 0.02 if fontsize <= 10 else 0.023
        needed = line_gap * len(lines) + 0.006
        self._ensure_space(needed)
        for line in lines:
            self.ax.text(0.08, self.y, line, ha="left", va="top", fontsize=fontsize, family="DejaVu Serif")
            self.y -= line_gap
        self.y -= 0.006

    def draw_bullets(self, items: Sequence[str]) -> None:
        for item in items:
            wrapped = textwrap.wrap(_strip_inline_markdown(item), width=98) or [""]
            needed = 0.02 * len(wrapped) + 0.004
            self._ensure_space(needed)
            first = True
            for line in wrapped:
                prefix = "• " if first else "  "
                self.ax.text(0.09, self.y, prefix + line, ha="left", va="top", fontsize=10, family="DejaVu Serif")
                self.y -= 0.02
                first = False
            self.y -= 0.002
        self.y -= 0.004

    def draw_table(self, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        n_rows = len(rows) + 1
        height = min(0.048 * n_rows + 0.02, 0.62)
        self._ensure_space(height + 0.02)
        if self.y < 0.72:
            self._new_page()
        bbox = [0.07, self.y - height, 0.86, height]
        table = self.ax.table(cellText=list(rows), colLabels=list(headers), cellLoc="center", loc="center", bbox=bbox)
        table.auto_set_font_size(False)
        table.set_fontsize(7.6)
        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_text_props(weight="bold")
        self.y -= height + 0.02

    def draw_image(self, relative_path: str, caption: str) -> None:
        image_path = (self.repo_root / relative_path).resolve()
        if not image_path.is_file():
            self.draw_paragraph(f"[Missing figure: {relative_path}]", fontsize=10)
            return
        if self.y < 0.72:
            self._new_page()
        img = mpimg.imread(str(image_path))
        h, w = img.shape[0], img.shape[1]
        aspect = h / max(1, w)
        max_height = min(0.56, self.y - 0.16)
        width = min(0.88, max_height / max(aspect, 1e-6))
        height = width * aspect
        if height > max_height:
            height = max_height
            width = height / max(aspect, 1e-6)
        x = (1.0 - width) / 2.0
        y0 = self.y - height
        img_ax = self.fig.add_axes([x, y0, width, height])
        img_ax.imshow(img)
        img_ax.axis("off")
        self.y = y0 - 0.012
        self.draw_paragraph(caption, fontsize=9)


def _render_pdf(md_text: str, repo_root: Path, out_path: Path) -> None:
    renderer = _PdfRenderer(out_path=out_path, repo_root=repo_root)
    try:
        for block in _parse_markdown(md_text):
            block_type = block["type"]
            if block_type == "heading":
                renderer.draw_heading(block["text"], block["level"])
            elif block_type == "title":
                renderer.draw_title(block["text"])
            elif block_type == "paragraph":
                renderer.draw_paragraph(block["text"])
            elif block_type == "bullets":
                renderer.draw_bullets(block["items"])
            elif block_type == "table":
                renderer.draw_table(block["headers"], block["rows"])
            elif block_type == "image":
                renderer.draw_image(block["path"], block["caption"])
    finally:
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RP(Lora)_v3 markdown and PDF from current artifacts.")
    parser.add_argument("--source-out", type=Path, default=Path("RP(Lora)_v3.md"))
    parser.add_argument("--pdf-out", type=Path, default=Path("RP(Lora)_v3.pdf"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_out = (repo_root / args.source_out).resolve()
    pdf_out = (repo_root / args.pdf_out).resolve()

    markdown = _build_markdown(repo_root)
    source_out.write_text(markdown + "\n", encoding="utf-8")
    _render_pdf(markdown, repo_root=repo_root, out_path=pdf_out)
    print(f"[rp_v3] wrote source={source_out.relative_to(repo_root)} pdf={pdf_out.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
