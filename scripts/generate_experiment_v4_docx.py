#!/usr/bin/env python3
"""Generate a paper-style experiment report for advisor presentation.

The document is intentionally structured like a CCF-A paper report instead of a
plain lab log. It uses only readable text formulas and black font throughout.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = PROJECT_ROOT / "文档记录"
CHART_DIR = DOC_DIR / "_charts_v4"
REPORT_DATE = datetime.now().strftime("%Y-%m-%d")
BLACK = RGBColor(0, 0, 0)

RUN_FILES = {
    "Ours v8_sota_5": PROJECT_ROOT
    / "results/runs/paper_instrdialog_ours_full_s123_v8_sota_5/final_metrics.json",
    "Ours v8_sota_8": PROJECT_ROOT
    / "results/runs/paper_instrdialog_ours_full_s123_v8_sota_8/final_metrics.json",
    "Ours v6_sota_2": PROJECT_ROOT
    / "results/runs/paper_instrdialog_ours_full_s123_v6_sota_2/final_metrics.json",
    "Ours v4_sota_39": PROJECT_ROOT
    / "results/runs/paper_instrdialog_ours_full_s123_v4_sota_39/final_metrics.json",
    "Sequential": PROJECT_ROOT / "results/runs/paper_instrdialog_seq_s123_true_fair/final_metrics.json",
    "Replay(50)": PROJECT_ROOT
    / "results/runs/paper_instrdialog_replay_b50_s123_true_fair/final_metrics.json",
    "PeriodicLatest": PROJECT_ROOT
    / "results/runs/paper_instrdialog_periodic_latest_s123_true_fair/final_metrics.json",
    "BankNoRouter": PROJECT_ROOT
    / "results/runs/paper_instrdialog_bank_no_router_s123_true_fair/final_metrics.json",
    "RouterOnly": PROJECT_ROOT
    / "results/runs/paper_instrdialog_router_only_s123_true_fair/final_metrics.json",
}

SEGMENT_FILES = {
    "v8_sota_5": PROJECT_ROOT / "results/tables/paper_instrdialog_ours_full_s123_v8_sota_5_segment_metrics.csv",
    "v8_sota_8": PROJECT_ROOT / "results/tables/paper_instrdialog_ours_full_s123_v8_sota_8_segment_metrics.csv",
    "v6_sota_2": PROJECT_ROOT / "results/tables/paper_instrdialog_ours_full_s123_v6_sota_2_segment_metrics.csv",
    "PeriodicLatest": PROJECT_ROOT
    / "results/tables/paper_instrdialog_periodic_latest_s123_true_fair_segment_metrics.csv",
    "Sequential": PROJECT_ROOT / "results/tables/paper_instrdialog_seq_s123_true_fair_segment_metrics.csv",
}

MAIN_METRICS = [
    ("eval.seen_avg_score", "Seen"),
    ("eval.seen_avg_task_aware_score", "Task-aware"),
    ("eval.forgetting", "Forgetting"),
    ("eval.task_aware_forgetting", "TA forgetting"),
    ("eval.token_f1_mean", "Token F1"),
    ("eval.lcs_overlap_mean", "LCS"),
]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "-"
    return f"{number:.{digits}f}"


def load_run(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    final = dict(data.get("final", {}))
    for section, prefix in [("drift_quality", "drift"), ("routing_quality", "routing")]:
        for key, value in (data.get(section) or {}).items():
            final[f"{prefix}.{key}"] = value
    return final


def load_all_runs() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, path in RUN_FILES.items():
        if path.is_file():
            out[name] = load_run(path)
        else:
            out[name] = {}
    return out


def resolve_docx_path() -> Path:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    version = 4
    while True:
        candidate = DOC_DIR / f"实验v{version}.docx"
        if not candidate.exists():
            return candidate
        version += 1


def set_document_style(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BLACK
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    for level in range(1, 4):
        style = styles[f"Heading {level}"]
        style.font.name = "Times New Roman"
        style.font.color.rgb = BLACK
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def set_run_font(run, east_asia: str = "Times New Roman", latin: str = "Times New Roman", size: Optional[float] = None) -> None:
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.color.rgb = BLACK
    if size is not None:
        run.font.size = Pt(size)


def add_title(doc: Document, title: str, subtitle: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    run.bold = True
    set_run_font(run, size=18)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(subtitle)
    set_run_font(run, size=10.5)
    run.italic = True


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        set_run_font(run)


def add_para(doc: Document, text: str, bold: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    set_run_font(run)


def add_bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    set_run_font(run)


def add_formula(doc: Document, label: str, formula: str, explanation: Optional[str] = None) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"{label}: {formula}")
    run.bold = True
    set_run_font(run, size=10)
    if explanation:
        add_para(doc, explanation)


def add_table(doc: Document, headers: List[str], rows: List[List[str]], widths: Optional[List[float]] = None) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = header
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                set_run_font(run, size=9.5)
        if widths:
            cell.width = Cm(widths[idx])

    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = value
            for paragraph in cells[idx].paragraphs:
                for run in paragraph.runs:
                    set_run_font(run, size=9)
            if widths:
                cells[idx].width = Cm(widths[idx])
    doc.add_paragraph()


def read_trajectory(path: Path) -> Tuple[List[int], List[float], List[float]]:
    segs: List[int] = []
    seen: List[float] = []
    ta: List[float] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segs.append(int(row["segment_id"]))
            seen.append(float(row.get("eval.seen_avg_score", 0.0)))
            ta.append(float(row.get("eval.seen_avg_task_aware_score") or row.get("eval.seen_avg_score") or 0.0))
    return segs, seen, ta


def create_charts(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Path]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    paths: Dict[str, Path] = {}

    names = ["Sequential", "Replay(50)", "PeriodicLatest", "BankNoRouter", "RouterOnly", "Ours v8_sota_5"]
    seen_vals = [runs[n].get("eval.seen_avg_score", 0.0) for n in names]
    ta_vals = [runs[n].get("eval.seen_avg_task_aware_score", 0.0) for n in names]
    x = range(len(names))

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.bar([i - 0.18 for i in x], seen_vals, width=0.36, color="#4d4d4d", label="Seen")
    ax.bar([i + 0.18 for i in x], ta_vals, width=0.36, color="#b3b3b3", label="Task-aware")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.65)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Final performance against true-fair baselines")
    fig.tight_layout()
    path = CHART_DIR / "v4_main_results.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["main_results"] = path

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    for name, style in [
        ("v8_sota_5", "-"),
        ("v8_sota_8", "--"),
        ("v6_sota_2", ":"),
        ("PeriodicLatest", "-."),
        ("Sequential", (0, (5, 5))),
    ]:
        csv_path = SEGMENT_FILES.get(name)
        if not csv_path or not csv_path.is_file():
            continue
        segs, seen, _ = read_trajectory(csv_path)
        ax.plot(segs, seen, linestyle=style, color="black", linewidth=1.5, label=name)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Seen avg score")
    ax.set_ylim(0, 0.65)
    ax.set_title("Seen trajectory")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = CHART_DIR / "v4_seen_trajectory.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["seen_trajectory"] = path

    return paths


def metric_rows(runs: Dict[str, Dict[str, Any]], names: Iterable[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for name in names:
        row = [name]
        for key, _label in MAIN_METRICS:
            row.append(fmt(runs.get(name, {}).get(key)))
        row.append(fmt(runs.get(name, {}).get("routing.oracle_agreement_rate")))
        rows.append(row)
    return rows


def add_method_section(doc: Document) -> None:
    add_heading(doc, "3  Method: Task-Free Online LoRA Bank with Prototype Routing", 1)
    add_para(
        doc,
        "The method addresses task-free continual instruction tuning. The data stream arrives segment by segment; no task ID is used during training or inference. "
        "The core design freezes the base LLM, maintains a dynamic LoRA bank, allocates new capacity through drift detection, selects or mixes branches through a prototype router, "
        "and applies anti-overlap regularization to reduce redundant branch behavior.",
    )

    add_heading(doc, "3.0 Contributions in the best run", 2)
    add_para(
        doc,
        "The closest-to-target internal result reported here is v8_sota_5. Its method is not a generic LoRA bank baseline: it is the v8_sota_4 spawn-synchronized prototype router plus one targeted change, namely multi-pass prototype EMA refinement within each segment. "
        "The implementation evidence is in core/methods/router.py, core/train.py, core/methods/overlap_loss.py, configs/paper/v8_sota_5.yaml, and archive/v8_sota_5/FAILURE_ANALYSIS.md.",
    )
    add_table(
        doc,
        ["Contribution", "Implementation detail", "Why it matters"],
        [
            [
                "Drift-anchored LoRA bank",
                "When drift is detected, old branches are frozen and a new branch is spawned. Branches and their prototypes are kept consistent.",
                "This separates adaptation phases and avoids direct overwriting of earlier LoRA parameters.",
            ],
            [
                "Spawn-synchronized prototype initialization",
                "At spawn time, the new branch prototype is initialized from the centroid of core and probe anchor features.",
                "This reduces cold-start routing failure for newly created branches.",
            ],
            [
                "Self-consistency pseudo-label routing",
                "For each segment example, all branches are scored by teacher-forced answer NLL; the branch with the minimum NLL becomes the pseudo-label.",
                "The router is trained without task IDs and uses the LoRA bank itself as the supervision source.",
            ],
            [
                "Multi-pass prototype EMA refinement",
                "v8_sota_5 sets prototype_update_steps=3 and repeats pseudo-label prototype updates three times after each segment.",
                "This is the unique v8_sota_5 change; it improves within-segment prototype fit without increasing LoRA training epochs.",
            ],
            [
                "Anti-overlap specialization",
                "The training step adds activation-space and weight-space overlap penalties with beta=0.05.",
                "This discourages branch redundancy and makes prototype routing more meaningful.",
            ],
        ],
        widths=[3.6, 6.0, 6.0],
    )

    add_heading(doc, "3.1 Problem formulation", 2)
    add_formula(
        doc,
        "Eq. 1",
        "D = {S_1, S_2, ..., S_T},  S_t = {(x_i, y_i)}",
        "Each S_t is an online instruction segment. After processing S_t, the model is evaluated on all previously seen segments without relying on task identifiers.",
    )
    add_formula(
        doc,
        "Eq. 2",
        "Score_t = (1 / t) * sum_{j=1..t} Acc_j(theta_t)",
        "The report uses strict seen average and task-aware seen average as complementary measurements of retained performance.",
    )

    add_heading(doc, "3.2 LoRA branch parameterization", 2)
    add_formula(
        doc,
        "Eq. 3",
        "W'_b = W + Delta W_b,  Delta W_b = B_b A_b,  rank(A_b, B_b) = r",
        "The base weight W is frozen. Each branch b trains only the low-rank LoRA parameters A_b and B_b. The v8_sota_5 run uses r=16, alpha=32, and targets q_proj and v_proj.",
    )

    add_heading(doc, "3.3 Drift-triggered LoRA bank", 2)
    add_para(
        doc,
        "The system maintains a bank B = {b_1, ..., b_m}. When anchor NLL statistics indicate a sustained distribution shift, the current branch is frozen and a new branch is spawned. "
        "The v8_sota_5 run uses train-split anchors, anchor_size=64, the easy_core_hard_probe curriculum, threshold=0.025, and disables oracle spawning from segment boundaries.",
    )
    add_formula(
        doc,
        "Eq. 4",
        "C_t = max(0, C_{t-1} + (s_t - s_ref) - kappa)",
        "Here s_t is the mean answer NLL on the probe or core anchors, and kappa is the slack term. A drift event is triggered when the probe-side CUSUM exceeds its threshold.",
    )
    add_formula(
        doc,
        "Eq. 5",
        "if C_t >= tau: freeze(b_m), spawn(b_{m+1}), trainable = b_{m+1}",
        "Freezing old branches prevents direct overwriting of earlier capabilities, while the newly spawned branch absorbs the new distribution.",
    )
    add_formula(
        doc,
        "Eq. 6",
        "p_new = normalize((1 / M) * sum_{a in A_core union A_probe} z(a))",
        "Spawn-synchronized initialization sets the new branch prototype to the normalized centroid of anchor features. This is implemented through init_prototype_from_anchor_features after branch spawning.",
    )

    add_heading(doc, "3.4 Self-consistency prototype routing", 2)
    add_para(
        doc,
        "The v8 series moves from a conventional learned head to a prototype router. Each branch maintains a normalized feature prototype p_b. "
        "After training, teacher-forced NLL is used to produce self-consistency pseudo-labels over existing branches, and the prototypes are updated by EMA. "
        "The key contribution in v8_sota_5 is spawn-synchronized prototype initialization plus three prototype refinement passes per segment, improving prototype fitting without increasing LoRA training epochs.",
    )
    add_formula(
        doc,
        "Eq. 7",
        "b_i* = argmin_b NLL(y_i | x_i, W + Delta W_b)",
        "The pseudo-label uses no task ID; it only asks which LoRA branch gives the lowest teacher-forced loss for the example.",
    )
    add_formula(
        doc,
        "Eq. 8",
        "p_b <- normalize(ema * p_b + (1 - ema) * mean(z_i | b_i* = b))",
        "In v8_sota_5, ema=0.8 and prototype_update_steps=3. The vector z_i is the pooled hidden feature from the frozen base model.",
    )
    add_formula(
        doc,
        "Eq. 9",
        "repeat Eq. 7 and Eq. 8 for K_proto = 3 passes after each segment",
        "This is the defining v8_sota_5 implementation change. It repeats the pseudo-label-and-EMA cycle three times on the same segment features, improving prototype convergence while keeping LoRA training budget fixed.",
    )
    add_formula(
        doc,
        "Eq. 10",
        "q_b(x) = softmax(cos(z(x), p_b) / T),  T = 0.08",
        "At inference time, cosine similarity to branch prototypes defines the routing distribution. The configuration enables both hard routing and soft top-3 routing.",
    )

    add_heading(doc, "3.5 Anti-overlap regularization", 2)
    add_para(
        doc,
        "As the LoRA bank grows, different branches may learn overlapping behaviors, which makes routing unstable. The v8_sota_5 run keeps anti-overlap regularization active with beta=0.05 and an activation-to-weight ratio of 0.6:0.4.",
    )
    add_formula(
        doc,
        "Eq. 11",
        "L_total = L_NLL + beta_a * L_activation_overlap + beta_w * L_weight_overlap",
        "The activation overlap term directly penalizes hidden-representation cosine similarity across branches on the same batch, while the weight overlap term penalizes redundant LoRA weight directions.",
    )
    add_formula(
        doc,
        "Eq. 12",
        "L_activation_overlap = mean_{i<j} mean_x 1 / (||h_i(x) - h_j(x)||_2 + epsilon)",
        "The differentiable implementation treats branch activations as repulsive particles. Lower activation distance creates a larger penalty, pushing branches toward functional specialization.",
    )

    add_heading(doc, "3.6 Implementation summary of v8_sota_5", 2)
    add_table(
        doc,
        ["Module", "Configuration or implementation"],
        [
            ["Backbone", "Llama-3.1-8B-Instruct, bf16, max_seq_len=512"],
            ["LoRA", "r=16, alpha=32, dropout=0.05, q_proj/v_proj"],
            ["Training budget", "1 epoch per segment, batch_size=2, lr=2e-4"],
            ["Bank", "max_branches=15, spawn_on_drift=true, freeze_old_branches=true"],
            ["Router", "prototype backend, soft top-3, temperature=0.08, warmup=1"],
            ["v8_sota_5 novelty", "spawn-sync prototype initialization plus 3-pass prototype EMA per segment"],
            ["Overlap", "beta=0.05, activation_beta=0.03, weight_beta=0.02"],
            ["Logging", "save every segment, W&B group v8_sota, full per-segment metrics"],
        ],
        widths=[4.0, 12.0],
    )


def add_experiment_section(doc: Document, runs: Dict[str, Dict[str, Any]], charts: Dict[str, Path]) -> None:
    add_heading(doc, "4  Experimental Setup", 1)
    add_table(
        doc,
        ["Item", "Setting"],
        [
            ["Benchmark", "CITB InstrDialog, 19 segments, seed=123"],
            ["Data per segment", "50 train examples and 10 eval examples after processed stream preparation"],
            ["Protocol", "Task-free online continual instruction tuning"],
            ["Baselines", "Sequential LoRA, Replay(50), PeriodicLatest, BankNoRouter, RouterOnly"],
            ["Baseline budget", "true_fair protocol: 5 epochs, batch_size=1, around 250 batches per segment"],
            ["Ours budget", "v8_sota_5: 1 epoch, batch_size=2, around 25 batches per segment"],
            ["Main comparison", "Final-segment performance and retention against true-fair baselines"],
        ],
        widths=[4.0, 12.0],
    )

    add_heading(doc, "5  Results", 1)
    add_para(
        doc,
        "This section reports two types of evidence: the final comparison against the true-fair baselines implemented under baselines/, and the best-performing internal run from the current method line.",
    )

    add_heading(doc, "5.1 Main comparison against true-fair baselines", 2)
    add_table(
        doc,
        ["Method", "Seen", "Task-aware", "Forgetting", "TA forgetting", "Token F1", "LCS", "Oracle agree"],
        metric_rows(
            runs,
            ["Sequential", "Replay(50)", "PeriodicLatest", "BankNoRouter", "RouterOnly", "Ours v8_sota_5"],
        ),
        widths=[3.2, 1.7, 2.0, 1.8, 2.0, 1.7, 1.7, 2.0],
    )
    add_para(
        doc,
        "v8_sota_5 clearly improves over the baselines on seen, task-aware, token_f1, and lcs. The strongest baseline is PeriodicLatest with task-aware=0.326, "
        "but its forgetting is 0.399, which indicates poor retention under fixed periodic capacity allocation. RouterOnly reaches oracle agreement=0.531, "
        "but its task-aware score is only 0.227, showing that routing alone is not sufficient without drift-triggered capacity allocation.",
    )

    add_heading(doc, "5.2 Best internal run: v8_sota_5", 2)
    best = runs["Ours v8_sota_5"]
    rows = [[label, fmt(best.get(key), 4)] for key, label in MAIN_METRICS]
    rows.extend(
        [
            ["Oracle agreement", fmt(best.get("routing.oracle_agreement_rate"), 4)],
            ["Number of routed examples", fmt(best.get("routing.num_routed"), 0)],
            ["Number of branches", fmt(best.get("num_branches"), 0)],
            ["Overlap mean cosine", fmt(best.get("overlap_mean_cosine"), 4)],
            ["Drift miss rate", fmt(best.get("drift.miss_rate"), 4)],
        ]
    )
    add_table(doc, ["Metric", "v8_sota_5"], rows, widths=[6.0, 4.0])
    add_para(
        doc,
        "v8_sota_5 is the current best internal run by the primary strict seen score. It reaches seen=0.353, task-aware=0.404, forgetting=0.049, TA forgetting=0.017, "
        "token_f1=0.419, and lcs=0.431. Compared with the strongest baseline, PeriodicLatest, it improves task-aware score by 0.078 and reduces forgetting by 0.350.",
    )

    add_heading(doc, "5.3 Internal method trajectory", 2)
    add_table(
        doc,
        ["Run", "Seen", "Task-aware", "Forgetting", "TA forgetting", "Token F1", "LCS", "Oracle agree"],
        metric_rows(runs, ["Ours v4_sota_39", "Ours v6_sota_2", "Ours v8_sota_5", "Ours v8_sota_8"]),
        widths=[3.2, 1.7, 2.0, 1.8, 2.0, 1.7, 1.7, 2.0],
    )
    add_para(
        doc,
        "v4_sota_39 has higher token_f1 and lcs, and v8_sota_8 has slightly lower forgetting. However, v8_sota_5 has the highest primary strict seen score, "
        "so it is the best overall checkpoint for this report. v8_sota_8 increases prototype_update_steps from 3 to 4 and improves some retention metrics, but its final seen score is slightly below v8_sota_5.",
    )
    if charts.get("seen_trajectory"):
        doc.add_picture(str(charts["seen_trajectory"]), width=Inches(6.4))
        add_para(doc, "Seen trajectory across representative method and baseline runs.")


def add_analysis_sections(doc: Document) -> None:
    add_heading(doc, "6  Analysis and Discussion", 1)
    add_heading(doc, "6.1 Why v8_sota_5 is the current champion", 2)
    add_bullet(doc, "Spawn-sync prototype init fixes cold-start: new branches receive a meaningful initial prototype when spawned, avoiding early routing collapse.")
    add_bullet(doc, "Three-pass prototype EMA improves pseudo-label fitting without increasing LoRA training epochs, preserving a low compute budget.")
    add_bullet(doc, "Soft top-3 routing keeps useful mixtures when the best branch is uncertain, while warmup=1 prevents unstable early routing.")
    add_bullet(doc, "Anti-overlap keeps branch specialization pressure active as branch count grows to around 10.")

    add_heading(doc, "6.2 Remaining bottlenecks", 2)
    add_bullet(doc, "Drift detector has miss_rate=0.5 in v8_sota_5, so half of true shifts are not detected promptly.")
    add_bullet(doc, "Overlap remains high: final overlap_mean_cosine is around 0.880, suggesting specialization is incomplete.")
    add_bullet(doc, "Router agreement is moderate: oracle agreement is 0.417 for v8_sota_5, below v4_sota_39 and below RouterOnly, although RouterOnly has much worse task performance.")

    add_heading(doc, "6.3 Lessons from negative runs", 2)
    add_table(
        doc,
        ["Run", "Change", "Outcome", "Interpretation"],
        [
            ["v8_sota_8", "prototype_update_steps=4", "seen=0.352, forgetting improves", "More prototype passes help stability but do not improve primary seen."],
            ["v8_sota_11", "router_warmup_segments=2", "early stop at seg4, seen=0.220", "Longer latest-branch warmup causes catastrophic forgetting."],
            ["v8_sota_13", "router lr=0.006", "seen=0.341", "Conservative router learning slows peak segment recovery."],
            ["v8_sota_14", "router lr=0.010", "early stop at seg7, seen=0.321", "Aggressive router learning increases late trajectory instability."],
        ],
        widths=[2.8, 4.5, 3.8, 5.2],
    )

    add_heading(doc, "7  Conclusion and Next Steps", 1)
    add_para(
        doc,
        "The main conclusion is that the drift-triggered LoRA bank with prototype routing substantially outperforms the sequential, replay, periodic, bank-only, and router-only baselines on the main retention and generation-quality metrics."
    )
    add_bullet(doc, "The first next step is to reduce drift miss rate, because missed shifts directly affect branch allocation and retention of earlier tasks.")
    add_bullet(doc, "The second priority is to reduce branch overlap so that the prototype routing feature space becomes more separable.")
    add_bullet(doc, "The third priority is to improve routing calibration while preserving the strict seen advantage of v8_sota_5.")

    add_heading(doc, "Appendix A  Reproducibility Checklist", 1)
    add_table(
        doc,
        ["Artifact", "Path or value"],
        [
            ["Champion config", "configs/paper/v8_sota_5.yaml"],
            ["Champion metrics", "results/runs/paper_instrdialog_ours_full_s123_v8_sota_5/final_metrics.json"],
            ["Segment metrics", "results/tables/paper_instrdialog_ours_full_s123_v8_sota_5_segment_metrics.csv"],
            ["Baseline metrics", "results/runs/paper_instrdialog_*_s123_true_fair/final_metrics.json"],
            ["W&B project", "lora-citb-sota for method runs; lora-citb-acl for baseline backfill"],
            ["Seed", "123"],
        ],
        widths=[4.0, 12.0],
    )

    add_heading(doc, "Appendix B  References Used in the Report", 1)
    add_bullet(doc, "Hu et al. LoRA: Low-Rank Adaptation of Large Language Models.")
    add_bullet(doc, "Zhang et al. CITB: A Benchmark for Continual Instruction Tuning.")
    add_bullet(doc, "Kirkpatrick et al. Overcoming Catastrophic Forgetting in Neural Networks.")
    add_bullet(doc, "Online-LoRA and task-free continual learning literature for drift-driven adaptation.")
    add_bullet(doc, "Mixture-of-experts and LoRA routing literature for inference-time branch selection.")


def blacken_document(doc: Document) -> None:
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            run.font.color.rgb = BLACK
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.color.rgb = BLACK


def build_doc() -> Path:
    runs = load_all_runs()
    charts = create_charts(runs)
    output = resolve_docx_path()

    doc = Document()
    set_document_style(doc)

    section = doc.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    add_title(
        doc,
        "Task-Free Online LoRA for Continual Instruction Tuning",
        f"Research Experiment Report | InstrDialog seed=123 | Generated on {REPORT_DATE}",
    )

    add_heading(doc, "Abstract", 1)
    add_para(
        doc,
        "Large language models deployed in online instruction environments face non-stationary streams, where instruction formats, domains, and user intents shift over time. "
        "This work studies a task-free continual instruction tuning setting in which task identities are unavailable during both training and inference. "
        "The proposed method maintains a drift-triggered LoRA bank, trains a prototype router through self-consistency pseudo-labels, and regularizes branch overlap. "
        "On InstrDialog with seed=123, the best current run, v8_sota_5, reaches seen=0.353 and task-aware=0.404, substantially outperforming the true-fair baselines implemented in baselines/.",
    )
    add_para(doc, "Keywords: continual instruction tuning; LoRA; task-free learning; drift detection; prototype routing; anti-overlap regularization.")

    add_heading(doc, "1  Introduction", 1)
    add_para(
        doc,
        "Real LLM systems receive non-stationary instruction data: product policies change, user intent distributions drift, and formats evolve. "
        "A naive sequential LoRA adapter can quickly overfit the current segment and forget previous ones. The central research question is how to adapt online "
        "without task IDs, while retaining earlier capabilities under a small memory and compute budget.",
    )
    add_para(
        doc,
        "A reliable solution must jointly solve four problems: detecting distribution shifts, allocating new adaptation capacity, selecting the correct adapter without task IDs, "
        "and preventing the growing adapter bank from collapsing into redundant branches. The method developed here treats these problems as a closed-loop system rather than isolated components.",
    )

    add_heading(doc, "2  Related Work and Research Gap", 1)
    add_bullet(doc, "LoRA and PEFT reduce training cost by updating low-rank adapters while freezing the base model.")
    add_bullet(doc, "Continual instruction tuning benchmarks show that sequential fine-tuning suffers from catastrophic forgetting under non-stationary streams.")
    add_bullet(doc, "Existing LoRA continual learning methods often assume task boundaries or fixed adapter allocation policies.")
    add_bullet(doc, "Routing-based methods can select experts, but task-free routing is unstable when adapter branches overlap or drift allocation is poor.")
    add_para(
        doc,
        "The gap addressed here is not only to add more adapters, but to close the loop among drift detection, capacity allocation, branch specialization, and task-free inference-time routing.",
    )

    add_method_section(doc)
    add_experiment_section(doc, runs, charts)
    add_analysis_sections(doc)

    blacken_document(doc)
    doc.save(output)
    return output


if __name__ == "__main__":
    out = build_doc()
    print(out)
