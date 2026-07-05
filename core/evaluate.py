from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.causal_lm_metrics import count_supervised_label_tokens, teacher_forced_token_accuracy_shifted
from core.data import Example, Segment
from core.formatting import format_for_infer, format_for_train
from core.train_labels import build_supervised_labels
from core.metrics_utils import (
    arper_woz3_corpus_bleu4 as _arper_woz3_corpus_bleu4,
    corpus_bleu4 as _corpus_bleu4,
    dialogue_slot_error_rate as _dialogue_slot_error_rate,
    lcs_length as _lcs_length,
    lcs_overlap as _lcs_overlap,
    rouge_l_fscore as _rouge_l_fscore,
    sentence_bleu4 as _sentence_bleu4,
    starts_incorrectly as _starts_incorrectly,
    token_f1 as _token_f1,
)
from core.normalize_answer import basic_answer_normalize as _basic_answer_normalize
from core.normalize_answer import normalize_for_eval as _normalize


@dataclass
class EvalResult:
    current_score: float
    seen_avg_score: float
    current_task_aware_score: float
    seen_avg_task_aware_score: float
    forgetting: float
    task_aware_forgetting: float
    num_seen_segments: int
    token_f1_mean: float
    lcs_overlap_mean: float
    rouge_l_mean: float
    bleu_mean: float
    slot_error_rate: float
    extra: Dict[str, Any]


def _extract_router_feature_for_prompt(model: Any, router: Any, lora_bank: Any, prompt: str) -> Any:
    import torch

    feature_adapter = (
        router.feature_adapter_name
        if hasattr(lora_bank, "has_adapter") and lora_bank.has_adapter(router.feature_adapter_name)
        else None
    )
    active_before = lora_bank.get_active_branch() if hasattr(lora_bank, "get_active_branch") else None
    if feature_adapter is not None and hasattr(lora_bank, "set_active_adapter"):
        lora_bank.set_active_adapter(feature_adapter)
    try:
        if hasattr(model, "get_activations_tensor"):
            return model.get_activations_tensor([prompt], with_grad=False)
        return torch.tensor(model.get_activations([prompt]), dtype=torch.float32)
    finally:
        if active_before is not None and hasattr(lora_bank, "set_active_adapter") and lora_bank.has_adapter(active_before):
            lora_bank.set_active_adapter(active_before)


def _normalized_routing_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, float(v)) for v in scores.values()))
    if total <= 0:
        n = max(1, len(scores))
        return {k: 1.0 / n for k in scores}
    return {k: float(max(0.0, float(v)) / total) for k, v in scores.items()}


def _routing_entropy(prob_scores: Dict[str, float]) -> float:
    import math

    entropy = 0.0
    for p in prob_scores.values():
        if p > 0:
            entropy -= float(p) * math.log(float(p) + 1e-12)
    return float(entropy)


def _example_references(ex: Example) -> List[str]:
    refs = [str(x) for x in getattr(ex, "output_references", ()) if str(x) != ""]
    if not refs:
        refs = [str(ex.output or "")]
    return refs


def _best_reference_by_score(
    *,
    pred: str,
    refs: Sequence[str],
    metric_name: str,
    normalize: bool = True,
) -> Tuple[str, float]:
    if not refs:
        refs = [""]
    pred_for_score = _basic_answer_normalize(pred) if normalize else str(pred or "")
    best_ref = str(refs[0])
    best_score = -1.0
    for ref in refs:
        ref_for_score = _basic_answer_normalize(ref) if normalize else str(ref or "")
        score = _continuous_task_score(metric_name=metric_name, prediction=pred_for_score, gold=ref_for_score)
        if score > best_score:
            best_score = float(score)
            best_ref = str(ref)
    return best_ref, float(max(0.0, best_score))


def _orthogonal_gate_blend_weights(
    adapters: List[str],
    weights: List[float],
    lora_bank: Any,
    lam: float,
) -> List[float]:
    """Down-weight blend coefficients when candidate LoRA branches overlap in weight space.

    w_i' ∝ w_i · exp(-λ Σ_{j≠i} |cos(v_i, v_j)|), then renormalized.
    Eval-time only; uses detached adapter vectors (no label leakage).
    """
    import math

    import torch
    import torch.nn.functional as F

    if lam <= 0.0 or len(adapters) <= 1:
        return weights

    wrapper = getattr(lora_bank, "_lora_wrapper", None)
    if wrapper is None or not hasattr(wrapper, "get_adapter_vector"):
        return weights

    vecs: Dict[str, torch.Tensor] = {}
    for name in adapters:
        try:
            vec = wrapper.get_adapter_vector(name, detach=True)
        except TypeError:
            vec = wrapper.get_adapter_vector(name)
        if vec.numel() > 0:
            vecs[name] = F.normalize(vec.float(), dim=0)

    if len(vecs) < 2:
        return weights

    gated: List[float] = []
    for i, name in enumerate(adapters):
        w = float(weights[i])
        if name not in vecs:
            gated.append(w)
            continue
        sim_sum = 0.0
        for j, other in enumerate(adapters):
            if i == j or other not in vecs:
                continue
            sim_sum += abs(
                float(F.cosine_similarity(vecs[name].unsqueeze(0), vecs[other].unsqueeze(0)).item())
            )
        gated.append(w * math.exp(-lam * sim_sum))

    total = float(sum(gated))
    if total <= 0.0:
        return weights
    return [g / total for g in gated]


def _oracle_branch_for_example(model: Any, lora_bank: Any, ex: Example, branch_names: List[str]) -> Dict[str, Any]:
    pairs = [(ex.instruction, ex.input)]
    refs = _example_references(ex)
    active_before = lora_bank.get_active_branch() if hasattr(lora_bank, "get_active_branch") else None
    losses: Dict[str, float] = {}
    try:
        for branch_name in branch_names:
            if hasattr(lora_bank, "set_active_adapter"):
                lora_bank.set_active_adapter(branch_name)
            ref_pairs = pairs * max(1, len(refs))
            ref_losses = model.score_answer_nlls(ref_pairs, refs)
            losses[branch_name] = float(min(ref_losses)) if ref_losses else float("inf")
    finally:
        if active_before is not None and hasattr(lora_bank, "set_active_adapter") and lora_bank.has_adapter(active_before):
            lora_bank.set_active_adapter(active_before)

    ranked = sorted((loss, name) for name, loss in losses.items())
    best_loss, best_branch = ranked[0]
    second_loss = ranked[1][0] if len(ranked) > 1 else best_loss
    return {
        "oracle_branch": best_branch,
        "oracle_best_loss": float(best_loss),
        "oracle_margin": float(second_loss - best_loss),
        "oracle_losses": losses,
    }


def _support_nll_task_branch(
    *,
    model: Any,
    lora_bank: Any,
    segment: Segment,
    branch_names: List[str],
    normalization_cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    cfg = normalization_cfg.get("eval_task_support_nll_fallback", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return "", {}
    if not hasattr(model, "score_answer_nlls") or not hasattr(lora_bank, "set_active_adapter"):
        return "", {}

    support = [ex for ex in segment.train if str(ex.output or "").strip()]
    max_examples = max(1, int(cfg.get("max_support_examples", 64)))
    support = support[:max_examples]
    if not support or not branch_names:
        return "", {}

    pairs = [(ex.instruction, ex.input) for ex in support]
    targets = [ex.output for ex in support]
    active_before = lora_bank.get_active_branch() if hasattr(lora_bank, "get_active_branch") else None
    losses: Dict[str, float] = {}
    try:
        for branch_name in branch_names:
            lora_bank.set_active_adapter(branch_name)
            branch_losses = model.score_answer_nlls(pairs, targets)
            losses[branch_name] = float(sum(branch_losses) / max(1, len(branch_losses))) if branch_losses else float("inf")
    finally:
        if active_before is not None and hasattr(lora_bank, "has_adapter") and lora_bank.has_adapter(active_before):
            lora_bank.set_active_adapter(active_before)

    ranked = sorted((loss, branch) for branch, loss in losses.items())
    if not ranked:
        return "", {}
    best_loss, best_branch = ranked[0]
    second_loss = ranked[1][0] if len(ranked) > 1 else best_loss
    margin = float(second_loss - best_loss)
    detail = {
        "support_nll_fallback_considered": True,
        "support_nll_best_branch": best_branch,
        "support_nll_best_loss": float(best_loss),
        "support_nll_margin": margin,
        "support_nll_num_examples": int(len(support)),
    }
    if margin < float(cfg.get("min_loss_margin", 0.0)):
        detail["support_nll_fallback_rejected"] = "low_margin"
        return "", detail
    return best_branch, detail


def evaluate_stream(
    *,
    model: Any,
    segments_seen: List[Segment],
    max_new_tokens: int = 64,
    router: Optional[Any] = None,
    lora_bank: Optional[Any] = None,
    segment_id: int,
    normalization_cfg: Optional[Dict[str, Any]] = None,
    save_debug_examples_dir: Optional[str] = None,
    historical_best_per_segment: Optional[Dict[int, float]] = None,
    historical_best_task_aware_per_segment: Optional[Dict[int, float]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Unified evaluation for continual instruction tuning.

    Returns a metrics dict that is stable across modes:
      - current_score: accuracy on current segment eval
      - seen_avg_score: average accuracy across all seen segments
      - forgetting: max(prev_best - current) over seen segments (simplified)
      - num_seen_segments

    Extension points:
      - drift metrics
      - routing metrics (branch usage entropy, decision confidence)
      - overlap metrics (activation similarity)
    """

    # Track per-segment accuracy over time
    per_seg_acc: List[Tuple[int, float]] = []
    per_seg_task_aware_acc: List[Tuple[int, float]] = []
    routing_stats = {
        "num_routed": 0,
        "branch_counts": {},
        "oracle_best_branch_counts": {},
        "oracle_agreement_count": 0,
        "oracle_num_examples": 0,
        "confidence_sum": 0.0,
        "entropy_sum": 0.0,
        "oracle_margin_sum": 0.0,
    }

    all_examples_for_dump: List[Dict[str, Any]] = []
    all_token_f1: List[float] = []
    all_lcs_overlap: List[float] = []
    all_rouge_l: List[float] = []
    all_bleu: List[float] = []
    all_slot_error: List[float] = []
    all_task_aware_scores: List[float] = []
    task_score_type_counts: Dict[str, int] = {}
    all_prefix1: List[int] = []
    all_prefix3: List[int] = []
    all_prefix5: List[int] = []
    num_bad_prefix = 0
    for seg in segments_seen:
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "eval_segment_start",
                    "outer_segment_id": int(segment_id),
                    "eval_segment_id": int(seg.segment_id),
                    "eval_segment_name": str(seg.segment_name),
                    "num_eval_examples": int(len(seg.eval)),
                }
            )
        acc, task_aware_acc, seg_routing, seg_examples = _eval_segment(
            model=model,
            segment=seg,
            max_new_tokens=max_new_tokens,
            router=router,
            lora_bank=lora_bank,
            segment_id=segment_id,
            normalization_cfg=normalization_cfg or {},
        )
        per_seg_acc.append((seg.segment_id, acc))
        per_seg_task_aware_acc.append((seg.segment_id, task_aware_acc))
        _merge_routing_stats(routing_stats, seg_routing)
        all_examples_for_dump.extend(seg_examples)
        all_token_f1.extend([float(x.get("token_f1", 0.0)) for x in seg_examples])
        all_lcs_overlap.extend([float(x.get("lcs_overlap", 0.0)) for x in seg_examples])
        all_rouge_l.extend([float(x.get("rouge_l", 0.0)) for x in seg_examples])
        all_bleu.extend([float(x.get("bleu", 0.0)) for x in seg_examples])
        all_slot_error.extend(
            [float(x["slot_error_rate"]) for x in seg_examples if x.get("slot_error_rate") is not None]
        )
        all_task_aware_scores.extend([float(x.get("task_aware_score", 0.0)) for x in seg_examples])
        for x in seg_examples:
            score_type = str(x.get("task_score_type", "unknown"))
            task_score_type_counts[score_type] = task_score_type_counts.get(score_type, 0) + 1
        num_bad_prefix += sum(1 for x in seg_examples if bool(x.get("bad_prefix_mismatch", False)))
        all_prefix1.extend([int(bool(x.get("prefix_1_match", False))) for x in seg_examples])
        all_prefix3.extend([int(bool(x.get("prefix_3_match", False))) for x in seg_examples])
        all_prefix5.extend([int(bool(x.get("prefix_5_match", False))) for x in seg_examples])
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "eval_segment_end",
                    "outer_segment_id": int(segment_id),
                    "eval_segment_id": int(seg.segment_id),
                    "eval_segment_name": str(seg.segment_name),
                    "num_eval_examples": int(len(seg.eval)),
                    "accuracy": float(acc),
                    "task_aware_accuracy": float(task_aware_acc),
                }
            )

    # current segment is the last in segments_seen
    current_score = per_seg_acc[-1][1] if per_seg_acc else 0.0
    seen_avg_score = sum(a for _, a in per_seg_acc) / max(1, len(per_seg_acc))
    current_task_aware_score = per_seg_task_aware_acc[-1][1] if per_seg_task_aware_acc else 0.0
    seen_avg_task_aware_score = sum(a for _, a in per_seg_task_aware_acc) / max(1, len(per_seg_task_aware_acc))
    anytime_score = float(seen_avg_score)
    anytime_task_aware_score = float(seen_avg_task_aware_score)

    if historical_best_per_segment is None:
        historical_best_per_segment = {}
    previous_segment_ids = {sid for sid, _ in per_seg_acc[:-1]}
    forgetting_by_segment: List[Dict[str, float]] = []
    forgetting_values: List[float] = []
    for sid, acc in per_seg_acc:
        best_before = float(historical_best_per_segment.get(sid, acc))
        seg_forgetting = float(max(0.0, best_before - acc)) if sid in previous_segment_ids else 0.0
        if sid in previous_segment_ids:
            forgetting_values.append(seg_forgetting)
        forgetting_by_segment.append(
            {
                "segment_id": int(sid),
                "accuracy": float(acc),
                "best_historical_accuracy": float(best_before),
                "forgetting": float(seg_forgetting),
            }
        )
        historical_best_per_segment[sid] = max(best_before, float(acc))
    forgetting = float(sum(forgetting_values) / max(1, len(forgetting_values))) if forgetting_values else 0.0

    if historical_best_task_aware_per_segment is None:
        historical_best_task_aware_per_segment = {}
    previous_task_segment_ids = {sid for sid, _ in per_seg_task_aware_acc[:-1]}
    task_aware_forgetting_by_segment: List[Dict[str, float]] = []
    task_aware_forgetting_values: List[float] = []
    for sid, acc in per_seg_task_aware_acc:
        best_before = float(historical_best_task_aware_per_segment.get(sid, acc))
        seg_forgetting = float(max(0.0, best_before - acc)) if sid in previous_task_segment_ids else 0.0
        if sid in previous_task_segment_ids:
            task_aware_forgetting_values.append(seg_forgetting)
        task_aware_forgetting_by_segment.append(
            {
                "segment_id": int(sid),
                "task_aware_accuracy": float(acc),
                "best_historical_task_aware_accuracy": float(best_before),
                "task_aware_forgetting": float(seg_forgetting),
            }
        )
        historical_best_task_aware_per_segment[sid] = max(best_before, float(acc))
    task_aware_forgetting = (
        float(sum(task_aware_forgetting_values) / max(1, len(task_aware_forgetting_values)))
        if task_aware_forgetting_values
        else 0.0
    )

    routing_num = int(routing_stats.get("num_routed", 0))
    branch_counts = dict(routing_stats.get("branch_counts", {}))
    branch_utilization = {
        branch: float(count) / max(1, routing_num)
        for branch, count in sorted(branch_counts.items())
    }
    oracle_num = int(routing_stats.get("oracle_num_examples", 0))
    corpus_predictions = [str(x.get("raw_generated_output", "")) for x in all_examples_for_dump]
    corpus_references = [
        [str(ref) for ref in x.get("gold_references", []) if str(ref).strip()]
        or [str(x.get("gold_output", ""))]
        for x in all_examples_for_dump
    ]

    extra = {
        "per_segment_accuracy": [{"segment_id": sid, "accuracy": acc} for sid, acc in per_seg_acc],
        "per_segment_task_aware_accuracy": [
            {"segment_id": sid, "task_aware_accuracy": acc} for sid, acc in per_seg_task_aware_acc
        ],
        "anytime_score": anytime_score,
        "anytime_task_aware_score": anytime_task_aware_score,
        "forgetting_by_segment": forgetting_by_segment,
        "task_aware_forgetting_by_segment": task_aware_forgetting_by_segment,
        "routing": {
            **routing_stats,
            "branch_utilization": branch_utilization,
            "decision_confidence_mean": float(routing_stats.get("confidence_sum", 0.0) / max(1, routing_num)),
            "decision_entropy_mean": float(routing_stats.get("entropy_sum", 0.0) / max(1, routing_num)),
            "oracle_agreement_rate": float(routing_stats.get("oracle_agreement_count", 0) / max(1, oracle_num)),
            "oracle_margin_mean": float(routing_stats.get("oracle_margin_sum", 0.0) / max(1, oracle_num)),
        },
        "token_f1_mean": float(sum(all_token_f1) / max(1, len(all_token_f1))),
        "lcs_overlap_mean": float(sum(all_lcs_overlap) / max(1, len(all_lcs_overlap))),
        "rouge_l_mean": float(sum(all_rouge_l) / max(1, len(all_rouge_l))),
        "bleu_mean": float(sum(all_bleu) / max(1, len(all_bleu))),
        "corpus_bleu4": float(_corpus_bleu4(corpus_predictions, corpus_references)),
        "arper_woz3_corpus_bleu4": float(_arper_woz3_corpus_bleu4(corpus_predictions, corpus_references)),
        "slot_error_rate": float(sum(all_slot_error) / max(1, len(all_slot_error))),
        "slot_error_count": int(len(all_slot_error)),
        "task_aware_score_mean": float(sum(all_task_aware_scores) / max(1, len(all_task_aware_scores))),
        "task_score_type_counts": task_score_type_counts,
        "prefix_1_match_mean": float(sum(all_prefix1) / max(1, len(all_prefix1))),
        "prefix_3_match_mean": float(sum(all_prefix3) / max(1, len(all_prefix3))),
        "prefix_5_match_mean": float(sum(all_prefix5) / max(1, len(all_prefix5))),
        "num_bad_prefix_mismatch": int(num_bad_prefix),
    }
    extra["likely_assistant_start_or_continuation_boundary"] = bool(
        extra["prefix_1_match_mean"] < 0.5 if (len(all_prefix1) > 0) else False
    )
    if save_debug_examples_dir:
        tok = getattr(model, "tokenizer", None)
        model_cfg = getattr(model, "cfg", None)
        debug_examples: List[Dict[str, Any]] = []
        per_source_debug_counts: Dict[int, int] = {}
        per_source_limit = int((normalization_cfg or {}).get("debug_examples_per_segment", 20))
        total_debug_limit = max(5, int((normalization_cfg or {}).get("debug_examples_max_total", 80)))
        for example in all_examples_for_dump:
            sid = int(example.get("source_segment_id", -1))
            if per_source_debug_counts.get(sid, 0) >= per_source_limit:
                continue
            debug_examples.append(example)
            per_source_debug_counts[sid] = per_source_debug_counts.get(sid, 0) + 1
            if len(debug_examples) >= total_debug_limit:
                break
        _save_debug_examples(
            save_dir=save_debug_examples_dir,
            segment_id=segment_id,
            examples=debug_examples,
            normalization_cfg=normalization_cfg or {},
            generation_cfg={
                "requested_max_new_tokens": max_new_tokens,
                "effective_max_new_tokens": _resolve_eval_max_new_tokens(
                    max_new_tokens=max_new_tokens,
                    eval_examples=[ex for seg in segments_seen for ex in seg.eval],
                    normalization_cfg=normalization_cfg or {},
                ),
                "do_sample": False,
                "no_repeat_ngram_size": getattr(model_cfg, "gen_no_repeat_ngram_size", None),
                "encoder_no_repeat_ngram_size": getattr(model_cfg, "gen_encoder_no_repeat_ngram_size", None),
                "repetition_penalty": getattr(model_cfg, "gen_repetition_penalty", None),
                "eos_token_id": getattr(tok, "eos_token_id", None),
                "pad_token_id": getattr(tok, "pad_token_id", None),
            },
        )
    return asdict(
        EvalResult(
            current_score=float(current_score),
            seen_avg_score=float(seen_avg_score),
            current_task_aware_score=float(current_task_aware_score),
            seen_avg_task_aware_score=float(seen_avg_task_aware_score),
            forgetting=float(forgetting),
            task_aware_forgetting=float(task_aware_forgetting),
            num_seen_segments=int(len(segments_seen)),
            token_f1_mean=float(sum(all_token_f1) / max(1, len(all_token_f1))),
            lcs_overlap_mean=float(sum(all_lcs_overlap) / max(1, len(all_lcs_overlap))),
            rouge_l_mean=float(sum(all_rouge_l) / max(1, len(all_rouge_l))),
            bleu_mean=float(sum(all_bleu) / max(1, len(all_bleu))),
            slot_error_rate=float(sum(all_slot_error) / max(1, len(all_slot_error))),
            extra=extra,
        )
    )


def _eval_segment(
    *,
    model: Any,
    segment: Segment,
    max_new_tokens: int,
    router: Optional[Any],
    lora_bank: Optional[Any],
    segment_id: int,
    normalization_cfg: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any], List[Dict[str, Any]]]:
    correct = 0
    task_aware_correct = 0
    total = 0

    routing_stats = {
        "num_routed": 0,
        "branch_counts": {},
        "oracle_best_branch_counts": {},
        "oracle_agreement_count": 0,
        "oracle_num_examples": 0,
        "confidence_sum": 0.0,
        "entropy_sum": 0.0,
        "oracle_margin_sum": 0.0,
    }

    tok = getattr(model, "tokenizer", None)
    if tok is None:
        raise RuntimeError("Model has no tokenizer; cannot format chat prompts for evaluation.")
    eval_examples = list(segment.eval)
    enable_infer_token_audit = bool(normalization_cfg.get("enable_infer_token_audit", False))
    enable_teacher_forced_eval = bool(normalization_cfg.get("enable_teacher_forced_eval", False))
    infer_token_audit_max_examples = int(normalization_cfg.get("infer_token_audit_max_examples", 8))
    teacher_forced_eval_max_examples = int(normalization_cfg.get("teacher_forced_eval_max_examples", 8))

    base_prompts = [format_for_infer(tok, ex.instruction, ex.input, add_generation_prompt=True) for ex in eval_examples]
    prompts, conditioning_details = _condition_prompts_with_target_prototypes(
        segment=segment,
        eval_examples=eval_examples,
        prompts=base_prompts,
        normalization_cfg=normalization_cfg,
    )
    targets = [ex.output for ex in eval_examples]
    target_refs = [_example_references(ex) for ex in eval_examples]
    effective_max_new_tokens = _resolve_eval_max_new_tokens(
        max_new_tokens=max_new_tokens,
        eval_examples=eval_examples,
        normalization_cfg=normalization_cfg,
    )
    effective_min_new_tokens = _resolve_eval_min_new_tokens(
        segment_name=segment.segment_name,
        max_new_tokens=effective_max_new_tokens,
        normalization_cfg=normalization_cfg,
    )
    generation_kwargs = _resolve_eval_generation_kwargs(
        segment=segment,
        max_new_tokens=effective_max_new_tokens,
        base_min_new_tokens=effective_min_new_tokens,
        normalization_cfg=normalization_cfg,
    )
    support_signature_templates = _build_da_signature_support_templates(
        segment=segment,
        normalization_cfg=normalization_cfg,
    )

    # If router/bank available, route per prompt (simplified hard routing).
    audited = enable_infer_token_audit and router is None and lora_bank is None and hasattr(model, "generate_with_ids")

    if router is not None and lora_bank is not None:
        branch_names = lora_bank.list_branches()
        branch_meta = lora_bank.state_dict()
        support_branch, support_detail = _support_nll_task_branch(
            model=model,
            lora_bank=lora_bank,
            segment=segment,
            branch_names=branch_names,
            normalization_cfg=normalization_cfg,
        )
        support_cfg = normalization_cfg.get("eval_task_support_nll_fallback", {})
        force_support_branch = bool(
            support_branch
            and (support_cfg.get("force", True) if isinstance(support_cfg, dict) else True)
        )
        preds: List[str] = []
        routing_details: List[Dict[str, Any]] = []
        for ex, route_prompt, generation_prompt, conditioning_detail in zip(
            eval_examples,
            base_prompts,
            prompts,
            conditioning_details,
        ):
            features = _extract_router_feature_for_prompt(model, router, lora_bank, route_prompt)
            decision = router.predict_branch(
                prompt=route_prompt,
                branch_names=branch_names,
                branch_meta=branch_meta,
                segment_id=segment_id,
                features=features,
                task_segment_id=segment.segment_id,
            )
            prob_scores = _normalized_routing_scores(decision.scores)
            use_nll_arbitration = bool(getattr(router, "nll_arbitration", False))
            forced_task_fallback = "task_aware_fallback_forced" in str(getattr(decision, "reason", ""))
            ranked = sorted(prob_scores.items(), key=lambda kv: kv[1], reverse=True) if prob_scores else []
            margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
            arbitrated_low_margin = False
            cand_nlls: Dict[str, float] = {}
            original_branch = ""
            # Verify-then-route: label-free prompt-NLL arbitration on low-margin
            # decisions. Works with soft routing too — uncertain cases hard-route
            # to the NLL winner instead of destructive parameter blending.
            if (
                use_nll_arbitration
                and not forced_task_fallback
                and hasattr(model, "score_prompt_nlls")
                and hasattr(lora_bank, "set_active_adapter")
                and len(prob_scores) > 1
                and margin < float(getattr(router, "arbitration_margin", 0.15))
            ):
                original_branch = decision.branch_name
                top_k = max(2, int(getattr(router, "arbitration_top_k", 3)))
                candidates = [b for b, _ in ranked[:top_k]]
                for cand in candidates:
                    lora_bank.set_active_adapter(cand)
                    cand_nlls[cand] = float(model.score_prompt_nlls([route_prompt])[0])
                arbitrated = min(cand_nlls, key=cand_nlls.get)
                routing_stats["nll_arbitration_count"] = routing_stats.get("nll_arbitration_count", 0) + 1
                routing_stats["nll_arbitration_changed"] = routing_stats.get("nll_arbitration_changed", 0)
                if arbitrated != original_branch:
                    routing_stats["nll_arbitration_changed"] = routing_stats.get("nll_arbitration_changed", 0) + 1
                decision.branch_name = arbitrated
                decision.reason = f"{decision.reason}+nll_arbitration"
                arbitrated_low_margin = True
                prob_scores = {b: (1.0 if b == arbitrated else 0.0) for b in prob_scores}
            support_overrode = False
            support_original_branch = ""
            if force_support_branch and support_branch in branch_names and decision.branch_name != support_branch:
                support_original_branch = decision.branch_name
                decision.branch_name = support_branch
                decision.reason = f"{decision.reason}+support_nll_task_fallback"
                support_overrode = True
                prob_scores = {b: (1.0 if b == support_branch else 0.0) for b in prob_scores}
                routing_stats["support_nll_fallback_count"] = routing_stats.get("support_nll_fallback_count", 0) + 1
            if "task_aware_fallback" in str(getattr(decision, "reason", "")):
                routing_stats["task_aware_fallback_count"] = routing_stats.get("task_aware_fallback_count", 0) + 1
            routing_stats["num_routed"] += 1
            routing_stats["branch_counts"][decision.branch_name] = (
                routing_stats["branch_counts"].get(decision.branch_name, 0) + 1
            )
            routing_stats["confidence_sum"] += float(max(prob_scores.values()) if prob_scores else 0.0)
            routing_stats["entropy_sum"] += float(_routing_entropy(prob_scores))
            oracle = _oracle_branch_for_example(model, lora_bank, ex, branch_names)
            routing_stats["oracle_num_examples"] += 1
            routing_stats["oracle_agreement_count"] += int(decision.branch_name == oracle["oracle_branch"])
            routing_stats["oracle_margin_sum"] += float(oracle["oracle_margin"])
            routing_stats["oracle_best_branch_counts"][oracle["oracle_branch"]] = (
                routing_stats["oracle_best_branch_counts"].get(oracle["oracle_branch"], 0) + 1
            )
            routing_details.append(
                {
                    "routing_selected_branch": decision.branch_name,
                    "routing_confidence": float(max(prob_scores.values()) if prob_scores else 0.0),
                    "routing_entropy": float(_routing_entropy(prob_scores)),
                    "routing_oracle_branch": oracle["oracle_branch"],
                    "routing_oracle_margin": float(oracle["oracle_margin"]),
                    "routing_oracle_best_loss": float(oracle["oracle_best_loss"]),
                    "routing_reason": str(getattr(decision, "reason", "")),
                    "routing_nll_arbitration_candidates": cand_nlls if arbitrated_low_margin else {},
                    "routing_nll_arbitration_original_branch": original_branch if arbitrated_low_margin else "",
                    "routing_nll_arbitration_changed": bool(arbitrated_low_margin and decision.branch_name != original_branch),
                    "routing_support_nll_fallback_branch": support_branch,
                    "routing_support_nll_fallback_changed": bool(support_overrode),
                    "routing_support_nll_fallback_original_branch": support_original_branch,
                    **support_detail,
                    **conditioning_detail,
                }
            )
            margin_gate_hard = bool(
                getattr(router, "margin_gated_soft_routing", False)
                and margin >= float(getattr(router, "margin_gate_threshold", 0.12))
            )
            if margin_gate_hard:
                routing_stats["margin_gate_hard_count"] = routing_stats.get("margin_gate_hard_count", 0) + 1
            # Switch adapter before generating (needed for real multi-adapter evaluation).
            if (
                getattr(router, "soft_routing", False)
                and not arbitrated_low_margin
                and not margin_gate_hard
                and hasattr(lora_bank, "set_soft_routing")
            ):
                # Normalize probabilities if requested
                import torch
                raw_scores = list(decision.scores.values())
                adapters = list(decision.scores.keys())
                temp = float(getattr(router, "soft_routing_temperature", 1.0))
                if temp != 1.0 and len(raw_scores) > 0:
                    logits = torch.tensor(raw_scores, dtype=torch.float32) / max(temp, 1e-5)
                    probs = torch.softmax(logits, dim=0)
                    # Filter top-k
                    top_k = int(getattr(router, "soft_routing_top_k", 3))
                    if top_k > 0 and top_k < len(adapters):
                        top_vals, top_idx = torch.topk(probs, top_k)
                        probs = torch.zeros_like(probs)
                        probs[top_idx] = top_vals
                        probs = probs / probs.sum()
                    weights = probs.tolist()
                else:
                    weights = raw_scores

                if bool(getattr(router, "orthogonal_blend", False)):
                    weights = _orthogonal_gate_blend_weights(
                        adapters,
                        weights,
                        lora_bank,
                        float(getattr(router, "orthogonal_blend_lambda", 0.5)),
                    )
                    routing_stats["orthogonal_blend_count"] = routing_stats.get("orthogonal_blend_count", 0) + 1

                # Free unused branches when memory is tight
                import gc
                torch.cuda.empty_cache()
                gc.collect()

                lora_bank.set_soft_routing(adapters, weights)
            elif (
                getattr(router, "soft_routing", False)
                and not arbitrated_low_margin
                and not margin_gate_hard
                and hasattr(lora_bank, "blend_adapters")
            ):
                adapters = list(prob_scores.keys())
                weights = list(prob_scores.values())
                if bool(getattr(router, "orthogonal_blend", False)):
                    weights = _orthogonal_gate_blend_weights(
                        adapters,
                        weights,
                        lora_bank,
                        float(getattr(router, "orthogonal_blend_lambda", 0.5)),
                    )
                    routing_stats["orthogonal_blend_count"] = routing_stats.get("orthogonal_blend_count", 0) + 1
                lora_bank.blend_adapters(adapters, weights, "blended")
                lora_bank.set_active_adapter("blended")
            elif hasattr(lora_bank, "set_active_adapter"):
                lora_bank.set_active_adapter(decision.branch_name)
            
            generated = _generate_with_optional_min_new_tokens(
                model,
                [generation_prompt],
                max_new_tokens=effective_max_new_tokens,
                min_new_tokens=effective_min_new_tokens,
                generation_kwargs=generation_kwargs,
            )[0]
            generated, collapse_retry_detail = _maybe_retry_bucket_collapse_generation(
                model,
                prompt=generation_prompt,
                prediction=generated,
                segment_name=segment.segment_name,
                max_new_tokens=effective_max_new_tokens,
                normalization_cfg=normalization_cfg,
            )
            generated, echo_repair_detail = _maybe_repair_dialogue_act_echo(
                input_text=ex.input,
                prediction=generated,
                segment_name=segment.segment_name,
                normalization_cfg=normalization_cfg,
            )
            generated, support_template_detail = _maybe_apply_da_signature_support_template(
                input_text=ex.input,
                prediction=generated,
                segment_name=segment.segment_name,
                support_templates=support_signature_templates,
                normalization_cfg=normalization_cfg,
            )
            generated, semantic_template_detail = _maybe_apply_dialogue_act_semantic_template(
                input_text=ex.input,
                prediction=generated,
                segment_name=segment.segment_name,
                normalization_cfg=normalization_cfg,
            )
            if collapse_retry_detail:
                routing_details[-1].update(collapse_retry_detail)
                routing_stats["bucket_collapse_retry_count"] = routing_stats.get("bucket_collapse_retry_count", 0) + 1
                if collapse_retry_detail.get("bucket_collapse_retry_accepted"):
                    routing_stats["bucket_collapse_retry_accepted_count"] = (
                        routing_stats.get("bucket_collapse_retry_accepted_count", 0) + 1
                    )
            if echo_repair_detail:
                routing_details[-1].update(echo_repair_detail)
                routing_stats["dialogue_act_echo_repair_count"] = routing_stats.get("dialogue_act_echo_repair_count", 0) + 1
            if support_template_detail:
                routing_details[-1].update(support_template_detail)
                routing_stats["da_signature_support_template_count"] = (
                    routing_stats.get("da_signature_support_template_count", 0) + 1
                )
            if semantic_template_detail:
                routing_details[-1].update(semantic_template_detail)
                routing_stats["dialogue_act_semantic_template_count"] = (
                    routing_stats.get("dialogue_act_semantic_template_count", 0) + 1
                )
            preds.append(generated)
            
            # Clear soft routing
            if getattr(router, "soft_routing", False) and hasattr(lora_bank, "set_soft_routing"):
                lora_bank.set_soft_routing(None, None)
    else:
        if audited:
            gen_audit = model.generate_with_ids(prompts, max_new_tokens=effective_max_new_tokens)
            preds = [x.get("raw_generated_text", "") for x in gen_audit]
        else:
            gen_audit = None
            preds = _generate_with_optional_min_new_tokens(
                model,
                prompts,
                max_new_tokens=effective_max_new_tokens,
                min_new_tokens=effective_min_new_tokens,
                generation_kwargs=generation_kwargs,
            )
        routing_details = list(conditioning_details)
        repaired_preds: List[str] = []
        for ex, pred, detail in zip(eval_examples, preds, routing_details):
            repaired, echo_repair_detail = _maybe_repair_dialogue_act_echo(
                input_text=ex.input,
                prediction=pred,
                segment_name=segment.segment_name,
                normalization_cfg=normalization_cfg,
            )
            if echo_repair_detail:
                detail.update(echo_repair_detail)
            repaired, support_template_detail = _maybe_apply_da_signature_support_template(
                input_text=ex.input,
                prediction=repaired,
                segment_name=segment.segment_name,
                support_templates=support_signature_templates,
                normalization_cfg=normalization_cfg,
            )
            repaired, semantic_template_detail = _maybe_apply_dialogue_act_semantic_template(
                input_text=ex.input,
                prediction=repaired,
                segment_name=segment.segment_name,
                normalization_cfg=normalization_cfg,
            )
            if support_template_detail:
                detail.update(support_template_detail)
                routing_stats["da_signature_support_template_count"] = (
                    routing_stats.get("da_signature_support_template_count", 0) + 1
                )
            if semantic_template_detail:
                detail.update(semantic_template_detail)
                routing_stats["dialogue_act_semantic_template_count"] = (
                    routing_stats.get("dialogue_act_semantic_template_count", 0) + 1
                )
            repaired_preds.append(repaired)
        preds = repaired_preds

    details: List[Dict[str, Any]] = []
    for example_idx, (ex, p, pred, y, refs, routing_detail) in enumerate(
        zip(eval_examples, prompts, preds, targets, target_refs, routing_details)
    ):
        total += 1
        norm_pred = _normalize(pred, prompt=p, cfg=normalization_cfg)
        norm_refs = [_normalize(ref, prompt=p, cfg=normalization_cfg) for ref in refs]
        norm_gold = norm_refs[0] if norm_refs else _normalize(y, prompt=p, cfg=normalization_cfg)
        official_norm_pred = _official_exact_normalize(pred)
        official_norm_refs = [_official_exact_normalize(ref) for ref in refs]
        matched = any(official_norm_pred == ref for ref in official_norm_refs)
        token_f1 = max((_token_f1(norm_pred, ref) for ref in norm_refs), default=0.0)
        lcs_overlap = max((_lcs_overlap(norm_pred, ref) for ref in norm_refs), default=0.0)
        best_rouge_ref, rouge_l = _best_reference_by_score(
            pred=pred,
            refs=refs,
            metric_name="rouge_l",
            normalize=False,
        )
        bleu = max((_sentence_bleu4(norm_pred, ref) for ref in norm_refs), default=0.0)
        slot_error = _dialogue_slot_error_rate(ex.input, norm_pred)
        task_score = _score_task_aware(
            pred=pred,
            gold=y,
            gold_references=refs,
            norm_pred=norm_pred,
            norm_gold=norm_gold,
            instruction=ex.instruction,
            input_text=ex.input,
            cfg=normalization_cfg,
        )
        bad_prefix = _starts_incorrectly(norm_pred, norm_gold)

        pred_tokens = [t for t in norm_pred.split() if t]
        gold_tokens = [t for t in norm_gold.split() if t]
        prefix_1_match = pred_tokens[:1] == gold_tokens[:1]
        prefix_3_match = pred_tokens[:3] == gold_tokens[:3]
        prefix_5_match = pred_tokens[:5] == gold_tokens[:5]

        teacher_forced_loss: Optional[float] = None
        teacher_forced_answer_token_acc: Optional[float] = None
        teacher_forced_num_loss_tokens: Optional[int] = None
        teacher_forced_num_supervised_label_tokens: Optional[int] = None
        if enable_teacher_forced_eval and total <= teacher_forced_eval_max_examples:
            # Teacher-forced forward pass over (prompt + gold assistant target).
            # Token accuracy must use the same shifted span as HF causal LM loss
            # (argmax(logits[:, :-1]) vs labels[:, 1:], ignoring -100); same-index
            # argmax vs labels was incorrect and inflated mismatch diagnostics.
            import torch

            was_training = getattr(model, "model", None).training if getattr(model, "model", None) is not None else False
            # Keep it in eval for determinism.
            if getattr(model, "model", None) is not None:
                model.model.eval()

            with torch.no_grad():
                backbone_cfg = getattr(model, "cfg", None)
                max_len = int(getattr(backbone_cfg, "max_seq_len", 2048))
                min_tgt = int(getattr(backbone_cfg, "min_target_tokens_for_loss", 1))
                labeling_mode = str(
                    normalization_cfg.get("teacher_forced_labeling_mode")
                    or (getattr(backbone_cfg, "train_labeling_mode", None) if backbone_cfg is not None else None)
                    or "manual"
                )
                completion_template = str(
                    normalization_cfg.get("completion_only_response_template")
                    or (
                        getattr(backbone_cfg, "completion_only_response_template", None)
                        if backbone_cfg is not None
                        else None
                    )
                    or "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                mask_eos = bool(normalization_cfg.get("mask_eos_token_in_labels", True))
                mask_spec = bool(normalization_cfg.get("mask_all_special_tokens_in_labels", True))

                enc = build_supervised_labels(
                    tok,
                    ex.instruction,
                    ex.input,
                    str(y),
                    max_len=max_len,
                    min_target_tokens=min_tgt,
                    mask_eos_token_in_labels=mask_eos,
                    mask_all_special_tokens_in_labels=mask_spec,
                    labeling_mode=labeling_mode,
                    completion_only_response_template=completion_template,
                )
                full_ids = enc.full_ids
                labels = enc.labels

                input_ids_t = torch.tensor([full_ids], dtype=torch.long, device=model.device)
                attention_mask_t = torch.ones_like(input_ids_t, dtype=torch.long)
                labels_t = torch.tensor([labels], dtype=torch.long, device=model.device)

                outputs = model.model(
                    input_ids=input_ids_t, attention_mask=attention_mask_t, labels=labels_t, return_dict=True
                )
                teacher_forced_loss = float(outputs.loss.detach().float().item())

                logits = outputs.logits  # [1, T, V]
                teacher_forced_answer_token_acc, _, n_loss = teacher_forced_token_accuracy_shifted(logits, labels_t)
                teacher_forced_num_loss_tokens = int(n_loss)
                teacher_forced_num_supervised_label_tokens = int(count_supervised_label_tokens(labels_t))

            if getattr(model, "model", None) is not None and was_training:
                model.model.train()

        if matched:
            correct += 1
        if bool(task_score["task_aware_match"]):
            task_aware_correct += 1
        details.append(
            {
                "instruction": ex.instruction,
                "input_text": ex.input,
                "gold_output": y,
                "gold_references": list(refs),
                "best_rouge_reference": best_rouge_ref,
                "source_segment_id": int(segment.segment_id),
                "source_segment_name": str(segment.segment_name),
                "source_example_idx": int(example_idx),
                "requested_max_new_tokens": int(max_new_tokens),
                "effective_max_new_tokens": int(effective_max_new_tokens),
                "effective_min_new_tokens": int(effective_min_new_tokens),
                "generation_kwargs": dict(generation_kwargs),
                "raw_generated_output": pred,
                "normalized_prediction": norm_pred,
                "normalized_gold": norm_gold,
                "match": matched,
                "strict_match": matched,
                **task_score,
                "token_f1": float(token_f1),
                "lcs_overlap": float(lcs_overlap),
                "rouge_l": float(rouge_l),
                "bleu": float(bleu),
                "slot_error_rate": None if slot_error is None else float(slot_error),
                "bad_prefix_mismatch": bool(bad_prefix),
                "prefix_1_match": bool(prefix_1_match),
                "prefix_3_match": bool(prefix_3_match),
                "prefix_5_match": bool(prefix_5_match),
                "formatted_infer_prompt": p,
                "teacher_forced_loss": teacher_forced_loss,
                "teacher_forced_answer_token_acc": teacher_forced_answer_token_acc,
                "teacher_forced_num_loss_tokens": teacher_forced_num_loss_tokens,
                "teacher_forced_num_supervised_label_tokens": teacher_forced_num_supervised_label_tokens,
                **routing_detail,
            }
        )

        # Optional continuation boundary / slicing audit.
        if audited and len(details) <= infer_token_audit_max_examples:
            idx = len(details) - 1
            audit = gen_audit[idx]
            prompt_len = int(audit.get("infer_prompt_token_len", 0))
            gen_full_ids = audit.get("generated_full_ids", [])
            gen_cont_ids = audit.get("generated_continuation_ids", [])
            assert gen_cont_ids == gen_full_ids[prompt_len:], "Eval slice assertion failed"

            # Verify decoded prediction comes from continuation ids only.
            tok_redecoded = tok.decode(gen_cont_ids, skip_special_tokens=True) if gen_cont_ids else ""
            if tok_redecoded != (audit.get("raw_generated_text", "") or ""):
                # Don't hard-fail; just record for debugging.
                details[-1]["continuation_decoding_mismatch"] = True

            details[-1].update(
                {
                    "formatted_infer_prompt_text": p,
                    "infer_prompt_token_ids": audit.get("infer_prompt_token_ids", []),
                    "infer_prompt_token_len": prompt_len,
                    "generated_full_ids": gen_full_ids,
                    "generated_continuation_ids": gen_cont_ids,
                    "decoded_prompt_tail": audit.get("decoded_prompt_tail", ""),
                    "decoded_continuation_head": audit.get("decoded_continuation_head", ""),
                    "raw_generated_text": audit.get("raw_generated_text", ""),
                }
            )

    acc = correct / max(1, total)
    task_aware_acc = task_aware_correct / max(1, total)
    return float(acc), float(task_aware_acc), routing_stats, details


def _merge_routing_stats(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst["num_routed"] += int(src.get("num_routed", 0))
    for k, v in src.get("branch_counts", {}).items():
        dst["branch_counts"][k] = dst["branch_counts"].get(k, 0) + int(v)
    for k, v in src.get("oracle_best_branch_counts", {}).items():
        dst["oracle_best_branch_counts"][k] = dst["oracle_best_branch_counts"].get(k, 0) + int(v)
    dst["oracle_agreement_count"] += int(src.get("oracle_agreement_count", 0))
    dst["oracle_num_examples"] += int(src.get("oracle_num_examples", 0))
    dst["confidence_sum"] += float(src.get("confidence_sum", 0.0))
    dst["entropy_sum"] += float(src.get("entropy_sum", 0.0))
    dst["oracle_margin_sum"] += float(src.get("oracle_margin_sum", 0.0))
    for key, value in src.items():
        if key.endswith("_count") and key not in dst:
            dst[key] = 0
        if key.endswith("_count") and key not in {
            "num_routed",
            "oracle_agreement_count",
            "oracle_num_examples",
        }:
            dst[key] = int(dst.get(key, 0)) + int(value or 0)


def _condition_prompts_with_target_prototypes(
    *,
    segment: Segment,
    eval_examples: List[Example],
    prompts: List[str],
    normalization_cfg: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    cfg = normalization_cfg.get("target_prototype_conditioning", {})
    if not isinstance(cfg, dict):
        cfg = {}
    if not bool(cfg.get("enabled", False)):
        return list(prompts), [{} for _ in prompts]

    patterns = cfg.get("segment_name_patterns", cfg.get("name_patterns", []))
    if isinstance(patterns, str):
        patterns = [patterns]
    name = str(segment.segment_name or "")
    if patterns and not any(re.search(str(pattern), name, flags=re.IGNORECASE) for pattern in patterns):
        return list(prompts), [{} for _ in prompts]

    candidates = [
        {
            "input_terms": set(_prediction_words(ex.input)),
            "target_terms": set(_prediction_words(ex.output)),
            "target": _truncate_target_prototype(ex.output, int(cfg.get("max_target_chars", 160))),
        }
        for ex in segment.train
        if str(ex.output or "").strip()
    ]
    candidates = [c for c in candidates if c["target"]]
    if not candidates:
        return list(prompts), [{} for _ in prompts]

    top_k = max(1, int(cfg.get("top_k", 2)))
    min_overlap = max(0, int(cfg.get("min_input_overlap", 1)))
    include_target_overlap = bool(cfg.get("include_target_term_overlap", True))
    header = str(
        cfg.get(
            "header",
            "Relevant response prototypes from similar training dialogues:",
        )
    ).strip()

    conditioned_prompts: List[str] = []
    details: List[Dict[str, Any]] = []
    for ex, prompt in zip(eval_examples, prompts):
        query_terms = set(_prediction_words(ex.input))
        ranked: List[Tuple[float, str]] = []
        for cand in candidates:
            input_overlap = len(query_terms & cand["input_terms"])
            target_overlap = len(query_terms & cand["target_terms"]) if include_target_overlap else 0
            score = float(input_overlap) + 0.25 * float(target_overlap)
            if input_overlap >= min_overlap or (min_overlap == 0 and score > 0.0):
                ranked.append((score, str(cand["target"])))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        selected: List[str] = []
        seen = set()
        for _, target in ranked:
            key = target.lower()
            if key in seen:
                continue
            selected.append(target)
            seen.add(key)
            if len(selected) >= top_k:
                break

        if selected:
            block = header + "\n" + "\n".join(f"- {target}" for target in selected)
            conditioned_prompts.append(f"{block}\n\n{prompt}")
        else:
            conditioned_prompts.append(prompt)
        details.append(
            {
                "target_prototype_conditioning_enabled": bool(selected),
                "target_prototype_conditioning_count": int(len(selected)),
                "target_prototype_conditioning_targets": selected,
            }
        )
    return conditioned_prompts, details


def _truncate_target_prototype(text: str, max_chars: int) -> str:
    target = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_chars <= 0 or len(target) <= max_chars:
        return target
    return target[: max(0, max_chars - 1)].rstrip() + "..."


def _resolve_eval_min_new_tokens(
    *,
    segment_name: str,
    max_new_tokens: int,
    normalization_cfg: Dict[str, Any],
) -> int:
    if not bool(normalization_cfg.get("enable_segment_min_new_tokens", False)):
        return 0
    patterns = normalization_cfg.get("segment_min_new_tokens_name_patterns", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    name = str(segment_name or "")
    if not any(re.search(str(pattern), name, flags=re.IGNORECASE) for pattern in patterns):
        return 0
    min_new_tokens = int(normalization_cfg.get("segment_min_new_tokens", 0))
    return max(0, min(int(max_new_tokens), min_new_tokens))


def _resolve_eval_generation_kwargs(
    *,
    segment: Segment,
    max_new_tokens: int,
    base_min_new_tokens: int,
    normalization_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    overrides = normalization_cfg.get("generation_overrides", [])
    if not isinstance(overrides, list):
        return {}
    selected: Dict[str, Any] = {}
    for raw in overrides:
        if not isinstance(raw, dict):
            continue
        patterns = raw.get("segment_name_patterns", raw.get("name_patterns", []))
        if isinstance(patterns, str):
            patterns = [patterns]
        name = str(segment.segment_name or "")
        if patterns and not any(re.search(str(pattern), name, flags=re.IGNORECASE) for pattern in patterns):
            continue
        selected = raw
        break
    if not selected:
        return {}

    kwargs: Dict[str, Any] = {}
    for src_key, dst_key, caster in [
        ("num_beams", "num_beams", int),
        ("min_new_tokens", "min_new_tokens", int),
        ("length_penalty", "length_penalty", float),
        ("no_repeat_ngram_size", "no_repeat_ngram_size", int),
        ("encoder_no_repeat_ngram_size", "encoder_no_repeat_ngram_size", int),
        ("repetition_penalty", "repetition_penalty", float),
    ]:
        if src_key in selected:
            kwargs[dst_key] = caster(selected[src_key])

    if bool(selected.get("use_train_target_length_prior", False)):
        prior_min = _train_target_length_prior_min_new_tokens(
            segment=segment,
            quantile=float(selected.get("train_target_length_quantile", 0.25)),
            min_value=int(selected.get("train_target_min_new_tokens_floor", 0)),
            max_value=int(selected.get("train_target_min_new_tokens_ceiling", max_new_tokens)),
        )
        if prior_min > 0:
            kwargs["min_new_tokens"] = max(int(kwargs.get("min_new_tokens", 0)), int(prior_min))

    if "min_new_tokens" in kwargs:
        kwargs["min_new_tokens"] = max(int(base_min_new_tokens), min(int(max_new_tokens), int(kwargs["min_new_tokens"])))
    return kwargs


def _train_target_length_prior_min_new_tokens(
    *,
    segment: Segment,
    quantile: float,
    min_value: int,
    max_value: int,
) -> int:
    lengths = sorted(len(_prediction_words(ex.output)) for ex in segment.train if _prediction_words(ex.output))
    if not lengths:
        return 0
    q = min(1.0, max(0.0, float(quantile)))
    idx = int(round(q * (len(lengths) - 1)))
    value = int(lengths[idx])
    return max(0, min(int(max_value), max(int(min_value), value)))


def _generate_with_optional_min_new_tokens(
    model: Any,
    prompts: List[str],
    *,
    max_new_tokens: int,
    min_new_tokens: int,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    bad_words_texts: Optional[List[str]] = None,
) -> List[str]:
    kwargs = dict(generation_kwargs or {})
    if "min_new_tokens" not in kwargs and int(min_new_tokens) > 0:
        kwargs["min_new_tokens"] = int(min_new_tokens)
    if bad_words_texts is not None:
        kwargs["bad_words_texts"] = bad_words_texts
    if int(min_new_tokens) <= 0:
        try:
            return model.generate(prompts, max_new_tokens=max_new_tokens, **kwargs)
        except TypeError:
            return model.generate(prompts, max_new_tokens=max_new_tokens)
    try:
        return model.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
    except TypeError:
        return model.generate(prompts, max_new_tokens=max_new_tokens)


def _maybe_retry_bucket_collapse_generation(
    model: Any,
    *,
    prompt: str,
    prediction: str,
    segment_name: str,
    max_new_tokens: int,
    normalization_cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    if not bool(normalization_cfg.get("enable_bucket_collapse_retry", False)):
        return prediction, {}
    patterns = normalization_cfg.get("bucket_collapse_retry_name_patterns", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    name = str(segment_name or "")
    if patterns and not any(re.search(str(pattern), name, flags=re.IGNORECASE) for pattern in patterns):
        return prediction, {}

    tokens = normalization_cfg.get("bucket_collapse_retry_tokens", ["no", "yes", "i"])
    if isinstance(tokens, str):
        tokens = [tokens]
    collapse_tokens = {str(token).strip().lower() for token in tokens if str(token).strip()}
    words = _prediction_words(prediction)
    if len(words) != 1 or words[0] not in collapse_tokens:
        return prediction, {}

    retry_min_new_tokens = int(normalization_cfg.get("bucket_collapse_retry_min_new_tokens", 6))
    retry_min_new_tokens = max(1, min(int(max_new_tokens), retry_min_new_tokens))
    retry_bad_words = _resolve_bucket_collapse_retry_bad_words(normalization_cfg)
    retry_prediction = _generate_with_optional_min_new_tokens(
        model,
        [prompt],
        max_new_tokens=max_new_tokens,
        min_new_tokens=retry_min_new_tokens,
        bad_words_texts=retry_bad_words,
    )[0]
    retry_words = _prediction_words(retry_prediction)
    min_accept_tokens = int(normalization_cfg.get("bucket_collapse_retry_min_accept_tokens", 2))
    template_reason = _prompt_template_continuation_reason(retry_words)
    accepted = (
        len(retry_words) >= max(2, min_accept_tokens)
        and retry_words[0] == words[0]
        and not template_reason
    )
    detail = {
        "bucket_collapse_retry_triggered": True,
        "bucket_collapse_retry_token": words[0],
        "bucket_collapse_retry_min_new_tokens": retry_min_new_tokens,
        "bucket_collapse_retry_bad_words_count": len(retry_bad_words),
        "bucket_collapse_retry_output": retry_prediction,
        "bucket_collapse_retry_accepted": bool(accepted),
        "bucket_collapse_retry_rejection_reason": "" if accepted else _bucket_collapse_retry_rejection_reason(
            retry_words=retry_words,
            original_token=words[0],
            min_accept_tokens=min_accept_tokens,
            template_reason=template_reason,
        ),
    }
    return (retry_prediction if accepted else prediction), detail


def _maybe_repair_dialogue_act_echo(
    *,
    input_text: str,
    prediction: str,
    segment_name: str,
    normalization_cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    cfg = normalization_cfg.get("dialogue_act_echo_repair", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return prediction, {}
    features = _parse_simple_dialogue_act_features(input_text)
    if not features:
        return prediction, {}
    pred_words = _prediction_words(prediction)
    input_words = _prediction_words(input_text)
    pred_lower = str(prediction or "").strip().lower()
    overlap = len(set(pred_words) & set(input_words))
    coverage = float(overlap / max(1, min(len(set(pred_words)), len(set(input_words)))))
    looks_like_input_echo = (
        bool(pred_words)
        and re.search(r"\bslot-[a-z0-9]+-[a-z0-9]+-[a-z0-9]+", pred_lower) is None
        and (
            pred_lower.replace(" ", "") in str(input_text or "").lower().replace(" ", "")
            or coverage >= float(cfg.get("min_input_echo_token_coverage", 0.6))
        )
    )
    if not looks_like_input_echo:
        return prediction, {}
    repaired = _dialogue_act_template_from_features(features, segment_name=segment_name)
    if not repaired:
        return prediction, {}
    return repaired, {
        "dialogue_act_echo_repair_applied": True,
        "dialogue_act_echo_repair_original": str(prediction or ""),
        "dialogue_act_echo_repair_output": repaired,
        "dialogue_act_echo_repair_num_features": int(len(features)),
    }


def _maybe_apply_dialogue_act_semantic_template(
    *,
    input_text: str,
    prediction: str,
    segment_name: str,
    normalization_cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    cfg = normalization_cfg.get("dialogue_act_semantic_template_repair", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return prediction, {}
    patterns = cfg.get("segment_name_patterns", [".*"])
    if isinstance(patterns, str):
        patterns = [patterns]
    if patterns and not any(re.search(str(pattern), str(segment_name or ""), flags=re.IGNORECASE) for pattern in patterns):
        return prediction, {}
    features = _parse_simple_dialogue_act_features(input_text)
    if not features:
        return prediction, {}
    act = str(features[0].get("act", "")).lower()
    allowed_acts = cfg.get("acts", ["inform"])
    if isinstance(allowed_acts, str):
        allowed_acts = [allowed_acts]
    if str(act) not in {str(x).lower() for x in allowed_acts}:
        return prediction, {}
    candidate = _dialogue_act_template_from_features(features, segment_name=segment_name)
    if not candidate or candidate.strip() == str(prediction or "").strip():
        return prediction, {}
    expected_slots = {feat["slot_token"] for feat in features if feat["slot"] != "none"}
    candidate_slots = set(re.findall(r"\bslot-[a-z0-9]+-[a-z0-9]+-[a-z0-9]+", candidate.lower()))
    if expected_slots and not expected_slots.issubset(candidate_slots):
        return prediction, {}
    risky = (
        _looks_like_generic_da_template(prediction)
        or bool(_prompt_template_continuation_reason(_prediction_words(prediction)))
        or _looks_like_slot_listing_output(prediction, features)
    )
    if not risky:
        return prediction, {}
    return candidate, {
        "dialogue_act_semantic_template_applied": True,
        "dialogue_act_semantic_template_original": str(prediction or ""),
        "dialogue_act_semantic_template_output": candidate,
        "dialogue_act_semantic_template_act": act,
        "dialogue_act_semantic_template_reason": (
            "generic_or_slot_listing"
        ),
    }


def _build_da_signature_support_templates(
    *,
    segment: Segment,
    normalization_cfg: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    cfg = normalization_cfg.get("da_signature_support_template_fallback", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return {}
    buckets: Dict[str, List[str]] = {}
    max_support = max(1, int(cfg.get("max_support_examples", 256)))
    for ex in list(segment.train)[:max_support]:
        signature = _dialogue_act_signature(ex.input)
        output = str(ex.output or "").strip()
        if not signature or not output:
            continue
        buckets.setdefault(signature, []).append(output)

    min_examples = max(1, int(cfg.get("min_support_examples_per_signature", 1)))
    out: Dict[str, Dict[str, Any]] = {}
    for signature, outputs in buckets.items():
        unique_outputs = list(dict.fromkeys(outputs))
        if len(unique_outputs) < min_examples:
            continue
        if str(cfg.get("prototype_selection_mode", "")).strip().lower() == "semantic_cluster":
            selected, selected_support_score, selection_detail = _select_semantic_support_output_with_score(
                unique_outputs,
                signature=signature,
            )
        else:
            selected, selected_support_score = _select_central_support_output_with_score(unique_outputs)
            selection_detail = {}
        out[signature] = {
            "output": selected,
            "num_candidates": int(len(unique_outputs)),
            "support_outputs": unique_outputs,
            "support_mean_rouge_l": float(selected_support_score),
            **selection_detail,
        }
    return out


def _maybe_apply_da_signature_support_template(
    *,
    input_text: str,
    prediction: str,
    segment_name: str,
    support_templates: Dict[str, Dict[str, Any]],
    normalization_cfg: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    cfg = normalization_cfg.get("da_signature_support_template_fallback", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return prediction, {}
    if not support_templates:
        return prediction, {}
    patterns = cfg.get("segment_name_patterns", [".*"])
    if isinstance(patterns, str):
        patterns = [patterns]
    if patterns and not any(re.search(str(pattern), str(segment_name or ""), flags=re.IGNORECASE) for pattern in patterns):
        return prediction, {}
    signature = _dialogue_act_signature(input_text)
    if not signature or signature not in support_templates:
        return prediction, {}
    signature_patterns = cfg.get("signature_name_patterns", [])
    if isinstance(signature_patterns, str):
        signature_patterns = [signature_patterns]
    if signature_patterns and not any(re.search(str(pattern), signature) for pattern in signature_patterns):
        return prediction, {}

    replacement = str(support_templates[signature].get("output", "")).strip()
    if not replacement:
        return prediction, {}
    force = bool(cfg.get("force", False))
    support_outputs = [str(x) for x in support_templates[signature].get("support_outputs", []) if str(x).strip()]
    if not support_outputs:
        support_outputs = [replacement]
    replacement_support_score = float(support_templates[signature].get("support_mean_rouge_l", 0.0))
    if replacement_support_score <= 0.0:
        replacement_support_score = _mean_support_score(replacement, support_outputs)
    prediction_support_score = _mean_support_score(prediction, support_outputs)
    support_margin = float(replacement_support_score - prediction_support_score)
    features = _parse_simple_dialogue_act_features(input_text)
    act = str(features[0].get("act", "") if features else "").lower()
    min_support_margin = float(
        _cfg_by_act(
            cfg,
            "min_support_score_margin",
            act=act,
            default=0.05,
        )
    )
    min_support_score = float(cfg.get("min_support_mean_score", 0.0))
    min_candidates = int(cfg.get("min_support_candidates_for_confidence", 1))
    num_candidates = int(support_templates[signature].get("num_candidates", 0))
    confidence_mode = str(cfg.get("support_confidence_mode", "score_or_count")).strip().lower()
    semantic_support_score = float(support_templates[signature].get("semantic_support_score", 0.0))
    min_semantic_score = float(cfg.get("min_semantic_support_score", 0.0))
    if confidence_mode == "score":
        support_confident = replacement_support_score >= min_support_score
    elif confidence_mode == "score_and_count":
        support_confident = (
            replacement_support_score >= min_support_score
            and num_candidates >= max(1, min_candidates)
        )
    elif confidence_mode == "semantic_or_score":
        support_confident = (
            replacement_support_score >= min_support_score
            or semantic_support_score >= min_semantic_score
        )
    else:
        support_confident = (
            num_candidates >= max(1, min_candidates)
            or replacement_support_score >= min_support_score
        )
    pred_slots = set(re.findall(r"\bslot-[a-z0-9]+-[a-z0-9]+-[a-z0-9]+", str(prediction or "").lower()))
    repl_slots = set(re.findall(r"\bslot-[a-z0-9]+-[a-z0-9]+-[a-z0-9]+", replacement.lower()))
    expected_slots = {feat["slot_token"] for feat in features if feat["slot"] != "none"}
    missing_expected_slots = bool(expected_slots - pred_slots)
    generic_template = _looks_like_generic_da_template(prediction)
    prompt_template = bool(_prompt_template_continuation_reason(_prediction_words(prediction)))
    candidate_better_slot_coverage = bool(expected_slots and expected_slots.issubset(repl_slots) and missing_expected_slots)
    model_output_risky = (
        prompt_template
        or generic_template
        or len(_prediction_words(prediction)) < int(cfg.get("min_prediction_words", 4))
        or candidate_better_slot_coverage
    )
    should_replace = force or (
        support_confident
        and model_output_risky
        and support_margin >= min_support_margin
    )
    if not should_replace:
        return prediction, {}
    return replacement, {
        "da_signature_support_template_applied": True,
        "da_signature_support_template_signature": signature,
        "da_signature_support_template_original": str(prediction or ""),
        "da_signature_support_template_output": replacement,
        "da_signature_support_template_act": act,
        "da_signature_support_template_num_candidates": num_candidates,
        "da_signature_support_template_prediction_support_score": float(prediction_support_score),
        "da_signature_support_template_replacement_support_score": float(replacement_support_score),
        "da_signature_support_template_support_margin": float(support_margin),
        "da_signature_support_template_confidence_mode": confidence_mode,
        "da_signature_support_template_semantic_support_score": semantic_support_score,
        "da_signature_support_template_selection_pattern": str(
            support_templates[signature].get("selection_pattern", "")
        ),
        "da_signature_support_template_reason": (
            "force" if force else
            "support_margin_prompt_template" if prompt_template else
            "support_margin_generic_template" if generic_template else
            "support_margin_slot_coverage" if candidate_better_slot_coverage else
            "support_margin_short_prediction"
        ),
    }


def _cfg_by_act(cfg: Dict[str, Any], key: str, *, act: str, default: Any) -> Any:
    by_act = cfg.get(f"{key}_by_act", {})
    if isinstance(by_act, dict) and str(act or "").lower() in by_act:
        return by_act[str(act or "").lower()]
    return cfg.get(key, default)


def _dialogue_act_signature(input_text: str) -> str:
    features = _parse_simple_dialogue_act_features(input_text)
    if not features:
        return ""
    return "|".join(
        sorted(f"{feat['domain']}:{feat['act']}:{feat['slot']}" for feat in features)
    )


def _select_central_support_output(outputs: Sequence[str]) -> str:
    return _select_central_support_output_with_score(outputs)[0]


def _select_central_support_output_with_score(outputs: Sequence[str]) -> Tuple[str, float]:
    if not outputs:
        return "", 0.0
    if len(outputs) == 1:
        return str(outputs[0]), 1.0
    best_output = str(outputs[0])
    best_key = (-1.0, 0, "")
    best_score = 0.0
    for output in outputs:
        scores = [
            _continuous_task_score(
                metric_name="rouge_l",
                prediction=str(output),
                gold=str(other),
            )
            for other in outputs
            if str(other) != str(output)
        ]
        mean_score = float(sum(scores) / max(1, len(scores)))
        key = (mean_score, -abs(len(_prediction_words(output)) - _median_word_count(outputs)), str(output))
        if key > best_key:
            best_key = key
            best_output = str(output)
            best_score = mean_score
    return best_output, float(best_score)


def _select_semantic_support_output_with_score(
    outputs: Sequence[str],
    *,
    signature: str,
) -> Tuple[str, float, Dict[str, Any]]:
    if not outputs:
        return "", 0.0, {}
    if len(outputs) == 1:
        return str(outputs[0]), 1.0, {
            "selection_pattern": _support_language_pattern(str(outputs[0]), signature),
            "semantic_support_score": 1.0,
            "semantic_cluster_size": 1,
        }

    expected_slots = _signature_slot_tokens(signature)
    median_len = max(1, _median_word_count(outputs))
    pattern_counts: Dict[str, int] = {}
    for output in outputs:
        pattern = _support_language_pattern(str(output), signature)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    best_output = str(outputs[0])
    best_support_score = 0.0
    best_pattern = ""
    best_semantic_score = -1.0
    best_key = (-1.0, -1.0, 0, "")
    for output in outputs:
        text = str(output)
        support_score = _mean_support_score(text, [str(other) for other in outputs if str(other) != text])
        pred_slots = set(re.findall(r"\bslot-[a-z0-9]+-[a-z0-9]+-[a-z0-9]+", text.lower()))
        slot_coverage = (
            len(expected_slots & pred_slots) / max(1, len(expected_slots))
            if expected_slots
            else 1.0
        )
        words = _prediction_words(text)
        length_score = max(0.0, 1.0 - (abs(len(words) - median_len) / max(1.0, float(median_len))))
        pattern = _support_language_pattern(text, signature)
        cluster_ratio = pattern_counts.get(pattern, 1) / max(1, len(outputs))
        act_score = _act_semantic_fit_score(text, signature)
        semantic_score = (
            0.35 * float(support_score)
            + 0.25 * float(slot_coverage)
            + 0.20 * float(cluster_ratio)
            + 0.10 * float(length_score)
            + 0.10 * float(act_score)
        )
        key = (
            semantic_score,
            support_score,
            -abs(len(words) - median_len),
            text,
        )
        if key > best_key:
            best_key = key
            best_output = text
            best_support_score = float(support_score)
            best_pattern = pattern
            best_semantic_score = float(semantic_score)
    return best_output, float(best_support_score), {
        "selection_pattern": best_pattern,
        "semantic_support_score": max(0.0, best_semantic_score),
        "semantic_cluster_size": int(pattern_counts.get(best_pattern, 1)),
    }


def _mean_support_score(prediction: str, support_outputs: Sequence[str]) -> float:
    outputs = [str(output) for output in support_outputs if str(output).strip()]
    if not outputs:
        return 0.0
    return float(
        sum(
            _continuous_task_score(
                metric_name="rouge_l",
                prediction=str(prediction or ""),
                gold=output,
            )
            for output in outputs
        )
        / max(1, len(outputs))
    )


def _median_word_count(outputs: Sequence[str]) -> int:
    counts = sorted(len(_prediction_words(output)) for output in outputs)
    if not counts:
        return 0
    return int(counts[len(counts) // 2])


def _signature_slot_tokens(signature: str) -> set:
    tokens = set()
    for part in str(signature or "").split("|"):
        bits = part.split(":")
        if len(bits) != 3:
            continue
        domain, act, slot = [bit.strip().lower() for bit in bits]
        if not domain or not act or not slot or slot == "none":
            continue
        tokens.add(f"slot-{domain}-{act}-{slot}")
    return tokens


def _signature_act(signature: str) -> str:
    part = str(signature or "").split("|")[0]
    bits = part.split(":")
    return bits[1].strip().lower() if len(bits) == 3 else ""


def _support_language_pattern(output: str, signature: str) -> str:
    text = str(output or "").lower()
    act = _signature_act(signature)
    if act in {"nobook", "nooffer"}:
        if "reference number" in text:
            return "nobook_reference"
        if any(term in text for term in ["unable", "not able", "n't able", "can not", "cannot", "unsuccessful"]):
            return "nobook_unable"
        if any(term in text for term in ["not available", "no availability", "no reservations", "no tables", "booked", "full"]):
            return "nobook_unavailable"
        if "sorry" in text or "unfortunately" in text:
            return "nobook_apology"
        return "nobook_other"
    if act == "inform":
        if "?" in text:
            return "inform_question"
        if "pick you up" in text or "taxi" in text:
            return "inform_taxi"
        if any(term in text for term in ["there are", "i have", "we have"]):
            return "inform_offer_count"
        if any(term in text for term in ["address", "postcode", "phone", "number"]):
            return "inform_contact"
        return "inform_statement"
    if act == "book":
        if "reference number" in text:
            return "book_reference"
        return "book_other"
    return "other"


def _act_semantic_fit_score(output: str, signature: str) -> float:
    text = str(output or "").lower()
    act = _signature_act(signature)
    if act in {"nobook", "nooffer"}:
        if "reference number" in text and "ref" not in signature:
            return 0.1
        if any(term in text for term in ["unable", "not available", "no availability", "no reservations", "no tables", "booked", "full", "unsuccessful", "sorry", "unfortunately"]):
            return 1.0
        return 0.4
    if act == "inform":
        if _signature_slot_tokens(signature) and not _signature_slot_tokens(signature).issubset(
            set(re.findall(r"\bslot-[a-z0-9]+-[a-z0-9]+-[a-z0-9]+", text))
        ):
            return 0.2
        return 0.9 if "?" not in text else 0.6
    return 0.8


def _looks_like_generic_da_template(prediction: str) -> bool:
    text = str(prediction or "").strip().lower()
    if not text:
        return True
    generic_phrases = [
        "sorry , there is no matching option available",
        "unfortunately , i can not book it at this time",
        "reference number is :",
        "the addr is",
        "the phone is",
        "the name is",
        "the choice is",
        "the area is",
        "the price is",
        "the time is",
    ]
    return any(phrase in text for phrase in generic_phrases)


def _looks_like_slot_listing_output(prediction: str, features: Sequence[Dict[str, str]]) -> bool:
    text = str(prediction or "").strip().lower()
    if not text:
        return True
    clauses = [clause.strip() for clause in re.split(r"\s*\.\s*", text) if clause.strip()]
    if not clauses:
        return False
    slot_names = {str(feat.get("slot", "")).replace("_", " ") for feat in features}
    listing_clauses = 0
    for clause in clauses:
        if any(clause.startswith(f"the {slot} is ") for slot in slot_names if slot):
            listing_clauses += 1
    return listing_clauses >= max(1, min(len(clauses), len(slot_names)))


def _parse_simple_dialogue_act_features(input_text: str) -> List[Dict[str, str]]:
    features: List[Dict[str, str]] = []
    for raw in str(input_text or "").split("|"):
        parts = [p.strip().lower() for p in raw.split("-") if p.strip()]
        if len(parts) < 3:
            continue
        domain, act, slot = parts[0], parts[1], parts[2]
        features.append(
            {
                "domain": domain,
                "act": act,
                "slot": slot,
                "slot_token": f"slot-{domain}-{act}-{slot}",
            }
        )
    return features


def _dialogue_act_template_from_features(features: List[Dict[str, str]], *, segment_name: str) -> str:
    by_slot = {feat["slot"]: feat["slot_token"] for feat in features}
    act = str(features[0].get("act", "") if features else "").lower()
    if act == "book":
        if "name" in by_slot and "ref" in by_slot:
            return f"you are booked into {by_slot['name']} . the reference number is {by_slot['ref']} ."
        if "ref" in by_slot:
            return f"reference number is : {by_slot['ref']} ."
        if "name" in by_slot:
            return f"you are booked into {by_slot['name']} ."
    if act in {"nobook", "nooffer"}:
        if "ref" in by_slot and len(by_slot) == 1:
            return f"your reference number is {by_slot['ref']} ."
        if "day" in by_slot and "people" in by_slot and "time" in by_slot:
            return (
                f"i 'm sorry , there are no available table on {by_slot['day']} "
                f"for {by_slot['people']} at {by_slot['time']} ."
            )
        if "day" in by_slot and "stay" in by_slot:
            return f"i 'm sorry but there is no availability for {by_slot['stay']} nights starting on {by_slot['day']} ."
        if "name" in by_slot and "time" in by_slot:
            return f"unfortunately , there are no reservations available for {by_slot['name']} at {by_slot['time']} ."
        if "name" in by_slot:
            return f"i 'm sorry , that time is not available at {by_slot['name']} ."
        if "time" in by_slot:
            return f"i 'm sorry , that time is not available at {by_slot['time']} ."
        if "day" in by_slot:
            return f"i 'm sorry , there is no availability on {by_slot['day']} ."
        return "unfortunately , i can not book it at this time ."
    if act == "inform":
        domain = str(features[0].get("domain", "") if features else "").lower()
        if "addr" in by_slot and "phone" in by_slot:
            return f"sure the address is {by_slot['addr']} and the phone number is {by_slot['phone']} ."
        if "addr" in by_slot and "name" in by_slot and "post" in by_slot:
            return f"the address for {by_slot['name']} is {by_slot['addr']} . the post code is {by_slot['post']} ."
        if "addr" in by_slot and "area" in by_slot and "name" in by_slot:
            return f"{by_slot['name']} is located at {by_slot['addr']} in the {by_slot['area']} ."
        if domain == "train" and "choice" in by_slot and "leave" in by_slot:
            return f"there are {by_slot['choice']} trains leaving near {by_slot['leave']} ."
        if domain == "train" and "arrive" in by_slot and "depart" in by_slot and "dest" in by_slot and "id" in by_slot:
            return (
                f"train {by_slot['id']} travels from {by_slot['depart']} to {by_slot['dest']} "
                f"and arrives at {by_slot['arrive']} ."
            )
        if domain == "train" and "arrive" in by_slot and "id" in by_slot and "leave" in by_slot:
            return f"train {by_slot['id']} leaves at {by_slot['leave']} and arrives at {by_slot['arrive']} ."
        if domain == "hotel" and "internet" in by_slot and "price" in by_slot and "stars" in by_slot and "type" in by_slot:
            return (
                f"it is a {by_slot['price']} {by_slot['stars']} star {by_slot['type']} "
                f"with {by_slot['internet']} internet ."
            )
        if "ticket" in by_slot:
            return f"tickets will be {by_slot['ticket']} ."
        if "arrive" in by_slot and len(by_slot) == 1:
            return f"it will arrive at {by_slot['arrive']} ."
        if "choice" in by_slot and len(by_slot) == 1:
            return f"there are {by_slot['choice']} options ."
        if "name" in by_slot and len(by_slot) == 1:
            return f"how about {by_slot['name']} ?"
    if act in {"welcome", "greet"}:
        return "hello , how can i help you ?"
    if act == "bye":
        return "good bye ."
    if act == "reqmore":
        return "is there anything else i can help you with ?"
    if act == "request":
        requested = " and ".join(feat["slot"].replace("_", " ") for feat in features)
        return f"what {requested} would you like ?" if requested else "what would you like ?"
    clauses = []
    for feat in features:
        slot_name = feat["slot"].replace("_", " ")
        clauses.append(f"the {slot_name} is {feat['slot_token']}")
    if clauses:
        return " . ".join(clauses) + " ."
    return ""


def _prediction_words(text: str) -> List[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").strip().lower())
    return [w for w in normalized.split() if w]


def _looks_like_prompt_template_continuation(words: List[str]) -> bool:
    return bool(_prompt_template_continuation_reason(words))


def _prompt_template_continuation_reason(words: List[str]) -> str:
    if not words:
        return ""
    template_singletons = {"definition", "input", "output"}
    for idx, word in enumerate(words):
        if word in template_singletons:
            return f"template_token:{word}@{idx}"
    template_phrases = [
        ("now", "complete"),
        ("following", "example"),
        ("positive", "example"),
        ("negative", "example"),
        ("concatenated", "string"),
        ("newline", "character"),
        ("valid", "prediction"),
        ("initial", "question"),
        ("clarifying", "question"),
    ]
    for idx in range(0, max(0, len(words) - 1)):
        pair = (words[idx], words[idx + 1])
        if pair in template_phrases:
            return f"template_phrase:{pair[0]}_{pair[1]}@{idx}"
    return ""


def _bucket_collapse_retry_rejection_reason(
    *,
    retry_words: List[str],
    original_token: str,
    min_accept_tokens: int,
    template_reason: str,
) -> str:
    if len(retry_words) < max(2, int(min_accept_tokens)):
        return "too_short"
    if not retry_words or retry_words[0] != original_token:
        return "bucket_token_changed"
    if template_reason:
        return template_reason
    return "unknown"


def _resolve_bucket_collapse_retry_bad_words(normalization_cfg: Dict[str, Any]) -> List[str]:
    if not bool(normalization_cfg.get("bucket_collapse_retry_block_prompt_template_tokens", True)):
        return []
    configured = normalization_cfg.get("bucket_collapse_retry_bad_words", None)
    if configured is None:
        configured = [
            "Now complete",
            "now complete",
            "following example",
            "Positive Example",
            "positive example",
            "Negative Example",
            "negative example",
            "Input",
            "input",
            "Output",
            "output",
            "Definition",
            "definition",
            "concatenated string",
            "separated by a newline",
            "valid prediction",
            "initial question",
            "clarifying question",
        ]
    if isinstance(configured, str):
        configured = [configured]
    return [str(text) for text in configured if str(text).strip()]


def _resolve_eval_max_new_tokens(
    *,
    max_new_tokens: int,
    eval_examples: List[Example],
    normalization_cfg: Dict[str, Any],
) -> int:
    """Use shorter greedy generations for structured short-answer evals, while keeping raw text."""
    if not bool(normalization_cfg.get("auto_short_answer_max_new_tokens", True)):
        return int(max_new_tokens)
    if not eval_examples:
        return int(max_new_tokens)
    short_limit = int(normalization_cfg.get("short_answer_max_new_tokens", 16))
    step_limit = int(normalization_cfg.get("step_answer_max_new_tokens", 8))
    golds = [str(ex.output or "") for ex in eval_examples]
    if all(_extract_after_step(g) is not None for g in golds):
        return int(min(max_new_tokens, step_limit))
    normalized_golds = [_basic_answer_normalize(g) for g in golds]
    max_gold_tokens = max((len(g.split()) for g in normalized_golds), default=0)
    if max_gold_tokens <= int(normalization_cfg.get("short_answer_gold_token_threshold", 6)):
        return int(min(max_new_tokens, short_limit))
    return int(max_new_tokens)


def _score_task_aware(
    *,
    pred: str,
    gold: str,
    gold_references: Sequence[str],
    norm_pred: str,
    norm_gold: str,
    instruction: str,
    input_text: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(cfg.get("enable_task_aware_score", True)):
        return {
            "task_aware_match": bool(norm_pred == norm_gold),
            "task_aware_score": float(norm_pred == norm_gold),
            "task_score_type": "strict_em",
            "extracted_prediction": "",
            "extracted_gold": "",
            "prediction_for_scoring": norm_pred,
        }

    refs = [str(ref) for ref in gold_references] or [str(gold or "")]
    pred_for_scoring = _truncate_prediction_for_scoring(pred, cfg)
    norm_pred_for_scoring = _basic_answer_normalize(pred_for_scoring)
    metric_name = str(cfg.get("task_score_metric", cfg.get("primary_score_metric", ""))).strip().lower()
    if metric_name in {"", "auto", "auto_official", "official"}:
        metric_name = _infer_official_metric_for_example(
            gold_references=refs,
            instruction=instruction,
            input_text=input_text,
        )
    if metric_name in {"exact", "exact_match", "em", "accuracy"}:
        pred_exact = _official_exact_normalize(pred_for_scoring)
        matched = any(pred_exact == _official_exact_normalize(ref) for ref in refs)
        return {
            "task_aware_match": bool(matched),
            "task_aware_score": float(matched),
            "task_score_type": "exact_match",
            "extracted_prediction": "",
            "extracted_gold": "",
            "prediction_for_scoring": norm_pred_for_scoring,
        }
    if metric_name in {"rouge_l", "rouge-l", "rougel", "bleu", "bleu4", "token_f1", "lcs_overlap"}:
        best_ref, metric_score = _best_reference_by_score(
            metric_name=metric_name,
            pred=pred_for_scoring,
            refs=refs,
            normalize=metric_name not in {"rouge_l", "rouge-l", "rougel"},
        )
        threshold = float(cfg.get("task_score_match_threshold", 0.5))
        return {
            "task_aware_match": bool(metric_score >= threshold),
            "task_aware_score": float(metric_score),
            "task_score_type": metric_name,
            "extracted_prediction": "",
            "extracted_gold": best_ref,
            "prediction_for_scoring": norm_pred_for_scoring,
        }

    pred_step = _extract_after_step(pred_for_scoring)
    gold_step = _extract_after_step(gold)
    if gold_step is not None:
        matched = pred_step == gold_step
        return {
            "task_aware_match": bool(matched),
            "task_aware_score": float(matched),
            "task_score_type": "after_step_extracted_em",
            "extracted_prediction": "" if pred_step is None else str(pred_step),
            "extracted_gold": str(gold_step),
            "prediction_for_scoring": norm_pred_for_scoring,
        }

    pred_label = _extract_label_like_answer(pred_for_scoring, gold, instruction=instruction, input_text=input_text)
    gold_labels = [
        _extract_label_like_answer(ref, ref, instruction=instruction, input_text=input_text) for ref in refs
    ]
    gold_labels = [label for label in gold_labels if label]
    gold_label = gold_labels[0] if gold_labels else None
    if gold_label:
        matched = pred_label in set(gold_labels)
        return {
            "task_aware_match": bool(matched),
            "task_aware_score": float(matched),
            "task_score_type": "label_accuracy",
            "extracted_prediction": pred_label or "",
            "extracted_gold": "|".join(gold_labels),
            "prediction_for_scoring": norm_pred_for_scoring,
        }

    matched = norm_pred == norm_gold
    return {
        "task_aware_match": bool(matched),
        "task_aware_score": float(matched),
        "task_score_type": "strict_em",
        "extracted_prediction": "",
        "extracted_gold": "",
        "prediction_for_scoring": norm_pred_for_scoring or norm_pred,
    }


def _official_exact_normalize(text: str) -> str:
    # Mirrors Tk-Instruct compute_metrics.normalize_answer: lowercase, drop
    # ASCII punctuation, then collapse whitespace. Articles are intentionally kept.
    import string

    out = str(text or "").lower()
    out = "".join(ch for ch in out if ch not in set(string.punctuation))
    return " ".join(out.split())


def _infer_official_metric_for_example(
    *,
    gold_references: Sequence[str],
    instruction: str,
    input_text: str,
) -> str:
    refs = [str(ref) for ref in gold_references if str(ref).strip()]
    context = f"{instruction}\n{input_text}".lower()
    looks_generation = any(
        key in context
        for key in [
            "generate",
            "generation",
            "write",
            "produce",
            "valid prediction",
            "response to",
        ]
    )
    explicit_classification = any(
        key in context
        for key in [
            "classify",
            "classification",
            "label",
            "category",
            "sentiment",
            "choose",
            "selecting the correct option",
            "output '",
        ]
    )
    looks_classification = any(
        key in context
        for key in [
            "classify",
            "classification",
            "label",
            "category",
            "sentiment",
            "intent",
            "choose",
            "selecting the correct option",
            "output '",
        ]
    )
    if looks_generation and not explicit_classification:
        return "rouge_l"
    label_like_refs = [
        ref
        for ref in refs
        if _extract_label_like_answer(ref, ref, instruction=instruction, input_text=input_text)
    ]
    if refs and looks_classification and len(label_like_refs) == len(refs):
        return "exact_match"
    return "rouge_l"


def _continuous_task_score(*, metric_name: str, prediction: str, gold: str) -> float:
    if metric_name in {"rouge_l", "rouge-l", "rougel"}:
        return float(_rouge_l_fscore(prediction, gold))
    if metric_name in {"bleu", "bleu4"}:
        return float(_sentence_bleu4(prediction, gold))
    if metric_name == "token_f1":
        return float(_token_f1(prediction, gold))
    if metric_name == "lcs_overlap":
        return float(_lcs_overlap(prediction, gold))
    return float(prediction == gold)


def _truncate_prediction_for_scoring(text: str, cfg: Dict[str, Any]) -> str:
    out = str(text or "")
    if bool(cfg.get("score_truncate_at_first_blankline", True)):
        out = out.split("\n\n", 1)[0]
    if bool(cfg.get("score_truncate_at_first_newline", True)):
        out = out.split("\n", 1)[0]
        
    # Remove chatty prefixes often generated by instruction-tuned models
    prefixes = [
        "the correct answer is ",
        "the correct answer is: ",
        "the answer is ",
        "the answer is: ",
        "the output is ",
        "the output is: ",
        "the correct label is ",
        "sure, ",
        "sure! ",
        "here is ",
        "my answer is ",
        "the correct option is "
    ]
    out_lower = out.lower().strip()
    for prefix in prefixes:
        idx = out_lower.find(prefix)
        if idx != -1:
            # slice out from the end of the prefix
            out = out[idx + len(prefix):]
            out_lower = out.lower().strip()
            
    if bool(cfg.get("score_truncate_at_first_sentence_end", True)):
        m = re.search(r"(?<!\b[A-Z])[.!?](?:\s|$)", out)
        if m is not None:
            out = out[: m.end()]
    return out.strip()


def _extract_after_step(text: str) -> Optional[int]:
    m = re.search(r"\bafter\s+step\s*(\d+)\b", str(text or ""), flags=re.IGNORECASE)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_label_like_answer(text: str, gold: str, *, instruction: str, input_text: str) -> Optional[str]:
    gold_norm = _basic_answer_normalize(gold)
    if not gold_norm:
        return None

    # Numeric / yes-no / true-false / A-D answers are common short structured labels.
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", gold_norm):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
        return _basic_answer_normalize(m.group(0)) if m else ""
    if gold_norm in {"yes", "no", "true", "false"}:
        m = re.search(r"\b(yes|no|true|false)\b", str(text or ""), flags=re.IGNORECASE)
        return _basic_answer_normalize(m.group(1)) if m else ""
    if re.fullmatch(r"[a-d]", gold_norm):
        m = re.search(r"\b([A-Da-d])\b", str(text or ""))
        return _basic_answer_normalize(m.group(1)) if m else ""

    gold_tokens = gold_norm.split()
    if len(gold_tokens) > 4:
        return None

    first_chunk = _basic_answer_normalize(_truncate_prediction_for_scoring(str(text or ""), {}))
    if first_chunk == gold_norm:
        return gold_norm

    # Only credit contained labels for very short class names to avoid over-crediting free-form answers.
    context = f"{instruction}\n{input_text}".lower()
    looks_classification = any(k in context for k in ["label", "class", "category", "sentiment", "intent", "choose"])
    if looks_classification and re.search(rf"(?<!\w){re.escape(gold_norm)}(?!\w)", first_chunk):
        return gold_norm
    return first_chunk if looks_classification and len(first_chunk.split()) <= 4 else None


def _extract_instruction_from_prompt(prompt: str) -> str:
    # Legacy helper: preserved for compatibility with old debug files.
    marker = "指令："
    if marker in prompt:
        rest = prompt.split(marker, 1)[1]
        return rest.split("\n", 1)[0]
    return prompt


def _save_debug_examples(
    *,
    save_dir: str,
    segment_id: int,
    examples: List[Dict[str, Any]],
    normalization_cfg: Dict[str, Any],
    generation_cfg: Dict[str, Any],
) -> None:
    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = {
        "segment_id": segment_id,
        "normalization_cfg": normalization_cfg,
        "generation_cfg": generation_cfg,
        "examples": examples,
    }
    (d / f"eval_segment_{segment_id:03d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

