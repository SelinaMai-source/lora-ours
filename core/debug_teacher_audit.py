"""
Teacher-forced / label-shift debugging utilities (baseline validity only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.causal_lm_metrics import (
    causal_lm_shifted_labels,
    causal_lm_shifted_logits_argmax,
    count_supervised_label_tokens,
    teacher_forced_token_accuracy_shifted,
)
from core.data import Example
from core.formatting import format_for_infer
from core.train_labels import assert_first_supervised_matches_target_start, build_supervised_labels


def _decoded_token_pieces(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    except Exception:
        return ""


def build_per_sample_teacher_audit_row(
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    instruction: str,
    input_text: str,
    target: str,
    max_len: int,
    min_target_tokens: int,
    mask_eos_token_in_labels: bool,
    mask_all_special_tokens_in_labels: bool,
    labeling_mode: str,
    completion_only_response_template: str,
    strict_assertions: bool = False,
) -> Dict[str, Any]:
    import torch

    enc = build_supervised_labels(
        tokenizer,
        instruction,
        input_text,
        str(target),
        max_len=max_len,
        min_target_tokens=min_target_tokens,
        mask_eos_token_in_labels=mask_eos_token_in_labels,
        mask_all_special_tokens_in_labels=mask_all_special_tokens_in_labels,
        labeling_mode=labeling_mode,
        completion_only_response_template=completion_only_response_template,
    )
    full_ids = enc.full_ids
    labels = enc.labels
    input_ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask_t = torch.ones_like(input_ids_t, dtype=torch.long)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_t, attention_mask=attention_mask_t, labels=labels_t, return_dict=True
        )
        logits = outputs.logits
        loss = float(outputs.loss.detach().float().item())

    shift_labels = causal_lm_shifted_labels(labels_t)[0].detach().cpu().tolist()
    shift_pred = causal_lm_shifted_logits_argmax(logits)[0].detach().cpu().tolist()
    supervised_mask_row = [int(x != -100) for x in labels]
    shifted_supervised_mask = [int(x != -100) for x in shift_labels]

    sup_idx = [i for i, v in enumerate(labels) if v != -100]
    first_sup = sup_idx[0] if sup_idx else -1
    last_sup = sup_idx[-1] if sup_idx else -1

    sup_ids = [full_ids[i] for i in sup_idx]
    gold_span = tokenizer.decode(sup_ids, skip_special_tokens=True) if sup_ids else ""
    pred_sup_ids: List[int] = []
    for j, lab in enumerate(shift_labels):
        if lab == -100:
            continue
        pred_sup_ids.append(int(shift_pred[j]))
    pred_span = tokenizer.decode(pred_sup_ids, skip_special_tokens=True) if pred_sup_ids else ""

    acc, n_correct, n_loss = teacher_forced_token_accuracy_shifted(logits, labels_t)
    n_sup_lab = int(count_supervised_label_tokens(labels_t))

    if strict_assertions:
        assert int(n_loss) == sum(1 for x in shift_labels if x != -100), (
            "num_loss_tokens must equal count of shifted label positions != -100 (HF causal LM span)"
        )

    ok_first, first_reason = assert_first_supervised_matches_target_start(
        tokenizer, labels, full_ids, str(target)
    )
    if strict_assertions and sup_idx:
        assert ok_first, f"first supervised token audit failed: {first_reason}"

    decoded_tokens = [_decoded_token_pieces(tokenizer, tid) for tid in full_ids]

    return {
        "instruction": instruction,
        "input": input_text,
        "target": str(target),
        "labeling_mode": labeling_mode,
        "manual_prompt_len": enc.manual_prompt_len,
        "completion_supervise_start": enc.completion_supervise_start,
        "raw_input_ids": full_ids,
        "decoded_token_pieces": decoded_tokens,
        "labels": labels,
        "supervised_mask": supervised_mask_row,
        "shifted_labels_for_loss": shift_labels,
        "shifted_supervised_mask": shifted_supervised_mask,
        "shifted_argmax_predictions": shift_pred,
        "first_supervised_token_index": int(first_sup),
        "last_supervised_token_index": int(last_sup),
        "decoded_supervised_gold_span": gold_span,
        "decoded_supervised_predicted_span": pred_span,
        "teacher_forced_loss": loss,
        "teacher_forced_answer_token_acc_shifted": float(acc),
        "num_correct_shifted": int(n_correct),
        "num_loss_tokens": int(n_loss),
        "num_supervised_label_tokens": int(n_sup_lab),
        "first_supervised_token_audit_ok": bool(ok_first),
        "first_supervised_token_audit_reason": first_reason,
    }


def dump_overfit_teacher_audit(
    *,
    backbone: Any,
    examples: List[Example],
    out_path: str,
    strict_assertions: bool = False,
) -> None:
    tok = getattr(backbone, "tokenizer", None)
    model = getattr(backbone, "model", None)
    cfg = getattr(backbone, "cfg", None)
    if tok is None or model is None or cfg is None:
        raise RuntimeError("backbone must expose tokenizer, model, cfg")

    is_seq2seq = hasattr(cfg, "max_source_len") and hasattr(cfg, "max_target_len") and not hasattr(cfg, "max_seq_len")
    was_training = bool(model.training)
    model.eval()
    try:
        rows: List[Dict[str, Any]] = []
        for i, ex in enumerate(examples):
            if is_seq2seq:
                row = build_per_sample_seq2seq_teacher_audit_row(
                    tokenizer=tok,
                    model=model,
                    device=backbone.device,
                    instruction=ex.instruction,
                    input_text=ex.input,
                    target=str(ex.output),
                    max_source_len=int(cfg.max_source_len),
                    max_target_len=int(cfg.max_target_len),
                )
            else:
                row = build_per_sample_teacher_audit_row(
                    tokenizer=tok,
                    model=model,
                    device=backbone.device,
                    instruction=ex.instruction,
                    input_text=ex.input,
                    target=str(ex.output),
                    max_len=int(cfg.max_seq_len),
                    min_target_tokens=int(cfg.min_target_tokens_for_loss),
                    mask_eos_token_in_labels=bool(cfg.mask_eos_token_in_labels),
                    mask_all_special_tokens_in_labels=bool(cfg.mask_all_special_tokens_in_labels),
                    labeling_mode=str(cfg.train_labeling_mode),
                    completion_only_response_template=str(cfg.completion_only_response_template),
                    strict_assertions=strict_assertions,
                )
            row["sample_index"] = i
            rows.append(row)
    finally:
        if was_training:
            model.train()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def build_per_sample_seq2seq_teacher_audit_row(
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    instruction: str,
    input_text: str,
    target: str,
    max_source_len: int,
    max_target_len: int,
) -> Dict[str, Any]:
    import torch

    source = format_for_infer(tokenizer, instruction, input_text, add_generation_prompt=False)
    enc = tokenizer(
        source,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=int(max_source_len),
    )
    target_enc = tokenizer(
        text_target=str(target),
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=int(max_target_len),
    )
    input_ids_t = enc["input_ids"].to(device)
    attention_mask_t = enc["attention_mask"].to(device)
    labels_t = target_enc["input_ids"].to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels_t = labels_t.masked_fill(labels_t == int(pad_id), -100)
    with torch.no_grad():
        outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t, labels=labels_t, return_dict=True)
        logits = outputs.logits
        loss = float(outputs.loss.detach().float().item())

    pred_ids = torch.argmax(logits, dim=-1)
    mask = labels_t.ne(-100)
    n_loss = int(mask.sum().detach().cpu().item())
    n_correct = int((pred_ids[mask] == labels_t[mask]).detach().cpu().sum().item()) if n_loss else 0
    acc = float(n_correct / max(1, n_loss))
    gold_ids = labels_t[0][mask[0]].detach().cpu().tolist()
    pred_sup_ids = pred_ids[0][mask[0]].detach().cpu().tolist()
    return {
        "instruction": instruction,
        "input": input_text,
        "target": str(target),
        "labeling_mode": "seq2seq_text_target",
        "source_token_len": int(input_ids_t.shape[-1]),
        "target_token_len": int(n_loss),
        "decoded_supervised_gold_span": tokenizer.decode(gold_ids, skip_special_tokens=True),
        "decoded_supervised_predicted_span": tokenizer.decode(pred_sup_ids, skip_special_tokens=True),
        "teacher_forced_loss": loss,
        "teacher_forced_answer_token_acc_shifted": acc,
        "teacher_forced_answer_token_acc_seq2seq": acc,
        "num_correct_shifted": int(n_correct),
        "num_loss_tokens": int(n_loss),
        "num_supervised_label_tokens": int(n_loss),
    }


def mean_teacher_forced_over_examples(
    *,
    backbone: Any,
    pairs: List[Tuple[str, str]],
    targets: List[str],
) -> Tuple[float, float, int]:
    """Returns (mean_loss, mean_shifted_token_acc, num_examples)."""
    tok = getattr(backbone, "tokenizer", None)
    model = getattr(backbone, "model", None)
    cfg = getattr(backbone, "cfg", None)
    if tok is None or model is None or cfg is None:
        return 0.0, 0.0, 0

    is_seq2seq = hasattr(cfg, "max_source_len") and hasattr(cfg, "max_target_len") and not hasattr(cfg, "max_seq_len")
    was_training = bool(model.training)
    model.eval()
    try:
        losses: List[float] = []
        accs: List[float] = []
        for (ins, inp), tgt in zip(pairs, targets):
            if is_seq2seq:
                row = build_per_sample_seq2seq_teacher_audit_row(
                    tokenizer=tok,
                    model=model,
                    device=backbone.device,
                    instruction=ins,
                    input_text=inp,
                    target=str(tgt),
                    max_source_len=int(cfg.max_source_len),
                    max_target_len=int(cfg.max_target_len),
                )
            else:
                row = build_per_sample_teacher_audit_row(
                    tokenizer=tok,
                    model=model,
                    device=backbone.device,
                    instruction=ins,
                    input_text=inp,
                    target=str(tgt),
                    max_len=int(cfg.max_seq_len),
                    min_target_tokens=int(cfg.min_target_tokens_for_loss),
                    mask_eos_token_in_labels=bool(cfg.mask_eos_token_in_labels),
                    mask_all_special_tokens_in_labels=bool(cfg.mask_all_special_tokens_in_labels),
                    labeling_mode=str(cfg.train_labeling_mode),
                    completion_only_response_template=str(cfg.completion_only_response_template),
                    strict_assertions=False,
                )
            losses.append(float(row["teacher_forced_loss"]))
            accs.append(float(row["teacher_forced_answer_token_acc_shifted"]))
    finally:
        if was_training:
            model.train()

    if not losses:
        return 0.0, 0.0, 0
    return float(sum(losses) / len(losses)), float(sum(accs) / len(accs)), len(losses)
