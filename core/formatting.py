from __future__ import annotations

import string
from typing import Any, Dict, List, Optional


def _supports_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")


def _resolve_format_style(tokenizer: Any, *, format_style: Optional[str] = None) -> str:
    explicit = str(format_style or getattr(tokenizer, "_ours_format_style", "") or "").strip().lower()
    if explicit:
        return explicit
    model_type = str(getattr(getattr(tokenizer, "config", None), "model_type", "") or "").lower()
    name_or_path = str(getattr(tokenizer, "name_or_path", "") or "").lower()
    if model_type == "t5" or "t5" in name_or_path:
        return "citb_t5"
    if not _supports_chat_template(tokenizer):
        return "plain"
    return "chat"


def format_citb_t5(instruction: str, input_text: str) -> str:
    """Official CITB / SuperNI T5 prompt (ni_collator, add_task_definition=True)."""
    ins = str(instruction or "").strip()
    if ins and ins[-1] not in string.punctuation:
        ins += "."
    definition = f"Definition: {ins}\n\n" if ins else ""

    inp = str(input_text or "").strip()
    task_input = "Now complete the following example -\n"
    if inp:
        if inp[-1] not in string.punctuation:
            inp += "."
        task_input += f"Input: {inp}\n"
    else:
        task_input += "Input:\n"
    task_input += "Output: "
    return definition + task_input


def build_user_content(instruction: str, input_text: str) -> str:
    ins = str(instruction or "").strip()
    inp = str(input_text or "").strip()
    
    constraint = "You must answer as concisely as possible without any explanations or conversational fillers."
    if constraint not in ins:
        ins = f"{ins}\n\n{constraint}"

    if inp:
        return f"{ins}\n\nInput:\n{inp}"
    return ins


def build_chat_messages(instruction: str, input_text: str, target: str | None = None) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "user", "content": build_user_content(instruction, input_text)}]
    if target is not None:
        messages.append({"role": "assistant", "content": str(target)})
    return messages


def format_for_infer(
    tokenizer: Any,
    instruction: str,
    input_text: str,
    *,
    add_generation_prompt: bool = True,
    format_style: Optional[str] = None,
) -> str:
    style = _resolve_format_style(tokenizer, format_style=format_style)
    if style == "citb_t5":
        _ = add_generation_prompt
        return format_citb_t5(instruction, input_text)
    if style == "plain" or not _supports_chat_template(tokenizer):
        _ = add_generation_prompt
        return build_user_content(instruction, input_text)
    messages = build_chat_messages(instruction, input_text, target=None)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
    )


def format_for_train(tokenizer: Any, instruction: str, input_text: str, target: str) -> Dict[str, str]:
    style = _resolve_format_style(tokenizer)
    prompt_text = format_for_infer(tokenizer, instruction, input_text)
    if style == "citb_t5":
        return {"prompt_text": prompt_text, "full_text": prompt_text}
    if style == "plain" or not _supports_chat_template(tokenizer):
        return {"prompt_text": prompt_text, "full_text": f"{prompt_text}\n\nTarget:\n{target}"}
    full_text = tokenizer.apply_chat_template(
        build_chat_messages(instruction, input_text, target=str(target)),
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"prompt_text": prompt_text, "full_text": full_text}
