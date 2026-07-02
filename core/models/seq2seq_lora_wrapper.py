from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.formatting import format_for_infer
from core.models.base_model import BaseBackbone


@dataclass
class HFSeq2SeqLMConfig:
    hf_model_name_or_path: str
    torch_dtype: str = "bfloat16"
    device: str = "auto"
    max_source_len: int = 1024
    max_target_len: int = 256
    gen_max_new_tokens: int = 64
    gen_do_sample: bool = False
    gen_num_beams: int = 1
    gen_no_repeat_ngram_size: int = 0
    gen_encoder_no_repeat_ngram_size: int = 0
    gen_repetition_penalty: float = 1.0
    format_style: str = "citb_t5"
    debug_print_formatted_examples: bool = False
    debug_print_tokenized_examples: bool = False
    debug_max_tokenized_examples: int = 2
    activation_batch_size: int = 4


class HFSeq2SeqLMBackbone(BaseBackbone):
    """T5-style seq2seq backbone with the same hooks as the causal ours runner."""

    peft_task_type = "SEQ_2_SEQ_LM"

    def __init__(self, cfg: HFSeq2SeqLMConfig, *, seed: int = 0):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg
        self._last_lr = 0.0
        self._debug_printed_formatted = False
        self._debug_printed_tokenized = False

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_name_or_path, use_fast=True)
        fmt_style = str(getattr(cfg, "format_style", "") or "citb_t5").strip().lower()
        setattr(self.tokenizer, "_ours_format_style", fmt_style)
        setattr(self.tokenizer, "_ours_max_source_len", int(cfg.max_source_len))
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
        self.device = torch.device(device)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(cfg.torch_dtype).lower(), torch.bfloat16)
        if self.device.type == "cpu" and torch_dtype in {torch.float16, torch.bfloat16}:
            torch_dtype = torch.float32

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.hf_model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.train()
        if hasattr(self.model, "config") and getattr(self.model.config, "use_cache", None):
            self.model.config.use_cache = False

        try:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def attach_peft_model(self, peft_model: Any) -> None:
        self.model = peft_model
        self.model.to(self.device)
        self.model.train()

    def _format_inputs(self, pairs: List[Tuple[str, str]]) -> List[str]:
        return [format_for_infer(self.tokenizer, ins, inp, add_generation_prompt=False) for ins, inp in pairs]

    def _tokenize_batch(self, sources: List[str], targets: Optional[List[str]] = None) -> Dict[str, Any]:
        enc = self.tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.cfg.max_source_len),
        )
        batch = {k: v.to(self.device) for k, v in enc.items()}
        if targets is not None:
            labels = self.tokenizer(
                text_target=[str(x) for x in targets],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.cfg.max_target_len),
            )["input_ids"]
            pad_id = int(self.tokenizer.pad_token_id)
            labels = labels.masked_fill(labels == pad_id, -100).to(self.device)
            batch["labels"] = labels
        return batch

    def fit_batch(self, pairs: List[Tuple[str, str]], targets: List[str], lr: float) -> Dict[str, float]:
        import torch

        self._last_lr = float(lr)
        sources = self._format_inputs(pairs)
        if self.cfg.debug_print_formatted_examples and not self._debug_printed_formatted:
            show_n = min(2, len(sources))
            print("\n[debug] Seq2Seq formatted training examples:")
            for i in range(show_n):
                print(f"[debug] example_{i}.source={repr(sources[i])}")
                print(f"[debug] example_{i}.target={repr(str(targets[i]))}")
            self._debug_printed_formatted = True

        batch = self._tokenize_batch(sources, targets)
        if self.cfg.debug_print_tokenized_examples and not self._debug_printed_tokenized:
            show_n = min(int(self.cfg.debug_max_tokenized_examples), len(sources))
            print("\n[debug] Seq2Seq tokenized examples:")
            for i in range(show_n):
                print(f"[debug] example_{i}.input_ids={batch['input_ids'][i].detach().cpu().tolist()}")
                print(f"[debug] example_{i}.labels={batch['labels'][i].detach().cpu().tolist()}")
            self._debug_printed_tokenized = True

        outputs = self.model(**batch, return_dict=True)
        loss = outputs.loss
        loss.backward()

        with torch.no_grad():
            labels = batch["labels"]
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            mask = labels.ne(-100)
            token_acc = float((pred_ids[mask] == labels[mask]).float().mean().item()) if bool(mask.any().item()) else 0.0
            preds = self.generate(sources, max_new_tokens=int(self.cfg.gen_max_new_tokens))
            exact = sum(1 for pred, gold in zip(preds, targets) if self._normalize(pred) == self._normalize(gold))

        return {
            "train_batch_acc": float(exact / max(1, len(targets))),
            "train_loss": float(loss.detach().float().item()),
            "train_answer_token_acc": float(token_acc),
            "num_total_tokens": int(batch["attention_mask"].sum().item()),
            "num_supervised_tokens": int(batch["labels"].ne(-100).sum().item()),
            "num_loss_tokens": int(batch["labels"].ne(-100).sum().item()),
            "lr": float(self._last_lr),
        }

    def score_answer_nlls(self, pairs: List[Tuple[str, str]], targets: List[str]) -> List[float]:
        import torch
        import torch.nn.functional as F

        if not pairs:
            return []
        sources = self._format_inputs(pairs)
        all_nlls: List[float] = []
        was_training = self.model.training
        self.model.eval()
        try:
            for idx in range(0, len(sources), 4):
                b_sources = sources[idx : idx + 4]
                b_targets = targets[idx : idx + 4]
                batch = self._tokenize_batch(b_sources, b_targets)
                with torch.no_grad():
                    outputs = self.model(**batch, return_dict=True)
                    labels = batch["labels"]
                    token_losses = F.cross_entropy(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(labels.shape)
                for i in range(labels.size(0)):
                    mask = labels[i].ne(-100)
                    all_nlls.append(float(token_losses[i][mask].mean().item()) if bool(mask.any().item()) else 0.0)
        finally:
            if was_training:
                self.model.train()
        return all_nlls

    def score_prompt_nlls(self, prompts: List[str]) -> List[float]:
        # Seq2seq encoders do not define a label-free prompt LM likelihood.
        return [0.0 for _ in prompts]

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        *,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        import torch
        from transformers import GenerationConfig

        was_training = self.model.training
        self.model.eval()
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.cfg.max_source_len),
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        nb = int(num_beams) if num_beams is not None else int(self.cfg.gen_num_beams)
        ds = bool(self.cfg.gen_do_sample) if do_sample is None else bool(do_sample)
        gen_cfg = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            num_beams=max(1, nb),
            do_sample=ds,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            no_repeat_ngram_size=max(0, int(self.cfg.gen_no_repeat_ngram_size)),
            encoder_no_repeat_ngram_size=max(0, int(self.cfg.gen_encoder_no_repeat_ngram_size)),
            repetition_penalty=max(1.0, float(self.cfg.gen_repetition_penalty)),
            early_stopping=nb > 1,
        )
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_cfg)
        
        # T5 seq2seq generate returns ONLY the generated sequence (starting with decoder_start_token_id).
        # We don't need to slice off the prompt.
        texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if was_training:
            self.model.train()
        return [str(t) for t in texts]

    def generate_with_ids(self, prompts: List[str], max_new_tokens: int = 64) -> List[Dict[str, Any]]:
        texts = self.generate(prompts, max_new_tokens=max_new_tokens)
        return [
            {
                "infer_prompt_token_ids": [],
                "infer_prompt_token_len": 0,
                "generated_full_ids": [],
                "generated_continuation_ids": [],
                "decoded_prompt_tail": "",
                "decoded_continuation_head": "",
                "raw_generated_text": text,
            }
            for text in texts
        ]

    def get_activations(self, prompts: List[str]) -> List[List[float]]:
        pooled = self.get_activations_tensor(prompts, with_grad=False)
        return [pooled[i].detach().cpu().float().tolist() for i in range(pooled.shape[0])]

    def get_activations_tensor(self, prompts: List[str], *, with_grad: bool) -> "Any":
        import torch

        was_training = self.model.training
        self.model.eval()
        batch_size = max(1, int(self.cfg.activation_batch_size))
        pooled_batches = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.cfg.max_source_len),
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if with_grad:
                enc_out = self.model.get_encoder()(**inputs, output_hidden_states=True, return_dict=True)
            else:
                with torch.no_grad():
                    enc_out = self.model.get_encoder()(**inputs, output_hidden_states=True, return_dict=True)
            hidden = enc_out.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            pooled_batches.append(pooled if with_grad else pooled.detach().cpu())
        pooled = torch.cat(pooled_batches, dim=0) if pooled_batches else torch.empty((0, 0))
        if was_training:
            self.model.train()
        return pooled

    @staticmethod
    def _normalize(text: str) -> str:
        return str(text or "").strip().lower()


def build_seq2seq_backbone(model_cfg: Dict[str, Any], *, seed: int) -> HFSeq2SeqLMBackbone:
    hf_path = str(model_cfg.get("hf_model_name_or_path", "")).strip()
    if not hf_path:
        raise ValueError("hf_model_name_or_path is required for seq2seq_lm backbone.")
    cfg = HFSeq2SeqLMConfig(
        hf_model_name_or_path=hf_path,
        torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
        device=str(model_cfg.get("device", "auto")),
        max_source_len=int(model_cfg.get("max_source_len", model_cfg.get("max_seq_len", 1024))),
        max_target_len=int(model_cfg.get("max_target_len", 256)),
        gen_max_new_tokens=int(model_cfg.get("gen_max_new_tokens", 64)),
        gen_do_sample=bool(model_cfg.get("gen_do_sample", False)),
        gen_num_beams=int(model_cfg.get("gen_num_beams", 1)),
        gen_no_repeat_ngram_size=int(model_cfg.get("gen_no_repeat_ngram_size", 0)),
        gen_encoder_no_repeat_ngram_size=int(model_cfg.get("gen_encoder_no_repeat_ngram_size", 0)),
        gen_repetition_penalty=float(model_cfg.get("gen_repetition_penalty", 1.0)),
        format_style=str(model_cfg.get("format_style", "citb_t5")),
        debug_print_formatted_examples=bool(model_cfg.get("debug_print_formatted_examples", False)),
        debug_print_tokenized_examples=bool(model_cfg.get("debug_print_tokenized_examples", False)),
        debug_max_tokenized_examples=int(model_cfg.get("debug_max_tokenized_examples", 2)),
        activation_batch_size=int(model_cfg.get("activation_batch_size", 4)),
    )
    return HFSeq2SeqLMBackbone(cfg, seed=seed)
