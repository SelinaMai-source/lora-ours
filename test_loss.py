import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")

sources = ["Definition: classify this text.\n\nNow complete the following example -\nInput: text here\nOutput: ", "Definition: classify this text.\n\nNow complete the following example -\nInput: other text\nOutput: "]
targets = ["1", "0"]

enc = tokenizer(sources, padding=True, return_tensors="pt")
labels = tokenizer(text_target=targets, padding=True, return_tensors="pt")["input_ids"]
labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

batch = {"input_ids": enc.input_ids.to(model.device), "attention_mask": enc.attention_mask.to(model.device), "labels": labels.to(model.device)}

with torch.no_grad():
    outputs = model(**batch)
print("Loss:", outputs.loss.item())
