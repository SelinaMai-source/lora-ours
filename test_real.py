import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from core.formatting import format_for_infer

path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
setattr(tokenizer, "_ours_format_style", "citb_t5")
model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")

data = json.load(open('/root/autodl-tmp/lora-baselines-run_v1/data/ccfa_three_suite/citb/citb_instrdialog_order1_train500_dev50_test100.json'))
ex = data['stream'][0]['eval'][0]
prompt = format_for_infer(tokenizer, ex['instruction'], ex['input'])
inputs = tokenizer([prompt], return_tensors="pt", max_length=1024, truncation=True).to(model.device)
labels = tokenizer(text_target=[ex['output']], return_tensors="pt")["input_ids"]
labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100).to(model.device)
batch = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": labels}

model.eval()
with torch.no_grad():
    loss = model(**batch).loss
    print("Eval Loss:", loss.item())

model.train()
with torch.no_grad():
    loss = model(**batch).loss
    print("Train Loss 1:", loss.item())
    loss = model(**batch).loss
    print("Train Loss 2:", loss.item())
