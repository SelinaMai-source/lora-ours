import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

with open("/root/autodl-tmp/lora-baselines-run_v1/data/ccfa_three_suite/citb/citb_instrdialog_order1_train500_dev50_test100.json", "r") as f:
    data = json.load(f)
    ex = data["stream"][0]["train"][0]

instruction = ex["instruction"]
input_text = ex["input"]
target = ex["output"]

prompt = format_citb_t5(instruction, input_text)
# Try truncating the prompt to avoid the indexing error
inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=64)

pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print("Target:", target)
print("Pred:", pred)
print("Output ids:", output_ids)
