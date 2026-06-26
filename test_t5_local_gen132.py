import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5
import os

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

print("Let's check the generation of the base model WITHOUT CUDA, on CPU.")

with open("/root/autodl-tmp/lora-baselines-run_v1/data/ccfa_three_suite/citb/citb_instrdialog_order1_train500_dev50_test100.json", "r") as f:
    data = json.load(f)
    
# Find an example that is not a classification task
found = False
for segment in data["stream"]:
    for i, ex in enumerate(segment["train"]):
        if "translate" in ex["instruction"].lower() or "summarize" in ex["instruction"].lower() or "answer" in ex["instruction"].lower():
            if i + 51 < len(segment["train"]):
                ex = segment["train"][i+51]
                found = True
                break
    if found:
        break

instruction = ex["instruction"]
input_text = ex["input"]
target = ex["output"]

prompt = format_citb_t5(instruction, input_text)
inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=64,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        num_return_sequences=5
    )

preds = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
print(f"Instruction: {instruction}")
print(f"Target: {target}")
for i, pred in enumerate(preds):
    print(f"Pred {i}: {pred}")
