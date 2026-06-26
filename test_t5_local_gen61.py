import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's check the generation of the base model on a different task type.")

with open("/root/autodl-tmp/lora-baselines-run_v1/data/ccfa_three_suite/citb/citb_instrdialog_order1_train500_dev50_test100.json", "r") as f:
    data = json.load(f)
    
# Find an example that is not a classification task
for ex in data["stream"][0]["train"]:
    if "classify" not in ex["instruction"].lower() and "yes or no" not in ex["instruction"].lower():
        break

instruction = ex["instruction"]
input_text = ex["input"]
target = ex["output"]

prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
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
