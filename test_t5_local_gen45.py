import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's check the base model's generation on the dataset with max_new_tokens=64 and without any other kwargs, but skip_special_tokens=True.")

with open("/root/autodl-tmp/lora-baselines-run_v1/data/ccfa_three_suite/citb/citb_instrdialog_order1_train500_dev50_test100.json", "r") as f:
    data = json.load(f)
    
for i in range(3):
    ex = data["stream"][0]["train"][i]
    instruction = ex["instruction"]
    input_text = ex["input"]
    target = ex["output"]

    prompt = format_citb_t5(instruction, input_text)
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=64
        )
    
    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"Example {i}")
    print(f"Target: {target}")
    print(f"Pred: {pred}")
    print("-" * 20)
