import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5
import os

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float32)

print("Let's check the generation of the base model WITHOUT CUDA, on CPU, with float32, and with a simple prompt, and without do_sample, and with a different prompt.")

prompt = "Hello"
inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=64
    )

preds = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
print(f"Prompt: {prompt}")
for i, pred in enumerate(preds):
    print(f"Pred {i}: {pred}")
