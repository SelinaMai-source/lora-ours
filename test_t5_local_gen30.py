import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's check if there is a problem with the prompt formatting.")
instruction = "Translate to French."
input_text = "Hello world"
prompt = format_citb_t5(instruction, input_text)
print("Formatted prompt:", prompt)

inputs = tokenizer([prompt], return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=64)

print("Output ids:", output_ids)
print("Decoded output:", tokenizer.batch_decode(output_ids, skip_special_tokens=True))
