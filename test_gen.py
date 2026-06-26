import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from core.formatting import format_for_infer

path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
setattr(tokenizer, "_ours_format_style", "citb_t5")
model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")

instructions = [
    "You are a helpful assistant. Please answer the following question.",
    "Translate the following English text to French."
]
inputs_list = [
    "What is the capital of France?",
    "Hello world"
]

for inst, inp in zip(instructions, inputs_list):
    prompt = format_for_infer(tokenizer, inst, inp, add_generation_prompt=True)
    print(f"Prompt:\n{prompt}")
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Generated:\n{decoded}\n")

