import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Let's read the first example from the dataset
with open("/root/autodl-tmp/data_cache/citb_instrdialog_order1_seed1.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if "train" in data:
            ex = data["train"][0]
            break

instruction = ex["instruction"]
input_text = ex["input"]
target = ex["output"]
print("Instruction:", instruction)
print("Input:", input_text)
print("Target:", target)

prompt = format_citb_t5(instruction, input_text)
prompts = [prompt]
inputs = tokenizer(prompts, return_tensors="pt")

gen_cfg = GenerationConfig(
    max_new_tokens=64,
    num_beams=1,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=model.config.decoder_start_token_id,
    early_stopping=False,
)

with torch.no_grad():
    output_ids = model.generate(**inputs, generation_config=gen_cfg)

print("Output ids:", output_ids)
print("Decoded output:", tokenizer.batch_decode(output_ids, skip_special_tokens=True))
