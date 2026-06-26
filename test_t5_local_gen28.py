import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("What if we pass the generation config explicitly like in the code, but without max_new_tokens?")

inputs = tokenizer("Hello", return_tensors="pt")

gen_cfg = GenerationConfig(
    max_length=64,
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
