import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/t5-small-lm-adapt")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small-lm-adapt")

instruction = "Translate to French."
input_text = "Hello world"
target = "Bonjour le monde"

from core.formatting import format_citb_t5
prompt = format_citb_t5(instruction, input_text)

enc = tokenizer(prompt, return_tensors="pt")
print("Prompt ids:", enc["input_ids"])

from transformers import GenerationConfig

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
    output_ids = model.generate(**enc, generation_config=gen_cfg)

print("Output ids:", output_ids)
print("Decoded output:", tokenizer.batch_decode(output_ids, skip_special_tokens=True))
