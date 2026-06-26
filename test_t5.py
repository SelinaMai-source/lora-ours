import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/t5-small-lm-adapt")

instruction = "Translate to French."
input_text = "Hello world"
target = "Bonjour le monde"

from core.formatting import format_citb_t5
prompt = format_citb_t5(instruction, input_text)
print("Prompt:", repr(prompt))

enc = tokenizer(prompt)
print("Prompt ids:", enc["input_ids"])
print("Decoded prompt:", tokenizer.decode(enc["input_ids"]))

target_enc = tokenizer(text_target=target)
print("Target ids:", target_enc["input_ids"])
print("Decoded target:", tokenizer.decode(target_enc["input_ids"]))

print("Decoded target without special:", tokenizer.decode(target_enc["input_ids"], skip_special_tokens=True))
