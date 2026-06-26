import json
from transformers import AutoTokenizer
from core.formatting import format_for_infer, format_for_train

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

instruction = "Translate to French."
input_text = "Hello world"
target = "Bonjour le monde"

prompt_text = format_for_infer(tokenizer, instruction, input_text, add_generation_prompt=True)
full_text = format_for_train(tokenizer, instruction, input_text, target)["full_text"]

prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

print("Prompt tokens:", len(prompt_ids))
print("Full tokens:", len(full_ids))
print("Prompt text:", repr(prompt_text))
print("Full text:", repr(full_text))

print("Prompt ids:", prompt_ids)
print("Full ids prefix:", full_ids[:len(prompt_ids)+2])

print("Decoded prompt ids:", [tokenizer.decode([t]) for t in prompt_ids[-5:]])
print("Decoded full ids:", [tokenizer.decode([t]) for t in full_ids[len(prompt_ids)-5:len(prompt_ids)+2]])
