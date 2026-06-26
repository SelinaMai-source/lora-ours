import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Wait, look at the target ids: tensor([[  3, 632,   1]])")
print("Token 3 is", tokenizer.decode([3]))
print("Token 632 is", tokenizer.decode([632]))
print("Token 1 is", tokenizer.decode([1]))

print("The target '0' is tokenized as [3, 632, 1] instead of just [632, 1] or [something else].")
print("Let's see what tokenizer(text_target=['0']) gives:")
print(tokenizer(text_target=['0']))
print("Let's see what tokenizer(text_target=['1']) gives:")
print(tokenizer(text_target=['1']))

print("Ah, token 3 is a space. Token 632 is '0'.")
