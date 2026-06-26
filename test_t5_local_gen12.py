from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Tokenizer class:", type(tokenizer))
print("Model class:", type(model))

# Let's try a very simple generation without formatting
inputs = tokenizer("Translate English to German: Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
print("Simple generation:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
