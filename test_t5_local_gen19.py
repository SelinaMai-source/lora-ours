from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "google/t5-small-lm-adapt"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Checking norm of embedding layer for original model:", model.shared.weight.norm().item())
print("Checking norm of first encoder layer for original model:", model.encoder.block[0].layer[0].SelfAttention.q.weight.norm().item())
print("Checking norm of lm_head for original model:", model.lm_head.weight.norm().item())
