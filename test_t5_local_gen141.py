import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5
import os

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float32)

print("Let's check the model config.")
print(model.config)

print("\nLet's check the model weights.")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}, min={param.min().item():.4f}, max={param.max().item():.4f}")
