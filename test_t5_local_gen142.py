import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5
import os

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float32)

print("Let's check the lm_head weights.")
lm_head_weight = model.lm_head.weight
print(f"lm_head.weight: mean={lm_head_weight.mean().item():.4f}, std={lm_head_weight.std().item():.4f}, min={lm_head_weight.min().item():.4f}, max={lm_head_weight.max().item():.4f}")

# Check if the weights of shared and lm_head are tied
if torch.equal(model.shared.weight, model.lm_head.weight):
    print("shared.weight and lm_head.weight are tied.")
else:
    print("shared.weight and lm_head.weight are NOT tied.")
