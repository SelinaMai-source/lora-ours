import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Wait, is the model actually trained on this data? Let's check the training script.")
print("The model is base_epoch15_lr1e-05_seed469. Let's see if it's a T5 model trained with causal LM objective by mistake.")
print("Model config architecture:", model.config.architectures)
