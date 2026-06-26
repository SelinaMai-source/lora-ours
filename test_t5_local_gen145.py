import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5
import os

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float32)

print("Let's check the generation of the base model WITHOUT CUDA, on CPU, with float32, and with a simple prompt, and without do_sample, and with a different prompt.")

prompt = "Translate English to French: Hello world"
inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    encoder_outputs = model.encoder(**inputs)
    
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id, 10465, 18686]])
    decoder_outputs = model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_outputs.last_hidden_state)
    
    lm_logits = model.lm_head(decoder_outputs.last_hidden_state)
    
    top_k = torch.topk(lm_logits[0, -1, :], 10)
    print("\nTop 10 predicted tokens for step 3:")
    for i in range(10):
        token_id = top_k.indices[i].item()
        token_str = tokenizer.decode([token_id])
        prob = torch.softmax(lm_logits[0, -1, :], dim=-1)[token_id].item()
        print(f"Token {token_id}: '{token_str}' (prob: {prob:.4f})")
