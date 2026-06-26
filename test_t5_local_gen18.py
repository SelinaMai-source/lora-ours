from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's check if the model weights are all zeros or NaNs.")
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in {name}")
    if torch.all(param == 0):
        print(f"All zeros in {name}")
    
print("Checking norm of embedding layer:", model.shared.weight.norm().item())
print("Checking norm of first encoder layer:", model.encoder.block[0].layer[0].SelfAttention.q.weight.norm().item())
print("Checking norm of lm_head:", model.lm_head.weight.norm().item())
