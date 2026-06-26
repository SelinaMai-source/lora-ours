from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's see the raw token IDs generated:")
inputs = tokenizer("Translate English to German: Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=20, 
        num_beams=1, 
        do_sample=False,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0
    )
print("Raw outputs:", outputs)
print("Decoded without skip_special:", tokenizer.batch_decode(outputs, skip_special_tokens=False))
