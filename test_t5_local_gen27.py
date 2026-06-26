from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to("cuda")

print("Let's check the generation of the model when on CUDA.")
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        num_beams=1,
        do_sample=False,
    )

print("Output:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
