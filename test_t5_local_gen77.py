import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's test the model on a prompt that it was definitely trained on in stage 1, with a different format.")
prompt = "Task: Translate the following sentence to French.\nSentence: Hello world\nTranslation:"
inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=64,
        do_sample=True,
        top_k=50,
        temperature=0.7,
        num_return_sequences=5
    )

preds = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
for i, pred in enumerate(preds):
    print(f"Pred {i}: {pred}")
