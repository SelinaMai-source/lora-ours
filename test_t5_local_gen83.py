import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's check the loss on the dev set of the first task.")

with open("/root/autodl-tmp/lora-baselines-run_v1/data/ccfa_three_suite/citb/citb_instrdialog_order1_train500_dev50_test100.json", "r") as f:
    data = json.load(f)
    
losses = []
for i in range(10):
    ex = data["stream"][0]["dev"][i]
    instruction = ex["instruction"]
    input_text = ex["input"]
    target = ex["output"]

    prompt = format_citb_t5(instruction, input_text)
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)
    labels = tokenizer(text_target=[target], return_tensors="pt")["input_ids"]
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
    
    losses.append(outputs.loss.item())

print("Average loss on 10 dev examples:", sum(losses) / len(losses))
