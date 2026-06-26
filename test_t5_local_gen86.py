import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to("cuda")

print("Let's check the loss on the dev set of the first task, using the exact formatting from seq2seq_lora_wrapper, and on CUDA, and with bfloat16.")
model = model.bfloat16()

with open("/root/autodl-tmp/lora-baselines-run_v1/data/ccfa_three_suite/citb/citb_instrdialog_order1_train500_dev50_test100.json", "r") as f:
    data = json.load(f)
    
losses = []
for i in range(10):
    ex = data["stream"][0]["dev"][i]
    instruction = ex["instruction"]
    input_text = ex["input"]
    target = ex["output"]

    prompt = format_citb_t5(instruction, input_text)
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to("cuda")
    labels = tokenizer(
        text_target=[str(target)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )["input_ids"].to("cuda")
    pad_id = int(tokenizer.pad_token_id)
    labels = labels.masked_fill(labels == pad_id, -100)

    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
    
    losses.append(outputs.loss.item())

print("Average loss on 10 dev examples:", sum(losses) / len(losses))
