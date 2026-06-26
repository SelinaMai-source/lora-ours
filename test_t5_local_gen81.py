import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's check if the model produces valid logits for a prompt from the dataset, step 2, but with token 71 ('A') as the first token.")

with open("/root/autodl-tmp/lora-baselines-run_v1/data/ccfa_three_suite/citb/citb_instrdialog_order1_train500_dev50_test100.json", "r") as f:
    data = json.load(f)
    
ex = data["stream"][0]["train"][0]
instruction = ex["instruction"]
input_text = ex["input"]
target = ex["output"]

prompt = format_citb_t5(instruction, input_text)
inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id, 71]])

with torch.no_grad():
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=decoder_input_ids
    )

logits = outputs.logits
print("Logits shape:", logits.shape)
print("Top 5 token IDs for second step:", torch.topk(logits[0, 1, :], 5).indices.tolist())
print("Top 5 token strings for second step:", [tokenizer.decode([idx]) for idx in torch.topk(logits[0, 1, :], 5).indices.tolist()])
