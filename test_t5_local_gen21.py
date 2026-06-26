from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's check if the model is producing valid logits.")
inputs = tokenizer("Translate English to German: Hello world", return_tensors="pt")
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

with torch.no_grad():
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=decoder_input_ids
    )

logits = outputs.logits
print("Logits shape:", logits.shape)
print("Logits min/max/mean:", logits.min().item(), logits.max().item(), logits.mean().item())
print("Top 5 token IDs for first step:", torch.topk(logits[0, 0, :], 5).indices.tolist())
print("Top 5 token strings for first step:", [tokenizer.decode([idx]) for idx in torch.topk(logits[0, 0, :], 5).indices.tolist()])
