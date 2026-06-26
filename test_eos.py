from transformers import AutoModelForSeq2SeqLM
model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
print("model eos_token_id:", model.config.eos_token_id)
