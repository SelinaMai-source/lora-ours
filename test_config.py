from transformers import AutoModelForSeq2SeqLM
model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
print("decoder_start_token_id:", model.config.decoder_start_token_id)
print("bos_token_id:", model.config.bos_token_id)
print("pad_token_id:", model.config.pad_token_id)
