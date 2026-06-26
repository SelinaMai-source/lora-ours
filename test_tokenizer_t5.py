from transformers import AutoTokenizer
model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("text_target:", tokenizer(text_target=["1"])["input_ids"])
print("text:", tokenizer(["1"])["input_ids"])
