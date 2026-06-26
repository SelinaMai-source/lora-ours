from transformers import AutoTokenizer
model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
targets = ["1"]
labels = tokenizer(text_target=targets, return_tensors="pt")["input_ids"]
print("Labels:", labels)
print("Decoded labels:", tokenizer.batch_decode(labels))
