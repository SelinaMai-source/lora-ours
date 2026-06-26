import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from core.formatting import format_citb_t5

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

instruction = "Translate to German: Hello world"
input_text = ""
target = "Hallo Welt"

prompt = format_citb_t5(instruction, input_text)
# Try truncating the prompt to avoid the indexing error
inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=512)

# Try with explicit generation config
gen_cfg = GenerationConfig(
    max_length=64,
    num_beams=4,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=model.config.decoder_start_token_id,
    early_stopping=False,
)

with torch.no_grad():
    output_ids2 = model.generate(**inputs, generation_config=gen_cfg)

pred2 = tokenizer.batch_decode(output_ids2, skip_special_tokens=True)[0]
print("Explicit Gen Target:", target)
print("Explicit Gen Pred:", pred2)
print("Explicit Gen Output ids:", output_ids2)
