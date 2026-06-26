from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "/root/autodl-tmp/model_cache/citb_superni_stage1/base_epoch15_lr1e-05_seed469"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Let's see if the model has a bad state or if it's just this prompt.")
prompts = [
    "What is the capital of France?",
    "Summarize this text: The quick brown fox jumps over the lazy dog.",
    "Solve 2 + 2",
    "def hello_world():",
    "Instruction: Answer yes or no. Input: Is the sky blue?"
]

inputs = tokenizer(prompts, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=20, 
        num_beams=1, 
        do_sample=False,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0
    )
    
for i, out in enumerate(outputs):
    print(f"Prompt: {prompts[i]}")
    print(f"Output: {tokenizer.decode(out, skip_special_tokens=True)}")
    print("-" * 20)
