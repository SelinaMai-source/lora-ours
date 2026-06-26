import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from core.models.lora_wrapper import build_lora_wrapper
from core.models.seq2seq_lora_wrapper import build_seq2seq_backbone
from core.formatting import format_for_infer
import yaml

with open('configs/debug_strict.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

model = build_seq2seq_backbone(cfg['model'], seed=1)
lora = build_lora_wrapper(model, cfg['lora'])

data = json.load(open('/root/autodl-tmp/lora-baselines-run_v1/data/ccfa_three_suite/citb/citb_instrdialog_order1_train500_dev50_test100.json'))
segment = data['stream'][0]

pairs = [(ex['instruction'], ex['input']) for ex in segment['train']]
targets = [ex['output'] for ex in segment['train']]

# Setup optim manually to see if it makes a difference, normally ours has a bug in optim?
import torch.optim as optim
optimizer = optim.AdamW(lora.peft_model.parameters(), lr=1e-4)

for epoch in range(5):
    for i in range(0, len(pairs), 8):
        b_pairs = pairs[i:i+8]
        b_targets = targets[i:i+8]
        metrics = model.fit_batch(b_pairs, b_targets, lr=1e-4)
        
        # Test BOTH step_adapter and direct optim step
        # lora.step_adapter()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch} loss: {metrics['train_loss']}")

model.model.eval()
ex = data['stream'][0]['eval'][0]
prompt = format_for_infer(model.tokenizer, ex['instruction'], ex['input'])
inputs = model.tokenizer([prompt], return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.model.generate(**inputs, max_new_tokens=64, decoder_start_token_id=0, num_beams=1) # greedy
decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
print("Raw tokens:", outputs)
print("Generated after training:", decoded)
print("Target:", ex['output'])
