import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, '/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl')

import torch

import train as todcl_train

OriginalTrainer = todcl_train.Trainer
OriginalGetDataLoaders = todcl_train.get_data_loaders

class BoundedTrainer(OriginalTrainer):
    def __init__(self, *args, **kwargs):
        kwargs['limit_train_batches'] = 1
        kwargs['limit_val_batches'] = 1
        kwargs['num_sanity_val_steps'] = 0
        kwargs['checkpoint_callback'] = True
        super().__init__(*args, **kwargs)

    def fit(self, model, *args, **kwargs):
        result = super().fit(model, *args, **kwargs)
        best_path = getattr(self.checkpoint_callback, 'best_model_path', '')
        if not best_path:
            ckpt_dir = Path(self.default_root_dir) / 'bounded_smoke_ckpt'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / 'bounded-smoke.ckpt'
            torch.save({'state_dict': model.state_dict()}, ckpt_path)
            self.checkpoint_callback.best_model_path = str(ckpt_path)
            print('BOUNDED_CHECKPOINT_WRITTEN', ckpt_path, flush=True)
        else:
            print('BOUNDED_CHECKPOINT_FOUND', best_path, flush=True)
        return result


def bounded_get_data_loaders(args, tokenizer, test=False):
    train_loader, val_loader, dev_val_loader, datasets = OriginalGetDataLoaders(args, tokenizer, test=test)
    train_datasets, val_datasets, test_datasets = datasets
    if isinstance(train_loader, dict) and train_loader:
        first_key = sorted(train_loader.keys())[0]
        train_loader = {first_key: train_loader[first_key]}
        val_loader = {first_key: val_loader[first_key]}
        train_datasets = {first_key: train_datasets[first_key]}
        val_datasets = {first_key: val_datasets[first_key]}
        print('BOUNDED_TASK_SELECTED', first_key, 'train_batches', len(train_loader[first_key]), 'val_batches', len(val_loader[first_key]), flush=True)
    return train_loader, val_loader, dev_val_loader, (train_datasets, val_datasets, test_datasets)

def bounded_test_model_seq2seq(args, model, tokenizer, test_loader, time="0_['']"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    results = []
    for idx_b, batch in enumerate(test_loader):
        if idx_b >= 1:
            break
        with torch.no_grad():
            if 'gpt2' in args.model_checkpoint:
                from test import test_generation_GPT2BATCH
                value_batch, _ = test_generation_GPT2BATCH(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=[b + '[SOS]' for b in batch['history']],
                    device=device,
                    max_length=8,
                )
            else:
                responses = model.generate(
                    input_ids=batch['encoder_input'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=8,
                )
                value_batch = tokenizer.batch_decode(responses, skip_special_tokens=True)
        for idx, resp in enumerate(value_batch):
            results.append({
                'id': batch['dial_id'][idx],
                'turn_id': batch['turn_id'][idx],
                'dataset': batch['dataset'][idx],
                'task_id': batch['task_id'][idx],
                'spk': batch['spk'][idx],
                'gold': batch['reply'][idx],
                'genr': resp,
                'hist': batch['history'][idx],
            })
    out_dir = Path(args.saving_dir) / time
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'generated_responses.json').open('w') as fp:
        json.dump(results, fp, indent=2)
    print('BOUNDED_EVAL_WRITTEN', out_dir / 'generated_responses.json', 'rows', len(results), flush=True)
    tokenizer.padding_side = 'right'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', default='tiny-gpt2-local')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--valid_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--dataset_list', default='SGD')
    parser.add_argument('--max_history', type=int, default=5)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--setting', default='single')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test_every_step', action='store_true')
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--bottleneck_size', type=int, default=100)
    parser.add_argument('--number_of_adpt', type=int, default=40)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--percentage_LAM0L', type=float, default=0.2)
    parser.add_argument('--reg', type=float, default=0.01)
    parser.add_argument('--episodic_mem_size', type=int, default=1)
    parser.add_argument('--task_type', default='NLG')
    parser.add_argument('--CL', default='VANILLA')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args([])
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    todcl_train.Trainer = BoundedTrainer
    todcl_train.get_data_loaders = bounded_get_data_loaders
    todcl_train.test_model_seq2seq = bounded_test_model_seq2seq
    print('BOUNDED_METHOD_SMOKE_START', vars(args), flush=True)
    todcl_train.train(args)
    print('BOUNDED_METHOD_SMOKE_DONE', args.saving_dir, flush=True)

if __name__ == '__main__':
    main()
