import yaml
with open('configs/debug_strict.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['model']['debug_print_formatted_examples'] = True
cfg['model']['debug_print_tokenized_examples'] = True
with open('configs/debug_strict.yaml', 'w') as f:
    yaml.safe_dump(cfg, f)
