# ToDCL ADAPTER NLG Official Anchor — 20260706

- Updated: 2026-07-07T04:02:29+08:00
- State: `failed_blocker`
- Session: `lora-ours-todcl-adapter-anchor`
- Log: `/root/autodl-tmp/lora-ours-logs/todcl_adapter_nlg_official_anchor_20260706.log`
- Launcher: `scripts/run_todcl_adapter_nlg_official_anchor.sh`
- Paper reference: BLEU **21.7719**, EER **0.164**
- Local BLEU: —
- Local EER: —
- Notes: GPT-2 pytorch_model.bin corrupt/missing; HF download unreachable — unrecoverable without model fix

## Exit / error signals
- Some weights of GPT2Adapter were not initialized from the model checkpoint at gpt2 and are newly initialized: ['h.0.attn.masked_bias', 'h.1.attn.masked_bias', 'h.2.attn.masked_bias', 'h.3.attn.masked_bias', 'h.4.attn.masked_bias', 'h.5.attn.masked_bias', 'h.6.attn.masked_bias', 'h.7.attn.masked_bias', 'h.8.attn.masked_bias', 'h.9.attn.masked_bias', 'h.10.attn.masked_bias', 'h.11.attn.masked_bias', 'lm_head.weight']
- You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
- Traceback (most recent call last):
-   File "train.py", line 221, in <module>
-     train(hyperparams)
-   File "train.py", line 49, in train
-     model = Seq2SeqToD(hparams)
-   File "/root/autodl-tmp/lora-baselines-run_v1/external_sources/todcl/CL_learner.py", line 30, in __init__
-     tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint, bos_token="[bos]", eos_token="[eos]", sos_token="[SOS]", sep_token="[sep]",pad_token='[PAD]')
-   File "/root/autodl-tmp/conda_envs/todcl_legacy_py37/lib/python3.7/site-packages/transformers/tokenization_utils_base.py", line 1629, in from_pretrained
-     local_files_only=local_files_only,
-   File "/root/autodl-tmp/conda_envs/todcl_legacy_py37/lib/python3.7/site-packages/transformers/file_utils.py", line 955, in cached_path
-     local_files_only=local_files_only,
-   File "/root/autodl-tmp/conda_envs/todcl_legacy_py37/lib/python3.7/site-packages/transformers/file_utils.py", line 1125, in get_from_cache
-     "Connection error, and we cannot find the requested files in the cached path."
- ValueError: Connection error, and we cannot find the requested files in the cached path. Please try again or make sure your Internet connection is on.
- EXIT=1
