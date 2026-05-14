# Task 1.2 — Implementation Notes

## What We Learned from SmolLM2 Training

### TRL API Changes (current Colab TRL version)
The TRL library changed its API in recent versions. Use these exact signatures:

| Old (broken) | New (working) |
|---|---|
| `TrainingArguments` | `SFTConfig` (imported from `trl`) |
| `max_seq_length=` in `SFTTrainer()` | `max_seq_length=` / `max_length=` in `SFTConfig` |
| `dataset_text_field=` in `SFTTrainer()` | `dataset_text_field=` in `SFTConfig` |
| `tokenizer=` in `SFTTrainer()` | `processing_class=` in `SFTTrainer()` |

### OOM / Colab Disconnect Fix
Default batch size (4) + sequence length (512) caused GPU RAM exhaustion on T4, killing the session.

Working config for SmolLM2 on T4:
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8` (effective batch = 8)
- `max_length=256`
- `model.gradient_checkpointing_enable()` before training
- `torch.cuda.empty_cache()` before training

### SmolLM2 Adapter Config (confirmed working)
- `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
- `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`
- `task_type=CAUSAL_LM`
- `save_strategy="no"` — avoid cluttering Colab disk

---

## flan-t5-small Key Differences

### Architecture
- SmolLM2 is **decoder-only** (causal LM) → uses `SFTTrainer`, single `text` field
- flan-t5-small is **encoder-decoder** (seq2seq) → uses `Seq2SeqTrainer`, separate `input` / `target` fields

### LoRA Config Differences
| | SmolLM2 | flan-t5-small |
|---|---|---|
| `task_type` | `CAUSAL_LM` | `SEQ_2_SEQ_LM` |
| `target_modules` | `["q_proj","k_proj","v_proj","o_proj"]` | `["q","v"]` |

flan-t5 names its attention projections `q` and `v` (no `_proj` suffix).

### Training API
`Seq2SeqTrainer` is standard HuggingFace (not TRL) so `tokenizer=` still works — no `processing_class` needed.

### Tokenization
Must tokenize input and target separately, then assign target `input_ids` as `labels`.
Set `padding="max_length"` so DataCollator doesn't need to pad further.

### Memory
flan-t5-small (~77M params) is much smaller than SmolLM2 (360M), so batch size can be higher.
Use `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`.
