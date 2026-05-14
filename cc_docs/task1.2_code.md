# Task 1.2 — QLoRA Fine-Tuning Code

## GPU

Use T4 for both models. No need to switch.
- SmolLM2-360M: ~180MB in 4-bit, trained fine on T4
- flan-t5-small: ~40MB in 4-bit, much smaller — T4 has headroom to spare

---

## SmolLM2-360M-Instruct (working version)

### Cell 1 — Load datasets + QLoRA config

Paste after Task 1.1 cell, before model loading.

```python
# ── Load Datasets into HuggingFace format ──────────────────────────────────
train_causal_ds = Dataset.from_list(pd.read_json(TRAIN_CAUSAL, lines=True).to_dict('records'))
test_causal_ds = Dataset.from_list(pd.read_json(TEST_CAUSAL, lines=True).to_dict('records'))
train_seq2seq_ds = Dataset.from_list(pd.read_json(TRAIN_SEQ2SEQ, lines=True).to_dict('records'))
test_seq2seq_ds = Dataset.from_list(pd.read_json(TEST_SEQ2SEQ, lines=True).to_dict('records'))

# ── QLoRA Config (shared) ─────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

print("Datasets loaded and QLoRA configuration prepared.")
```

### Cell 2 — Load SmolLM2

```python
model_id_causal = "HuggingFaceTB/SmolLM2-360M-Instruct"

tokenizer_causal = AutoTokenizer.from_pretrained(model_id_causal)
tokenizer_causal.pad_token = tokenizer_causal.eos_token

model_causal = AutoModelForCausalLM.from_pretrained(
    model_id_causal,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model_causal = prepare_model_for_kbit_training(model_causal)
model_causal = get_peft_model(model_causal, lora_config)

model_causal.print_trainable_parameters()
print("SmolLM2 model loaded and adapters attached.")
```

### Cell 3 — Train SmolLM2 + save to Drive

```python
from trl import SFTTrainer, SFTConfig

torch.cuda.empty_cache()
model_causal.gradient_checkpointing_enable()

smollm2_config = SFTConfig(
    output_dir="./smollm2_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    max_length=256,
    dataset_text_field="text",
)

trainer_causal = SFTTrainer(
    model=model_causal,
    train_dataset=train_causal_ds,
    args=smollm2_config,
    processing_class=tokenizer_causal,
)

print("Training SmolLM2-360M-Instruct...")
trainer_causal.train()
print("Training complete.")

model_causal.save_pretrained("smollm2_adapter")
tokenizer_causal.save_pretrained("smollm2_adapter")

import os
for fname in os.listdir("smollm2_adapter"):
    upload_to_drive(f"smollm2_adapter/{fname}", SMOLLM2_ADAPTER_ID)

print("SmolLM2 adapter saved to Drive.")
```

**API notes (TRL breaking changes):**
- Use `SFTConfig` not `TrainingArguments`
- `max_length` and `dataset_text_field` go in `SFTConfig`, not `SFTTrainer`
- `processing_class=` not `tokenizer=` in `SFTTrainer`
- `gradient_checkpointing_enable()` required to avoid OOM on T4

---

## flan-t5-small (encoder-decoder)

No GPU change needed. T4 has plenty of headroom for this model.

### Cell 4 — Load flan-t5-small

```python
import torch
torch.cuda.empty_cache()

model_id_seq2seq = "google/flan-t5-small"
tokenizer_seq2seq = AutoTokenizer.from_pretrained(model_id_seq2seq)

# flan-t5 uses "q"/"v" not "q_proj"/"v_proj"
lora_config_seq2seq = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model_seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    model_id_seq2seq,
    quantization_config=bnb_config,
    device_map="auto",
)

model_seq2seq = prepare_model_for_kbit_training(model_seq2seq)
model_seq2seq = get_peft_model(model_seq2seq, lora_config_seq2seq)
model_seq2seq.print_trainable_parameters()
print("flan-t5-small loaded and adapters attached.")
```

### Cell 5 — Train flan-t5 + save to Drive

```python
def preprocess_seq2seq(examples):
    model_inputs = tokenizer_seq2seq(
        examples["input"], max_length=128, truncation=True, padding="max_length"
    )
    labels = tokenizer_seq2seq(
        examples["target"], max_length=256, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_seq2seq_tokenized = train_seq2seq_ds.map(
    preprocess_seq2seq, batched=True, remove_columns=["input", "target"]
)

data_collator = DataCollatorForSeq2Seq(tokenizer_seq2seq, model=model_seq2seq, padding=True)

seq2seq_args = Seq2SeqTrainingArguments(
    output_dir="./flant5_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="no",
    predict_with_generate=False,
    report_to="none",
)

trainer_seq2seq = Seq2SeqTrainer(
    model=model_seq2seq,
    args=seq2seq_args,
    train_dataset=train_seq2seq_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer_seq2seq,   # standard HF Trainer — tokenizer= still works here
)

print("Training flan-t5-small...")
trainer_seq2seq.train()
print("Training complete.")

model_seq2seq.save_pretrained("flant5_adapter")
tokenizer_seq2seq.save_pretrained("flant5_adapter")

import os
for fname in os.listdir("flant5_adapter"):
    upload_to_drive(f"flant5_adapter/{fname}", FLANT5_ADAPTER_ID)

print("flan-t5 adapter saved to Drive.")
```

**If OOM occurs** (unlikely): drop `per_device_train_batch_size` to 2.

---

## Session Recovery

If Colab disconnected between SmolLM2 and flan-t5 training, re-run these cells before Cell 4:
- Cell with `pydrive2` auth (restores `drive`)
- Cell with `upload_to_drive` / `create_drive_folder` (restores `FLANT5_ADAPTER_ID`)
- Cell with `from transformers import ...` (restores imports)
- Cell 1 above (restores `bnb_config`, `train_seq2seq_ds`)
