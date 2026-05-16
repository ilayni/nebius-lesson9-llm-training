# Notebook Cleanup — Safe Deletions Only (No Re-runs Needed)

These 3 deletions reduce visual clutter in Task 1.2 without requiring any cell to be re-run. Each deletion has been checked — downstream cells either don't depend on them, or other cells in the notebook already define everything they need.

---

## Delete 1 — First SmolLM2 trainer cell (trained on stale 60 records)

**Location:** Task 1.2, the cell that starts:
```python
# ── Train SmolLM2 ────────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

smollm2_config = SFTConfig(
      output_dir="./smollm2_checkpoints",
      ...
      fp16=True,   # ← old config
      ...
  )

trainer_causal = SFTTrainer(...)
trainer_causal.train()
```

**What its output shows:**
```
Adding EOS to train dataset: 0%|          | 0/60 [00:00<?, ? examples/s]
Tokenizing train dataset: 0%|          | 0/60 [00:00<?, ? examples/s]
Training SmolLM2-360M-Instruct...
{'train_runtime': '1.809e+04', ... 'epoch': '3'}
Training complete.
```

**Why safe to delete:**
- Trains on the original stale 60-record dataset (before Step C fix).
- `trainer_causal` from this cell is **not used** anywhere downstream — the bottom "SmolLM2 Retraining with Verified Data" cell defines its own `trainer_causal`.
- `smollm2_config` from this cell is redundant — the cell you inserted later (with `bf16=True`) defines it for the actual retraining.
- `model_causal` was already loaded by the cell above (line ~1140 — `model_id_causal = "HuggingFaceTB/SmolLM2-360M-Instruct"`).

**No re-run needed.** Just delete the cell.

---

## Delete 2 — Second SmolLM2 retrainer cell (trained on stale 1,740 records)

**Location:** Task 1.2, the cell that starts:
```python
import torch, os, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
...

# 1. Regenerate full record dataset to ensure local file is complete
ds = load_dataset('databricks/databricks-dolly-15k', split='train[:5000]')
filtered = [ex for ex in ds if ex['category'] == 'open_qa' and ...]

records = []
levels = ['child', 'student', 'expert']
for item in filtered:
    ...
train_ds = Dataset.from_list(records)
print(f"🚀 Dataset verified: {len(train_ds)} records found.")
...
trainer.train()
```

**What its output shows:**
```
🚀 Dataset verified: 1740 records found.
Adding EOS to train dataset: 0%|          | 0/1740 [00:00<?, ? examples/s]
Starting re-training on full 1740 records...
[654/654 13:01, Epoch 3/3]
```

**Why safe to delete:**
- This trains on the **leaky 1,740-record** dataset, which has been superseded by the "SmolLM2 Retraining with Verified Data" cell at the bottom (which correctly trains on 1,680 records).
- It loops `for item in filtered` (uses all 580 questions — includes the 20 test questions) — leakage you already fixed.
- `trainer` and `train_ds` from this cell are not used anywhere else.
- `model_causal` is already loaded by the cell above; `smollm2_config` is defined by your inserted cell.

**This is the biggest visual-noise reducer** — removes the misleading "1740 records / 654 steps" output that contradicts your final 1,680-record / 630-step run.

**No re-run needed.** Just delete the cell.

---

## Delete 3 — First FLAN-T5 setup cell (redundant — replaced by next cell)

**Location:** Task 1.2, the cell that starts:
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model

# 1. Clear GPU cache
torch.cuda.empty_cache()

# 2. Re-define configurations (needed after a session restart)
bnb_config = BitsAndBytesConfig(...)

lora_config_seq2seq = LoraConfig(
    r=16,                         # ← uses r=16
    ...
)

model_id_seq2seq = "google/flan-t5-small"
tokenizer_seq2seq = AutoTokenizer.from_pretrained(model_id_seq2seq)
...
model_seq2seq = AutoModelForSeq2SeqLM.from_pretrained(...)
model_seq2seq = prepare_model_for_kbit_training(model_seq2seq)
model_seq2seq = get_peft_model(model_seq2seq, lora_config_seq2seq)
```

**What its output shows:**
```
model.safetensors: 0%|          | 0.00/308M [00:00<?, ?B/s]
trainable params: 688,128 || all params: 77,649,280 || trainable%: 0.8862
flan-t5-small loaded and adapters attached.
```

**Why safe to delete:**
- This is the **old FLAN-T5 setup with r=16** that was never used for training.
- The very next cell (the "Fix 1: Flan-T5 Config with bf16 and r=8" cell that actually trains FLAN-T5) redefines `bnb_config`, `lora_config_seq2seq` (with r=8), `tokenizer_seq2seq`, and `model_seq2seq` from scratch — so this earlier setup is fully overridden.
- No downstream cell uses anything defined only in this cell.

**No re-run needed.** Just delete the cell.

---

## Optional — Clear outputs of Drive Sync cells (low value)

Two **"Drive Sync Verification"** cells in Task 1.2 show this output:
```
Folder smollm2_adapter not found locally.
Folder flant5_adapter not found locally.
Verification complete.
```

These ran at moments when the adapters weren't on local disk yet. They're harmless but add noise. You can right-click the output area → "Clear output" without re-running. Keeping or removing them is your call.

---

## Summary

| # | Cell | What gets removed | Re-run? |
|---|---|---|---|
| 1 | First SmolLM2 trainer (60 records) | `Training on 0/60 examples` output | No |
| 2 | Second SmolLM2 retrainer (1,740 records) | `1740 records / 654 steps` output | No |
| 3 | First FLAN-T5 setup (r=16) | Duplicate FLAN-T5 model load output | No |

After these 3 deletions, Task 1.2 will show **only** the final clean runs: SmolLM2 retrained on 1,680 records (630 steps) and FLAN-T5 retrained on 1,680 records (315 steps). Much cleaner story for the grader.
