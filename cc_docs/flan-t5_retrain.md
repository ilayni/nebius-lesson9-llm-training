# Remaining Steps to Complete

## Current state
- ✅ os._exit(0) cells deleted
- ✅ Step C ran — 1,680 clean records written and synced to Drive
- ✅ FLAN-T5 retrained on 1,680 records
- ❌ SmolLM2 retraining cell NOT re-run — still shows old 1,740/654 output
- ❌ Two NameError tracebacks visible in evaluation cells

---

## Step 1 — Re-run SmolLM2 setup cells

Run these two cells in the Task 1.2 area:

**Cell 1** — starts with:
```python
import torch
from transformers import BitsAndBytesConfig
...
TRAIN_CAUSAL, TRAIN_SEQ2SEQ = "train_causal.jsonl", "train_seq2seq.jsonl"
```
Expected: `✅ Datasets loaded successfully into memory. QLoRA configuration prepared.`

**Cell 2** — starts with:
```python
model_id_causal = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer_causal = AutoTokenizer.from_pretrained(model_id_causal)
```
Expected: `trainable params: 3,276,800 || ... trainable%: 0.8975`

**SKIP the next cell** — the one with `smollm2_config = SFTConfig(...)` and `trainer_causal.train()`.

---

## Step 2 — Define smollm2_config and retrain SmolLM2

Insert a new cell **right before** the "SmolLM2 Retraining with Verified Data" cell and paste this in:

```python
from trl import SFTConfig

smollm2_config = SFTConfig(
    output_dir="./smollm2_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    max_length=512,
    dataset_text_field="text",
)
print("smollm2_config ready.")
```

Run it, then run the "SmolLM2 Retraining with Verified Data" cell below it.

Expected output:
```
🚀 Dataset Ready: 1680 records found. Starting training...
[630/630 ..., Epoch 3/3]
```
Takes ~13 min.

---

## Step 3 — Run SmolLM2 evaluation

Find the SmolLM2 eval cell in the Task 1.3 area — it defines `generate_smollm_response` and produces `smollm_df`. Use the **later** version (the one with `PeftModel.from_pretrained` that reloads the adapter from disk).

Running this clears the two NameError tracebacks and generates fresh results. Takes ~2–3 min.

---

## Step 4 — Run the Final Evaluation Results cells

Run the "Final Evaluation Results" summary cells that combine both model tables.

If you get `flant5_df not defined`, run the FLAN-T5 eval cell first (the one with `get_t5_ans`). Otherwise skip it — FLAN-T5 eval already ran correctly and its results are saved to `flant5_eval.csv` on Drive.

---

## Step 5 — Write Tasks 1.4 and 1.5

See `cc_docs/full_asgmnt_feedback.md` for the paste-ready text. Before pasting, check:
- Update "1,740 examples" → "1,680 examples" everywhere.
- Confirm SmolLM2 still hits 20/20 prefix accuracy in the new eval table (adjust if lower).
- Confirm FLAN-T5 still shows 0/20 changed (adjust if different).

---

## Final save

File → Save → File → Download .ipynb
