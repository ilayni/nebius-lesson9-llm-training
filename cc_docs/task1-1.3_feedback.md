# Task 1.3 — Evaluation Feedback (20 points)

## ✅ What works
- Both models (SmolLM2 and Flan-T5) produce output tables with all 20 test rows — no more errors
- Both tables include `Question`, `Level`, `Base Output`, `Fine-tuned Output`, and `Expected Answer` columns
- Adapter download from Drive works correctly before model loading

---

## ❌ Issues

### 1. Missing required columns — "Better after FT?" and "Notes"

The task explicitly specifies this table structure:

```
|Question|Level|Base Output|Fine-tuned Output|Expected Answer|Better after FT?|Notes|
```

Both tables stop at `Expected Answer`. Required additions:
- **Better after FT?** — a Yes / No / Partially per row
- **Notes** — a brief observation on quality, level adaptation, hallucination, faithfulness

### 2. Base Output = Fine-tuned Output for every single row (both models)

All 20 rows in both the SmolLM2 and Flan-T5 tables show identical `Base Output` and `Fine-tuned Output`. The comparison is not working.

Root cause: both models were trained on only 60 records with high loss (SmolLM2 train_loss: 2.147), so the adapters learned almost nothing. Fix requires retraining on the full 1719-record dataset (see Task 1.2 issue).

### 3. Flan-T5 outputs are near-meaningless — caused by fp16/bf16 mismatch

Base/fine-tuned outputs for Flan-T5 are single words or fragments: `"a mile"`, `"guru"`, `"sex"`, `"acrobatics"`, `"food is a food"`. This has **two causes**:

1. **fp16 instability (primary cause)**: Flan-T5 was pretrained in bf16. The notebook trains with `fp16=True` in `Seq2SeqTrainingArguments`, which causes numerical instability for this model family. Fix: `bf16=True, fp16=False`. This is a known issue confirmed by other students in the course.
2. **60-record training data** (secondary): also contributes, but fixing bf16 is the priority for T5.

Confirmed best config from class discussion: `LR=1e-4`, `rank=8`, `target_modules=["q", "v"]`, `bf16=True, fp16=False`.

### 4. No written analysis in the evaluation section

The task requires evaluating whether each answer:
- matches the requested expertise level
- is clear and relevant
- stays faithful to the question

There is no written commentary in the notebook for Task 1.3 — only the tables. A markdown analysis cell is required below each table.

---

## Summary

| Requirement | Status |
|---|---|
| Both models evaluated on 20 test rows | ✅ |
| Base Output column present | ✅ |
| Fine-tuned Output column present | ✅ |
| "Better after FT?" column | ❌ Missing |
| "Notes" column | ❌ Missing |
| Base ≠ Fine-tuned outputs | ❌ Identical for all rows |
| Written analysis of outputs | ❌ Missing |
