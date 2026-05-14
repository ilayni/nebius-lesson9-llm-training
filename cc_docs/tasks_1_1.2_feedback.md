# Tasks 1.1 & 1.2 — Grade Feedback

## Task 1.1 — Prepare a Custom Dataset (15 points)

### ✅ Fixed
- Test set (`test_causal.jsonl`) now displays all 20 rows with a verification message.
- Training set sample (`train_causal.jsonl`, `train_seq2seq.jsonl`) displays via `head()` with visible cell output.

### ❌ Still broken

**`test_seq2seq.jsonl` is never displayed**
The submission requirement states all test sets must be presented as DataFrame tables. The notebook displays `test_causal.jsonl` (all 20 rows) but never loads or displays `test_seq2seq.jsonl`.

Fix — add after the existing `show_file` calls:
```python
show_file("test_seq2seq.jsonl", "Full Test Set (Seq2Seq — Grading Requirement: 20 Rows)", head_only=False)
```

**Training datasets contain only 60 records each**
Both `train_causal.jsonl` and `train_seq2seq.jsonl` have only 60 records (visible in cell output: `Total records in train_causal.jsonl: 60`, `Total records in train_seq2seq.jsonl: 60`). The expected size is 1719 records (580 questions × 3 levels). The Drive still contains the old small files from an earlier test run.

Fix — regenerate and re-upload both training files:
1. Delete the cached `train_causal.jsonl` and `train_seq2seq.jsonl` from Drive.
2. Re-run the dataset generation cell with the full `filtered` dataset (580 questions).
3. Verify `Total records in train_causal.jsonl: 1719` in cell output before proceeding.

---

## Task 1.2 — Fine-Tune Models with QLoRA (10 points)

### ✅ Fixed
- Flan-T5 trained successfully: 12 steps, 3 epochs, adapter saved and synced to Drive.

### ❌ Still broken

**SmolLM2 trained on only 60 records**
Cell output shows `Tokenizing train dataset: 0/60` and `train_loss: 2.147`. Root cause is the same as Task 1.1 — the Drive had a 60-record file and Colab loaded that instead of the full dataset.

Fix — after fixing Task 1.1 (re-uploading the 1719-record files), delete the locally cached `train_causal.jsonl`, re-run the data loading cell, and retrain SmolLM2. Verify `Tokenizing train dataset: 0/1719` in cell output.

**Flan-T5 also trained on only 60 records**
`train_seq2seq.jsonl` also has 60 records. The same re-upload fix above applies — retrain flan-t5 after the full `train_seq2seq.jsonl` is restored.
