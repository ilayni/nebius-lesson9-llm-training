# Assignment Grade — Ilay Nisim & Boris Kogan

**Notebook:** `From_Finetuning_to_Attention_Inside_LLMs_colab_ilay_nisim_boris_kogan.ipynb`

---

## Score Summary

| Task | Points Available | Estimated Score | Change vs. prior review |
|---|---|---|---|
| 1.1 — Prepare Dataset | 15 | 12 | — |
| 1.2 — Fine-Tune with QLoRA | 10 | 9 | ↑ +1 (cleanup applied) |
| 1.3 — Evaluation | 20 | 14 | — |
| 1.4 — Model Comparison | 15 | 13 | ↑ +3 (contradiction fixed) |
| 1.5 — Conceptual Questions | 15 | 13 | — |
| 2.1 — Attention Implementation | 5 | 5 | — |
| 2.2 — Visualize Attention | 5 | 5 | — |
| 2.3 — Experiments | 15 | 14 | — |
| **Total** | **100** | **~85** | **↑ +4** |

---

## Task 1.1 — Prepare a Custom Dataset (12/15)

**What's done well:**
- Both causal and seq2seq formats correctly separated.
- 1,680 training records and 20-row test set present as DataFrame tables.
- Test set covers all three expertise levels with seed-42 shuffle, correctly partitioned from training.
- Files saved as `.jsonl` and synced to Drive.

**Deductions:**
- **Test answers are templated, not hand-crafted.** Every answer follows a rigid template (`"Imagine this: [raw Dolly response]. It's very simple!"`) — the assignment says the test set must be "manually reviewed for quality." A grader will see the "expected answers" are just the original Dolly responses wrapped in a prefix.
- **No visible smoke test** as recommended in the instructions.

---

## Task 1.2 — Fine-Tune Models with QLoRA (9/10)

**What's done well:**
- Both models loaded in 4-bit (NF4, bfloat16) ✅
- LoRA adapters attached: SmolLM2 r=16, FLAN-T5 r=8 ✅
- Only adapter parameters trained (0.9% trainable for SmolLM2) ✅
- Both adapters synced to Drive ✅
- SmolLM2 retrained on 1,680 records (output: `🚀 Dataset Ready: 1680 records found`) ✅
- FLAN-T5 retrained on 1,680 records (output: `Map: 0/1680`) ✅
- **Cleanup applied:** the three stale training cells (60-record run, 1,740-record run, redundant FLAN-T5 setup) were removed. Task 1.2 now tells a clean story ✅

**Deductions:**
- The SmolLM2 retraining cell lives in a separate "SmolLM2 Retraining with Verified Data" section that appears after Task 1.3 — a grader has to scroll back to confirm the model was actually trained. Minor structural quirk.

---

## Task 1.3 — Evaluation (14/20)

**What's done well:**
- Both models evaluated on the 20-row test set ✅
- Columns: Question, Level, Base Output, Fine-tuned Output, Expected Answer, Better after FT? ✅
- SmolLM2 fine-tuned outputs consistently show correct prefixes (all 20 rows) ✅
- FLAN-T5 manual evaluation notes table is detailed and well-written ✅
- Final combined evaluation table renders both models ✅

**Deductions (unfixed from prior review):**

1. **SmolLM2 Notes column is empty.** The DataFrame creates a `Notes` column then assigns it the empty string for all 20 rows. The FLAN-T5 table has thorough per-row notes; SmolLM2 has none. This is a direct omission against the assignment's "manually reviewed" requirement.

2. **FLAN-T5 analysis still contains internal contradictions:**
   - Says *"0 out of 20 rows changed after fine-tuning"* — but the table directly above shows 18/20 rows marked "Yes" in Better after FT?.
   - Says *"(only 12 training steps)"* — but the FLAN-T5 training output above clearly shows 315 steps on 1,680 records.
   - Both numbers are leftovers from the original 60-record run that wasn't updated when the model was retrained.

   These contradictions are easy to spot and weaken the credibility of the evaluation. Quick fix: change "0 out of 20" to "18 out of 20 (but in name only — outputs went from one bad token to another)" and remove the "12 training steps" parenthetical.

---

## Task 1.4 — Model Comparison and Discussion (13/15)

**What's done well:**
- Natural student voice throughout ✅
- References specific examples from the eval table ✅
- Architectural explanation (decoder-only vs encoder-decoder) is correct ✅
- Covers all four comparison dimensions ✅
- **Fixed:** the "0/20 rows changed" contradiction is gone — now correctly describes 18/20 outputs changed but not usefully ✅
- Correctly cites 1,680 record count ✅

**Deductions:**
- **Small grammar/flow issue.** The fixed sentence reads:

  > *"No output resembled a real answer at either level of expertise.even after training on 1,680 examples."*

  Missing space before "even" and the merged sentence reads awkwardly. Suggested fix:

  > *"No output resembled a real answer at either level of expertise, even after training on 1,680 examples."*

- The Paris-as-Alaska-answer example referenced in the hallucination paragraph is a great anecdote, but if it doesn't appear cleanly in the visible eval table a grader may not be able to verify it.

---

## Task 1.5 — Conceptual Questions (13/15)

**What's done well:**
- All four questions answered in natural student voice ✅
- Q1: Good distinction between format adaptation and content depth ✅
- Q2: Correct explanation, cites 0.9% trainable figure ✅
- Q3: Correctly identifies overfitting risk and memory tradeoff ✅
- Q4: Covers when prompting suffices, when fine-tuning is needed, advantages ✅

**Deductions:**
- **Q4 doesn't explicitly cover "control" as a tradeoff dimension.** The prompt asks about cost, flexibility, AND control. Cost and flexibility are addressed; control (e.g., fine-tuning lets you bake in behaviour but you give up runtime steering you get with prompts) is missing.

---

## Task 2.1 — Implement Scaled Dot-Product Attention (5/5)

**Perfect.**
- `scores = QK^T / sqrt(d_k)` ✅
- `softmax(scores, dim=-1)` ✅
- `output = weights @ V` ✅
- Mask correctly applies `-inf` before softmax ✅
- Verification passes (shapes, weights sum to 1.0, causal mask test) ✅

---

## Task 2.2 — Visualize Attention (5/5)

**Perfect.**
- Heatmap rendered ✅
- Axes labeled (Query / Key) ✅
- Written explanation correctly notes uniform distribution with random embeddings and identical rows/columns for repeated "the" tokens ✅

---

## Task 2.3 — Experiments (14/15)

**What's done well:**
- Experiment 1: both heatmaps rendered side-by-side ✅
- Experiment 2: two heads with independent W_Q/W_K/W_V matrices ✅
- All three questions answered ✅
- Hash-based deterministic embeddings is a smart implementation choice ✅

**Minor deduction:**
- Q3 could go one level deeper on the W_O output projection that integrates the head outputs after concatenation.

---

## Outstanding Items Before Submission

Quick fixes that would push the score higher:

1. **Task 1.4 grammar (~30 sec):** Fix `"expertise.even"` → `"expertise, even"`.
2. **Task 1.3 FLAN-T5 analysis (~1 min):** Change "0 out of 20 rows changed" → "18 out of 20 rows changed in name only — outputs swapped one bad token for another". Remove "(only 12 training steps)".
3. **Task 1.3 SmolLM2 Notes (~5 min):** Add at least brief notes for the 20 rows. Even short observations like "format prefix correct, content faithful" or "prefix correct but truncated" would address the missing column.
4. **Task 1.1 (optional, larger effort):** Add a one-sentence acknowledgement that test answers use a template based on Dolly responses rather than hand-crafted level-appropriate answers — shows awareness.

After 1+2+3 (about 7 minutes of work), expected score: **~88/100**.

---

## Overall Comments

**Strengths:** Part 2 is excellent — clean attention implementation, smart experiment design, well-written answers. Both models successfully retrained on clean 1,680-record dataset (no leakage). Task 1.2 is now visually clean after cleanup. Task 1.4 correctly explains the architectural differences. The FLAN-T5 manual notes table is detailed and honest.

**Main weakness:** Task 1.3 has two unresolved issues — empty SmolLM2 Notes column and a self-contradicting FLAN-T5 analysis paragraph. Both are quick to fix and worth the points.
