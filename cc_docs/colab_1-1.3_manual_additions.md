# Manual Additions for Task 1.3 — Current Run

These are ready to paste directly into Colab cells based on the outputs already in the notebook.
When you retrain on the full dataset, update me with the new outputs and I'll regenerate.

---

## SmolLM2 — Add this cell after the existing eval table

Replaces the current `display(eval_df)` line. Paste this as a new code cell right after the SmolLM2 eval cell:

```python
# Add required columns manually based on current outputs
better_ft = ["No"] * 20  # Base = Fine-tuned for all rows (adapter undertrained on 60 records)

notes = [
    "Child level not adapted — uses technical language ('standard unit of length'); does not answer 5280 feet",
    "Student level partial — somewhat relevant but full output unclear; no educational framing",
    "Expert level not adapted — factually questionable (12 tribes is too low); no technical depth",
    "Child level not adapted — correct answer but no simplification for a child audience",
    "Student level not adapted — one-sentence answer ('The best sport is basketball'); no educational context",
    "Expert level partial — on-topic and factually reasonable; lacks technical precision",
    "Child level not adapted — relevant but not simplified; reads like a neutral description",
    "HALLUCINATION — output is about job interview preparation, completely unrelated to Tableau",
    "Expert level not adapted — generic tourist-style answer; no technical sailing/physics content",
    "Child level not adapted — relevant but no child-friendly simplification",
    "Student level partial — relevant and fairly clear; closest to meeting the level requirement",
    "Answer truncated mid-sentence ('A stock index is a statistical'); incomplete output",
    "HALLUCINATION — states 'Women have beards because they are a symbol of…', factually inverts the question",
    "Student level partial — relevant answer but no clear educational framing",
    "HALLUCINATION — output is about Washington D.C., completely unrelated to India cricket wins",
    "Child level partial — correct and reasonably simple; acceptable",
    "Student level partial — relevant and plausible; adequate",
    "Expert level not adapted — describes pineapple generically; no botanical or scientific depth",
    "Child level partial — relevant but not simplified; reads like a neutral explanation",
    "Answer truncated — output is '1. Brush your teeth', barely begins the answer",
]

eval_df["Better after FT?"] = better_ft
eval_df["Notes"] = notes

# Reorder columns to match assignment requirement
eval_df = eval_df[["Question", "Level", "Base Output", "Fine-tuned Output", "Expected Answer", "Better after FT?", "Notes"]]
display(eval_df)
```

---

## SmolLM2 — Add this markdown cell after the table

```
**SmolLM2 Evaluation Analysis**

- **Level adaptation**: The model does not meaningfully adapt its answers to the requested expertise level. Child, student, and expert prompts receive similarly structured responses with no clear simplification or technical deepening.
- **Answer clarity**: Most outputs are readable and on-topic, but several are truncated (row 11, row 19) or only answer the surface question without depth.
- **Faithfulness**: The majority of answers are loosely related to the question, but 3 rows contain clear hallucinations — row 7 answers a job interview question instead of explaining Tableau, row 12 inverts the question about beards, and row 14 answers with Washington D.C. instead of India cricket.
- **Fine-tuning effect**: Base Output and Fine-tuned Output are identical for all 20 rows. The adapter trained on only 60 records with a high training loss (2.147) learned almost nothing, so the PEFT adapter has no observable effect on generation.
- **Overall**: SmolLM2 shows some baseline instruction-following ability from its pre-training, but fine-tuning failed to improve it due to insufficient training data.
```

---

## Flan-T5 — Add this cell after the existing eval table

Paste this as a new code cell right after the Flan-T5 eval cell:

```python
# Add required columns manually based on current outputs
better_ft_t5 = ["No"] * 20  # Base = Fine-tuned for all rows (adapter undertrained on 60 records)

notes_t5 = [
    "Wrong answer ('a mile' instead of 5280 feet); extremely short; no child-level explanation",
    "Nonsensical — 'a student' has no relation to Project Mosaic",
    "Wrong — '3' is not the number of Alaska Native tribes; no expert framing",
    "Grammatically broken — 'food is a food' is not an answer",
    "Wrong — 'acrobatics' is unrelated to the question about best sport",
    "Wrong — 'guru' is not an explanation of karma in Buddhism",
    "Wrong — 'scrimmage' is a football term, not an explanation of the difference",
    "Nonsensical — 'a student' has no relation to Tableau",
    "Wrong — 'a person who is a novice' does not answer why sailing is fun",
    "Non-answer — echoes the topic word 'cricket' without explanation",
    "Non-answer — echoes the topic 'weed wacker' without explanation",
    "Partially relevant — 'equities' is related to stocks but is not an explanation",
    "Inappropriate and wrong — 'sex' is not an explanation of beard biology",
    "Incomplete — 'a dollar' is partially relevant but not a full answer",
    "Wrong — 'batting' is a cricket term, unrelated to counting India's World Cup wins",
    "Inappropriate and wrong — 'sex' is unrelated to Romance languages",
    "Wrong — 'acrobatics' is unrelated to the Gun Show in New Zealand",
    "Non-answer — echoes the topic word 'pineapple' without explanation",
    "Partial — 'a dog' is one example of a pet but does not explain the pet vs cattle difference",
    "Nonsensical — 'is a good student' is not about dental hygiene",
]

results_t5_df = pd.DataFrame(results)
results_t5_df["Better after FT?"] = better_ft_t5
results_t5_df["Notes"] = notes_t5

# Reorder columns to match assignment requirement
results_t5_df = results_t5_df[["Question", "Level", "Base Output", "Fine-tuned Output", "Expected", "Better after FT?", "Notes"]]
results_t5_df = results_t5_df.rename(columns={"Expected": "Expected Answer"})
display(results_t5_df)
```

---

## Flan-T5 — Add this markdown cell after the table

```
**Flan-T5 Evaluation Analysis**

- **Level adaptation**: Flan-T5 shows no level adaptation. Outputs are single words or short fragments with no variation between child, student, and expert prompts.
- **Answer clarity**: Nearly all outputs are too short to be useful — single words, topic echoes, or grammatically broken fragments. The model is not generating coherent answers.
- **Faithfulness**: Most outputs are either non-answers (echoing the topic word) or factually wrong. Several are inappropriate (rows 12, 15 return 'sex' as the answer).
- **Fine-tuning effect**: Base Output and Fine-tuned Output are identical for all 20 rows. Same root cause as SmolLM2 — the adapter was trained on only 60 records and did not learn.
- **Overall**: Flan-T5-small trained on 60 records completely fails at this task. The base model alone already performs poorly on open-ended generation, and the fine-tuning had no measurable effect. Retraining on the full 1719-record dataset is required to produce meaningful results.
```
