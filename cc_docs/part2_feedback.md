# Part 2 — Feedback (updated)

## Status

All previous issues are fixed:
- Stray closing ` ``` ` removed ✅
- Task 2.2 observation cell added ✅
- Answers rewritten to sound more human ✅
- All plots generated and saved to Drive ✅

---

## Remaining Issues

### 1. Missing "Answers" header before the Q&A (minor)

The three answers appear immediately after the Experiment 2 output with no section break. Add a short markdown cell between the experiment output and the answers:

```markdown
**Answers**
```

This makes the answers easier to find and signals to the grader where the written responses begin.

---

### 2. Experiment 1 is missing a comparison sentence (grading gap)

The task instruction says: *"5. Compare the two heatmaps and explain whether the attention pattern changed."*

Since both experiment plots are in one cell you can't insert a markdown cell between them. Instead, add an explicit comparison at the top of the Answers markdown cell, before Q1:

```markdown
**Experiment 1 — Comparison:** The two heatmaps are visibly different — the row and column for "cat" in the first plot does not match the row and column for "dog" in the second. Since "dog" has a completely different random embedding, its dot products with all other tokens shift, changing both how it attends to others and how others attend to it.
```

---

### 3. Answers are in blockquote format (cosmetic)

The answers use Markdown blockquote syntax (`>`), which renders as indented grey boxes. This looks slightly unusual for written answers. Not a grading risk, but you can remove the `>` prefix from each answer to make them plain paragraph text if you prefer.

---

## Overall

Part 2 is complete and correct. The code is clean, all cells ran without errors, plots are saved to Drive. The only remaining items are the two minor presentation fixes above (answers header and Experiment 1 comparison sentence) — both are one-line additions.
