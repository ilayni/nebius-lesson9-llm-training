# Tasks 1.4 and 1.5 — Manual Answer Additions

Add a markdown cell after each task heading with the content below.

---

## Task 1.4 — Model Comparison and Discussion

```markdown
**SmolLM2** adapted significantly better to the requested expertise levels. After fine-tuning on 1,740 records, it consistently generated the correct stylistic prefixes ("Imagine this:" for child, "Educational Explanation:" for student, "Technical Analysis:" for expert) in all 20 test rows. **Flan-T5** showed no meaningful adaptation — outputs were single words or short fragments (e.g. "a mile", "guru", "scrimmage") with no level differentiation.

**Clarity:** SmolLM2 produced coherent, paragraph-length answers. Flan-T5 outputs were too short to be useful answers.

**Format adherence:** SmolLM2 achieved 100% format adherence after fine-tuning. Flan-T5 failed entirely at the generative format — it often echoed the level keyword ("a student") rather than producing an answer.

**Hallucinations:** The base SmolLM2 model hallucinated frequently (e.g. row 4 base output discusses Paris; row 14 mentions a painting). Fine-tuning significantly reduced this. Flan-T5 did not hallucinate in the traditional sense but produced irrelevant tokens.

**Architectural explanation:** SmolLM2 is a decoder-only causal LM optimized for instruction-following and open-ended generation. This makes it well-suited for learning new output styles through fine-tuning. Flan-T5 is an encoder-decoder model pretrained on short structured tasks (translation, classification). Asking it to generate multi-sentence explanations at different expertise levels is outside its training distribution, and its small size (80M parameters) limits what 60 training examples can teach it.
```

---

## Task 1.5 — Conceptual Questions

```markdown
**1. What changed after fine-tuning?**

SmolLM2 transitioned from a general instruction-follower to a specialized model that enforces the expertise-level format. The base model produced generic answers with no style differentiation. After QLoRA fine-tuning on 1,740 examples, it adopted the three distinct output styles ("Imagine this:", "Educational Explanation:", "Technical Analysis:") consistently. Content accuracy improved but remains imperfect — the model learned format better than factual depth.

**2. Why is QLoRA more memory efficient?**

QLoRA combines two techniques:
- **4-bit NF4 quantization** compresses the frozen base model weights from 16/32-bit to 4-bit, reducing VRAM footprint by ~4×.
- **LoRA adapters** add small trainable low-rank matrices (rank `r`) to attention layers. Only these tiny matrices are updated during training — the base model stays frozen. This means gradients and optimizer states are computed only for the adapter parameters (~0.9% of total), not the full model.

Together they allow fine-tuning a 360M parameter model on a free Colab GPU.

**3. What happens if you increase the LoRA rank?**

A higher rank `r` increases the number of trainable parameters, giving the model more capacity to learn. However:
- More parameters → higher risk of **overfitting** on small datasets
- More parameters → higher **GPU memory** usage
- For this assignment (1,740 examples, small model), `r=16` for SmolLM2 and `r=8` for Flan-T5 were appropriate. Increasing to `r=64` would likely overfit with no quality gain.

**4. Why use QLoRA instead of prompt engineering?**

- **Prompt engineering** is sufficient when the base model already understands the task and only needs steering (e.g. "explain this to a 5-year-old"). It requires no training but can be brittle and inconsistent across inputs.
- **Fine-tuning (QLoRA)** is necessary when you need to enforce a strict output format across all inputs, adapt to a domain the model hasn't seen, or reduce inference costs by removing long system prompts.
- **QLoRA advantages over prompting:** consistent output format, no per-request prompt overhead, and the model "internalizes" the style rather than inferring it each time.
- **Trade-offs:** QLoRA requires a labeled dataset and a training phase. It is less flexible than prompting if requirements change — you would need to retrain rather than edit a prompt.
```
