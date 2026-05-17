# Learning Takeaways — Lesson 9: Fine-Tuning & Attention

This assignment had two big themes: **how to adapt a pretrained LLM cheaply (QLoRA)** and **how the attention mechanism actually works under the hood**.

---

## Part 1 — QLoRA Fine-Tuning

### Core idea
Full fine-tuning of a modern LLM updates **all** the weights. Even a small 360M model needs gigabytes of GPU memory for gradients and optimizer state. **QLoRA = Quantized LoRA** sidesteps that:

1. **4-bit quantization (NF4)** — the frozen base model's weights are stored in 4 bits instead of 16/32, cutting model size by ~4×.
2. **LoRA adapters** — instead of updating the giant base weight matrices, you add tiny low-rank matrices (`A ∈ R^(d×r)`, `B ∈ R^(r×d)`) on top of attention/projection layers, where `r` is small (8, 16, 32...). Only `A` and `B` are trained.
3. **Result:** in your runs only ~0.9% of parameters were trainable, yet the model still learned a new behaviour.

### The rank `r` tradeoff
- Higher `r` → more adapter capacity → can fit more complex patterns, but more memory and more overfitting risk on small datasets.
- Lower `r` → faster, less memory, but may underfit.
- You used `r=16` for SmolLM2 and `r=8` for FLAN-T5 — sensible for ~1,700 examples.

### When to fine-tune vs prompt-engineer
- **Prompt engineering** is enough when the base model already understands the task — just steer it.
- **Fine-tuning** is necessary when you need consistent format, domain adaptation, or want to remove long system prompts at inference time.
- Tradeoffs: fine-tuning bakes behaviour into the weights (more consistent, but less runtime steering); prompts are flexible (just edit the text) but inconsistent.

### Architecture matters more than data size
The most striking result in your assignment: **SmolLM2 succeeded, FLAN-T5 failed**, despite both seeing the same 1,680 training examples.

- **SmolLM2** = decoder-only causal LM, pretrained for open-ended generation. Asking it to produce multi-sentence level-adapted explanations matches its pretraining objective.
- **FLAN-T5** = encoder-decoder, pretrained on short structured tasks (translation, classification, Q&A). Asking it to generate paragraph-length explanations is outside its training distribution.

**Lesson:** more data doesn't fix a model/task mismatch. Choose the architecture for the task first.

### Data format matters per architecture
- **Causal models** (SmolLM2): one continuous text field with delimiters (`### Question:`, `### Answer:`).
- **Seq2seq models** (FLAN-T5): separate `input` and `target` fields, fed to encoder and decoder respectively.

Mixing these up = the model trains on something completely wrong.

---

## Part 2 — Attention Mechanism

### Scaled dot-product attention
The formula:
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

Intuition:
- **Q** (query): "what am I looking for?"
- **K** (key): "what do I represent?"
- **V** (value): "what information do I carry?"
- `QKᵀ` measures how well each query matches each key (dot product = similarity).
- Divide by `√d_k` so the softmax doesn't saturate for large embedding dimensions.
- Softmax over the last dim gives a probability distribution — each query "spends" its attention budget across all keys.
- Multiply by `V` to mix the values weighted by attention.

### Why softmax sums to 1
That's what makes attention a **weighted average** — every token's output is a convex combination of all the values. This is why your verification test `weights.sum(dim=-1) == 1.0` matters.

### Causal masking
For autoregressive generation, future tokens shouldn't leak into past ones. Trick: set future positions to `-∞` **before** softmax — softmax of `-∞` is 0, so future tokens get zero weight.

### Multi-head attention — why?
A single attention head produces one distribution per query. Multiple heads, each with its own `W_Q`, `W_K`, `W_V` projections, give you **multiple parallel views** of the same tokens. In your Experiment 2 you saw the two heads attended very differently to the same sentence.

In real transformers different heads end up specializing — one might track syntactic agreement, another nearby tokens, another long-range coreferences — and the outputs are concatenated and projected through `W_O` to combine them.

### Why your random-embedding heatmaps were uniform
With random vectors, there's no semantic structure → dot products are roughly equal → softmax distributes attention nearly uniformly (~1/seq_len per cell). You also noticed the two `"the"` tokens had identical rows/columns — because identical words produced identical embeddings, so their attention behaviour was identical too. In a trained model, learned embeddings + position encodings would give the heatmap real structure.

---

## Practical / Process Lessons (from the bumps you hit)

These aren't in the assignment text but are worth remembering:

### 1. Train/test leakage is silent until you check
Your test set was drawn from the same `filtered` pool the training set looped over — so the model trained on its own test questions. Always **partition first, then generate records**. The fix was a one-line change (`shuffled[20:]` for training), but it required retraining both models.

**Rule of thumb:** before reporting any eval metric, ask "could this row have appeared in training?" and verify with a set intersection if unsure.

### 2. bfloat16 vs fp16 — pick one, consistently
You hit `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'` because the base model was loaded with `bnb_4bit_compute_dtype=torch.bfloat16` but the trainer was configured with `fp16=True`. They conflict.

**Rule:** if you load the model in bfloat16, set `bf16=True, fp16=False` in the training args.

### 3. "Better after FT?" measuring string inequality is misleading
Your FLAN-T5 table showed 18/20 "Yes" because the strings differed — but they just swapped one bad token (`"a mile"`) for another (`"s"`). A simple boolean "differs" check can give false confidence. For real eval, always pair the boolean with a manual quality column (which you did for FLAN-T5 — that's why those notes were valuable).

### 4. Templated test answers ≠ hand-crafted test answers
The instruction said "manually reviewed for quality." Wrapping the original Dolly response in `"Imagine this: [response]. It's very simple!"` is not the same as writing a child-appropriate explanation. For a real eval set you'd write each test answer yourself for the target level.

### 5. Keep the notebook story clean
Old training runs with stale outputs (60 records, 1,740 records, multiple FLAN-T5 setups) create internal contradictions that graders and reviewers will catch. Either re-run cells from scratch or delete the dead ones — don't ship a notebook where Task 1.4 says one thing and the cell output above says another.

### 6. Sync to Drive but verify what's there
Multiple times during the assignment, a "sync" cell overwrote a file you'd just fixed with a stale version from Drive. **Local truth vs. Drive truth** can diverge fast in Colab. Print the record count after every regen so you catch divergences immediately.

---

## What you can now actually do

- Take any decoder-only HF model under ~1B params and fine-tune it on a custom instruction-style dataset in a free Colab GPU using QLoRA.
- Choose the right architecture (causal vs seq2seq) for a given generation task and explain why.
- Format datasets correctly for each architecture.
- Implement scaled dot-product attention from scratch, including causal masking.
- Read an attention heatmap and explain what the patterns mean.
- Diagnose common fine-tuning failure modes: data leakage, dtype mismatches, base/target mismatch.

---

## Useful references for going further

- **QLoRA paper:** Dettmers et al., 2023 — "QLoRA: Efficient Finetuning of Quantized LLMs"
- **LoRA paper:** Hu et al., 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"
- **Attention paper:** Vaswani et al., 2017 — "Attention Is All You Need"
- **PEFT library docs:** https://huggingface.co/docs/peft — actively maintained, covers LoRA/QLoRA/IA3/prompt tuning
- **The Annotated Transformer** (Harvard NLP) — line-by-line walkthrough of Vaswani et al. with PyTorch code
