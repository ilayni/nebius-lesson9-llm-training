# Task 2.3 — Experiments

Two cells to paste: the code cell, then a markdown cell with the answers.

---

## Code Cell

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ── Helper: word-specific deterministic embeddings ───────────────────────────
# Same word → same vector; different word → different vector

def sentence_embeddings(sentence, d_model):
    tokens = sentence.split()
    embeddings = []
    for word in tokens:
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(word)) % (2**32)
        gen = torch.Generator()
        gen.manual_seed(seed)
        embeddings.append(torch.randn(d_model, generator=gen))
    return tokens, torch.stack(embeddings)  # (seq_len, d_model)


def plot_heatmap(ax, weights, tokens, title):
    sns.heatmap(
        weights, ax=ax,
        xticklabels=tokens, yticklabels=tokens,
        cmap="viridis", annot=True, fmt=".2f",
        linewidths=0.5, vmin=0, vmax=1
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)


def get_weights(embeddings):
    Q = embeddings.unsqueeze(0).unsqueeze(0)
    K = embeddings.unsqueeze(0).unsqueeze(0)
    V = embeddings.unsqueeze(0).unsqueeze(0)
    _, w = scaled_dot_product_attention(Q, K, V)
    return w.squeeze(0).squeeze(0).detach().numpy()


d_model = 16

# ── Experiment 1 — Change One Word ───────────────────────────────────────────

sent1 = "the cat sat on the mat"
sent2 = "the dog sat on the mat"

tokens1, emb1 = sentence_embeddings(sent1, d_model)
tokens2, emb2 = sentence_embeddings(sent2, d_model)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plot_heatmap(ax1, get_weights(emb1), tokens1, f'"{sent1}"')
plot_heatmap(ax2, get_weights(emb2), tokens2, f'"{sent2}"')
fig1.suptitle("Experiment 1 — Change One Word", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("task2_3_exp1.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Experiment 1 plotted.")

# ── Experiment 2 — Two Attention Heads ───────────────────────────────────────

tokens, embeddings = sentence_embeddings(sent1, d_model)
d_head = 8  # each head projects d_model → d_head

torch.manual_seed(0)
W_Q1 = torch.randn(d_model, d_head)
W_K1 = torch.randn(d_model, d_head)
W_V1 = torch.randn(d_model, d_head)

torch.manual_seed(1)
W_Q2 = torch.randn(d_model, d_head)
W_K2 = torch.randn(d_model, d_head)
W_V2 = torch.randn(d_model, d_head)

def head_weights(X, W_Q, W_K, W_V):
    Q = (X @ W_Q).unsqueeze(0).unsqueeze(0)
    K = (X @ W_K).unsqueeze(0).unsqueeze(0)
    V = (X @ W_V).unsqueeze(0).unsqueeze(0)
    _, w = scaled_dot_product_attention(Q, K, V)
    return w.squeeze(0).squeeze(0).detach().numpy()

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
plot_heatmap(ax3, head_weights(embeddings, W_Q1, W_K1, W_V1), tokens, "Head 1 (seed=0)")
plot_heatmap(ax4, head_weights(embeddings, W_Q2, W_K2, W_V2), tokens, "Head 2 (seed=1)")
fig2.suptitle('Experiment 2 — Two Attention Heads — "the cat sat on the mat"', fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("task2_3_exp2.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Experiment 2 plotted.")

# ── Save both figures to Drive ────────────────────────────────────────────────

try:
    if "drive" not in globals():
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from google.colab import auth, userdata
        from oauth2client.client import GoogleCredentials
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)
        FOLDER_ID = userdata.get("FOLDER_ID")

    for fname in ["task2_3_exp1.png", "task2_3_exp2.png"]:
        q = f"'{FOLDER_ID}' in parents and title = '{fname}' and trashed = false"
        fl = drive.ListFile({"q": q}).GetList()
        f_drive = fl[0] if fl else drive.CreateFile({"title": fname, "parents": [{"id": FOLDER_ID}]})
        f_drive.SetContentFile(fname)
        f_drive.Upload()
        print(f"✅ {fname} saved to Drive.")
except Exception as e:
    print(f"⚠️  Drive save skipped: {e}")
```

---

## Markdown Cell (paste after the code cell)

```markdown
**Answers**

**1. Did changing one word affect the attention weights? Why or why not?**

Yes. Replacing "cat" with "dog" changed the attention pattern because each word receives a unique embedding vector. Attention scores are computed as Q·Kᵀ/√d_k — dot products between embedding vectors. "Dog" has a different vector than "cat", so its dot products with all other tokens differ, producing a different softmax distribution. In a trained model the effect is more semantically meaningful, but even with random embeddings the geometry changes.

**2. Do different attention heads focus on different tokens?**

Yes. Each head applies different learned projection matrices (W_Q, W_K, W_V), projecting the same input into different subspaces. The dot products in those subspaces capture different geometric relationships between tokens, resulting in different attention distributions per head. The two heatmaps show visibly different patterns despite seeing the same sentence.

**3. Why might multi-head attention be useful in Transformer models?**

Multi-head attention lets the model attend to multiple types of relationships simultaneously. A single attention head can only capture one kind of dependency per layer. With multiple heads running in parallel, the model can learn different patterns at once — for example, one head for syntactic structure, another for semantic similarity, another for positional proximity. Their outputs are concatenated and linearly projected, giving the model a much richer contextual representation than a single head could provide.
```
