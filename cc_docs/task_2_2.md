# Task 2.2 — Visualize Attention

Paste this as the code cell for Task 2.2, replacing the existing TODO cell.

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ── Task 2.2 — Visualize Attention Heatmap ───────────────────────────────────

torch.manual_seed(42)  # reproducible embeddings

sentence = "the cat sat on the mat"

tokens = sentence.split()
seq_len = len(tokens)

# Dummy embeddings — shape (seq_len, d_model)
d_model = 16
embeddings = torch.randn(seq_len, d_model)

# scaled_dot_product_attention expects (batch, heads, seq_len, d_k)
# Add batch and head dimensions: (1, 1, seq_len, d_model)
Q = embeddings.unsqueeze(0).unsqueeze(0)
K = embeddings.unsqueeze(0).unsqueeze(0)
V = embeddings.unsqueeze(0).unsqueeze(0)

_, attention_weights = scaled_dot_product_attention(Q, K, V)

# Remove batch and head dims for plotting: (seq_len, seq_len)
weights_2d = attention_weights.squeeze(0).squeeze(0).detach().numpy()

# ── Plot ──────────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 6))
sns.heatmap(
    weights_2d,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="viridis",
    annot=True,
    fmt=".2f",
    linewidths=0.5
)
plt.title("Attention Weights Heatmap — \"the cat sat on the mat\"")
plt.xlabel("Key (attended to)")
plt.ylabel("Query (attending from)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("task2_2_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Heatmap saved locally.")

# ── Save to Drive ─────────────────────────────────────────────────────────────

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

    q = f"'{FOLDER_ID}' in parents and title = 'task2_2_heatmap.png' and trashed = false"
    fl = drive.ListFile({"q": q}).GetList()
    f_drive = fl[0] if fl else drive.CreateFile({"title": "task2_2_heatmap.png", "parents": [{"id": FOLDER_ID}]})
    f_drive.SetContentFile("task2_2_heatmap.png")
    f_drive.Upload()
    print("✅ Heatmap saved to Drive.")
except Exception as e:
    print(f"⚠️  Drive save skipped: {e}")
```
