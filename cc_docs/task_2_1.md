# Task 2.1 — Scaled Dot-Product Attention

Paste this as the code cell for Task 2.1, replacing the existing TODO cell.

```python
import torch
import torch.nn.functional as F
import math
import json
import os

# ── Task 2.1 — Scaled Dot-Product Attention ──────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implements Scaled Dot-Product Attention.

    Args:
        Q: (batch, heads, seq_len_q, d_k)
        K: (batch, heads, seq_len_k, d_k)
        V: (batch, heads, seq_len_v, d_v)
        mask: (batch, heads, seq_len_q, seq_len_k) or None

    Returns:
        output: (batch, heads, seq_len_q, d_v)
        attention_weights: (batch, heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # Dot product between Q and K^T, then scale
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Mask out invalid positions (set to -inf so softmax → 0)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over key dimension to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# ── Verification ──────────────────────────────────────────────────────────────

def _verify():
    batch, heads, seq_len, d_k = 2, 4, 6, 16
    Q = torch.randn(batch, heads, seq_len, d_k)
    K = torch.randn(batch, heads, seq_len, d_k)
    V = torch.randn(batch, heads, seq_len, d_k)

    out, w = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (batch, heads, seq_len, d_k), f"Bad output shape: {out.shape}"
    assert w.shape == (batch, heads, seq_len, seq_len), f"Bad weights shape: {w.shape}"
    assert torch.allclose(w.sum(dim=-1), torch.ones(batch, heads, seq_len), atol=1e-5), \
        "Weights don't sum to 1"

    # Causal mask test — future positions must get zero weight
    mask = torch.ones(batch, heads, seq_len, seq_len).tril()
    _, w_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert torch.allclose(w_masked[..., 0, 1:], torch.zeros(batch, heads, seq_len - 1), atol=1e-5), \
        "Causal mask not applied correctly"

    return {
        "output_shape": list(out.shape),
        "weights_shape": list(w.shape),
        "weights_sum_mean": round(float(w.sum(dim=-1).mean()), 6),
        "mask_test": "passed"
    }

results = _verify()
print("✅ scaled_dot_product_attention verified:")
print(f"   Output shape:       {results['output_shape']}")
print(f"   Weights shape:      {results['weights_shape']}")
print(f"   Weights sum (mean): {results['weights_sum_mean']}")
print(f"   Causal mask:        {results['mask_test']}")


# ── Save checkpoint to Drive ──────────────────────────────────────────────────

checkpoint = {"task": "2.1", "status": "verified", **results}
with open("task2_checkpoint.json", "w") as f:
    json.dump(checkpoint, f)

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

    q = f"'{FOLDER_ID}' in parents and title = 'task2_checkpoint.json' and trashed = false"
    fl = drive.ListFile({"q": q}).GetList()
    f_drive = fl[0] if fl else drive.CreateFile({"title": "task2_checkpoint.json", "parents": [{"id": FOLDER_ID}]})
    f_drive.SetContentFile("task2_checkpoint.json")
    f_drive.Upload()
    print("✅ Checkpoint saved to Drive.")
except Exception as e:
    print(f"⚠️  Drive save skipped: {e}")
```
