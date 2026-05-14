# Colab Setup Guide — Lesson 9: LLM Fine-Tuning

## Context

Task 1.2 (QLoRA fine-tuning) requires a GPU. `bitsandbytes` 4-bit quantization is CUDA-only and will not run on Mac or CPU. Use Google Colab with a T4 GPU — it has 16 GB VRAM, which is more than enough for SmolLM2-360M and flan-t5-small with 4-bit quant.

Course coordinator confirmed: **you must use a GPU runtime. Dropping quantization is not allowed.**

---

## Step 1 — Open Your Notebook in Colab

Your notebook folder on Drive:
`https://drive.google.com/drive/folders/1sJXbK8OsbJpMf6irIp5lImVbBTvXZPJg`

Open the notebook from Drive directly in Colab, or upload it via File → Upload notebook.

---

## Step 2 — Enable GPU Runtime

In Colab: **Runtime → Change runtime type → T4 GPU → Save**

Do this before running any cells. If you forget, Runtime → Disconnect and delete runtime, then reconnect with GPU.

---

## Step 3 — Mount Google Drive

Add this cell at the top of the notebook (after the install cell):

```python
from google.colab import drive
drive.mount("/content/drive")

import os

# Find your folder name after mounting
print(os.listdir("/content/drive/MyDrive"))
```

Look for the folder that contains your notebook (e.g. `lesson9_llm_training`). Then set:

```python
DRIVE_DIR    = "/content/drive/MyDrive/<your-folder-name>/"
ADAPTER_DIR  = DRIVE_DIR + "adapters/"
os.makedirs(ADAPTER_DIR + "smollm2",  exist_ok=True)
os.makedirs(ADAPTER_DIR + "flan_t5",  exist_ok=True)
```

---

## Step 4 — Set Your Nebius API Key

In Colab, use Secrets instead of hardcoding the key:

1. Click the key icon in the left sidebar (Secrets)
2. Add a secret named `NEBIUS_API_KEY` with your key value
3. Enable notebook access for that secret

Then in the notebook, replace the `os.environ.get(...)` line with:

```python
from google.colab import userdata
os.environ["NEBIUS_API_KEY"] = userdata.get("NEBIUS_API_KEY")
NEBIUS_API_KEY = os.environ["NEBIUS_API_KEY"]
```

---

## Step 5 — Upload Already-Generated JSONL Files

You already ran Task 1.1 locally. Upload the four files so the notebook skips the Nebius API calls entirely (cells 14 and 15 will detect them and load from disk):

```
train_causal.jsonl
train_seq2seq.jsonl
test_causal.jsonl
test_seq2seq.jsonl
```

In Colab: left sidebar → Files (folder icon) → Upload. Files land in `/content/` by default, which is where the notebook looks for them.

Alternatively, copy from Drive:

```python
import shutil
for f in ["train_causal.jsonl", "train_seq2seq.jsonl",
          "test_causal.jsonl",  "test_seq2seq.jsonl"]:
    shutil.copy(DRIVE_DIR + f, f"/content/{f}")
```

---

## Step 6 — Save LoRA Adapters to Drive After Training

Call this immediately after `trainer.train()` completes for each model. If the Colab session disconnects before you save, the trained weights are gone.

```python
# SmolLM2 adapter
model.save_pretrained(ADAPTER_DIR + "smollm2/")
tokenizer.save_pretrained(ADAPTER_DIR + "smollm2/")

# flan-t5 adapter
model.save_pretrained(ADAPTER_DIR + "flan_t5/")
tokenizer.save_pretrained(ADAPTER_DIR + "flan_t5/")
```

`save_pretrained` on a PEFT model saves only the adapter weights (a few MB), not the full base model.

---

## Step 7 — Reload Adapters for Evaluation (Task 1.3)

If your session restarted between training and evaluation:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# SmolLM2
base = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
smollm2_ft = PeftModel.from_pretrained(base, ADAPTER_DIR + "smollm2/")

# flan-t5
from transformers import AutoModelForSeq2SeqLM
base_t5 = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small",
    quantization_config=bnb_config,
    device_map="auto",
)
flan_ft = PeftModel.from_pretrained(base_t5, ADAPTER_DIR + "flan_t5/")
```

---

## Tips

- **T4 is enough.** A100 has more VRAM but is much more restricted in availability. Stick with T4.
- **Keep the tab active.** Colab disconnects idle sessions. Don't run training and then switch away for an hour.
- **Run from the top each session.** Colab does not persist in-memory state between sessions — always re-run the install and import cells.
- **Local CPU was slower than Colab GPU** (confirmed from prior experience with assignment 1). Don't try to run training locally even without quantization — use Colab.

---

## Preventing Idle Disconnects

Colab disconnects after ~90 minutes of no interaction in the browser tab. A training cell that is actively running will not be killed mid-run, but if you close the laptop or switch away for too long, the browser loses its connection and the session ends.

**Option 1 — JavaScript keep-alive (recommended)**

Open the browser console (`Cmd+Option+J` on Mac) while your Colab tab is open, paste this, and press Enter:

```javascript
function keepAlive() {
  console.log("Colab keep-alive ping");
  document.querySelector("colab-connect-button").click();
}
setInterval(keepAlive, 60000);
```

This clicks the connect button every 60 seconds, resetting the idle timer. It runs only in your browser tab — close the tab or refresh and it stops.

**Option 2 — keep the tab visible**

Browsers throttle background tabs. Keep the Colab tab in the foreground while training runs, and do not close the laptop lid (which can suspend the browser process).

**Option 3 — add a Python heartbeat cell**

Run this in a separate cell before starting training. It prints a dot every 5 minutes so the cell stays active:

```python
import threading, time

def heartbeat():
    while True:
        print(".", end="", flush=True)
        time.sleep(300)

t = threading.Thread(target=heartbeat, daemon=True)
t.start()
```

**Note:** Free Colab has a hard 12-hour session limit regardless. If your training takes longer, save checkpoints to Drive during training using `trainer` callbacks.
