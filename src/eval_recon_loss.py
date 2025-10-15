# waf/src/eval_recon_loss.py
import json, os, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "waf/models/ae_out"
DATA_FILE = "waf/data/tokenized/benign.txt"
OUT_STATS = "waf/models/loss_stats.json"
OUT_NPY = "waf/models/benign_losses.npy"
MAX_LEN = 128
LIMIT = None  # set to int to limit samples (speed)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("device:", device)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
model.eval()

def loss_for_text(text):
    enc = tok(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN)
    enc.pop("token_type_ids", None)
    for k in enc:
        enc[k] = enc[k].to(device)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
        return float(out.loss.cpu().item())

losses = []
count = 0
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        t = line.strip()
        if not t:
            continue
        losses.append(loss_for_text(t))
        count += 1
        if count % 500 == 0:
            print("computed losses:", count)
        if LIMIT and count >= LIMIT:
            break

losses = np.array(losses, dtype=float)
stats = {
    "n": int(len(losses)),
    "mean": float(np.mean(losses)) if len(losses) else None,
    "std": float(np.std(losses)) if len(losses) else None,
    "p95": float(np.percentile(losses, 95)) if len(losses) else None,
    "p99": float(np.percentile(losses, 99)) if len(losses) else None,
}
os.makedirs(os.path.dirname(OUT_STATS), exist_ok=True)
with open(OUT_STATS, "w") as fh:
    json.dump(stats, fh, indent=2)
np.save(OUT_NPY, losses)
print("Saved stats to", OUT_STATS)
print(stats)
