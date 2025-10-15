# waf/services/inference_service_simple.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import asyncio
import json
from typing import Any, Dict, List
import re

# ==============================
# Inline regex-based rule engine
# ==============================

INLINE_RULES = [
    # ---- SQL Injection (SQLi) ----
    ("sqli_union_select", r"\bUNION\b.*\bSELECT\b", "SQLi - UNION SELECT", 9),
    ("sqli_or_true", r"(?:(?:')|(?:\"))\s*OR\s+(?:'1'='1'|1=1|true)", "SQLi - OR '1'='1'", 8),
    ("sqli_comment", r"(--\s|/\*.*?\*/|#\s)", "SQLi - comment syntax", 6),
    ("sqli_sleep", r"\bSLEEP\(\s*\d+\s*\)", "SQLi - time-based injection", 8),
    ("sqli_stack", r";\s*(DROP|SELECT|INSERT|UPDATE|DELETE)\b", "SQLi - stacked query", 7),

    # ---- XSS (Cross-site scripting) ----
    ("xss_script_tag", r"<\s*script\b", "XSS - <script> tag", 9),
    ("xss_on_event", r"\bon\w+\s*=", "XSS - on* event attribute", 6),
    ("xss_eval", r"\beval\s*\(", "XSS - eval() usage", 7),
    ("xss_src_js", r"src\s*=\s*['\"]\s*javascript:", "XSS - javascript: src", 8),
    ("xss_iframe", r"<\s*iframe\b", "XSS - iframe injection", 7),
    ("xss_img_js", r"<\s*img\b[^>]*\bonerror\s*=", "XSS - <img onerror>", 8),

    # ---- Path Traversal ----
    ("path_traversal", r"\.\./|\.\.\\", "Path traversal attempt", 7),

    # ---- Command Injection ----
    ("cmd_injection", r"[;&|]\s*(cat|ls|whoami|pwd|curl|wget|rm|echo)\b", "Command injection attempt", 8),

    # ---- SSRF ----
    ("ssrf_internal", r"http://(127\.0\.0\.1|169\.254\.169\.254|localhost)", "SSRF - internal host", 8),

    # ---- Misc ----
    ("double_encoded", r"%25[0-9A-Fa-f]{2}", "Double-encoded sequence", 5),
]

# Precompile all regex patterns once at startup
COMPILED_RULES = [
    (rule_id, re.compile(pattern, re.IGNORECASE | re.DOTALL), desc, sev)
    for rule_id, pattern, desc, sev in INLINE_RULES
]

def check_rules(text: str, max_len: int = 2000) -> List[Dict]:
    """Run inline regex checks on given text and return all matches."""
    if not text:
        return []
    if len(text) > max_len:
        text = text[:max_len]

    matches = []
    for rule_id, pattern, desc, sev in COMPILED_RULES:
        for m in pattern.finditer(text):
            matches.append({
                "rule_id": rule_id,
                "description": desc,
                "severity": sev,
                "match": m.group(0),
                "start": m.start(),
                "end": m.end(),
            })
    return matches

# ==============================
# Model & FastAPI service setup
# ==============================

MODEL_DIR = "waf/models/ae_out"
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model/tokenizer from {MODEL_DIR}: {e}")

app = FastAPI(title="Hybrid WAF — Rules + ML Inference")

class Req(BaseModel):
    raw: str

def load_threshold(path: str = "waf/models/loss_stats.json") -> float:
    """Load p99 threshold from JSON file. Returns +inf if missing."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return float(json.load(fh).get("p99", float("inf")))
    except Exception:
        return float("inf")

async def _compute_loss(text: str) -> float:
    """Compute reconstruction loss using the trained autoencoder."""
    def sync_infer(t: str) -> float:
        enc = tokenizer(
            t,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        enc.pop("token_type_ids", None)
        for k in enc.keys():
            enc[k] = enc[k].to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            return float(out.loss.cpu().item())

    return await asyncio.to_thread(sync_infer, text)

@app.post("/score")
async def score(req: Req) -> Dict[str, Any]:
    text = req.raw or ""

    # 1️⃣ Run regex-based detection
    rules = []
    try:
        rules = check_rules(text)
    except Exception:
        pass  # ignore rule errors

    # 2️⃣ Compute ML anomaly score
    try:
        score_value = await _compute_loss(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # 3️⃣ Determine anomaly using p99 threshold
    threshold = load_threshold()
    is_anomaly = score_value > threshold

    # 4️⃣ Decide action (policy)
    action = "allow"
    if rules:
        max_sev = max(r["severity"] for r in rules)
        if max_sev >= 9:
            action = "block"
        elif max_sev >= 6:
            action = "alert"
    elif is_anomaly:
        action = "alert"

    # 5️⃣ Build unified response
    response: Dict[str, Any] = {
        "score": score_value,
        "rules": rules,
        "anomaly": is_anomaly,
        "threshold": None if threshold == float("inf") else threshold,
        "action": action,
    }

    return response
