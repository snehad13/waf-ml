A minimal, runnable starter kit for a hybrid WAF that combines deterministic rule-based detection (regex signatures) with an ML autoencoder (reconstruction loss) to detect anomalous HTTP requests. This repository contains data generation, parsing & normalization, tokenizer training, autoencoder training, a non-blocking FastAPI inference service, and a streaming ingest pipeline.
Quick start (developer flow)

Run these commands from the repository root.

# 1) Create & activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# 2) Install Python dependencies
pip install -r requirements.txt
# minimal runtime if you want only the server:
pip install fastapi uvicorn transformers torch aiohttp aiofiles

# 3) (optional) Generate synthetic logs for testing
python src/synth_logs.py --out data/synthetic_access.log --n 10000

# 4) Parse & normalize logs to one request-per-line
python src/parser_normalizer.py --in data/synthetic_access.log --out data/normalized_requests.txt

# 5) Train tokenizer (produces models/req_tokenizer/)
python src/train_tokenizer.py --in data/normalized_requests.txt --tok_dir models/req_tokenizer

# 6) Train a toy autoencoder (produces models/ae_out/)
python src/train_ae.py --data data/normalized_requests.txt --tok models/req_tokenizer --out models/ae_out

# 7) Compute recon-loss statistics (p95/p99 thresholds)
python src/eval_recon_loss.py --model models/ae_out --data data/normalized_requests.txt --out models/loss_stats.json

Run the inference service

Important: run from the repository root so import waf... resolves. If you run from another directory, set PYTHONPATH=. (examples below).

Production / normal run (recommended, GPU/MPS safe)
# activate venv first
source .venv/bin/activate

# start the FastAPI scoring service bound to all interfaces (single worker)
uvicorn waf.services.inference_service_simple:app --host 0.0.0.0 --port 8000 --workers 1


Why these flags

waf.services.inference_service_simple:app — imports the app FastAPI instance from waf/services/inference_service_simple.py.

--host 0.0.0.0 — binds to all interfaces (use 127.0.0.1 for local-only).

--port 8000 — listens on TCP port 8000.

--workers 1 — single worker. Use 1 when using GPU/MPS because each worker will load the model into device memory; multiple workers can exhaust memory.

Dev run (auto-reload)
# auto-reload on file changes (dev convenience)
PYTHONPATH=. uvicorn waf.services.inference_service_simple:app --reload --host 127.0.0.1 --port 8000

If you see import errors

If ModuleNotFoundError appears (e.g. No module named 'waf.services'), either:

run the uvicorn command from the repo root, or

set PYTHONPATH=.:

PYTHONPATH=. uvicorn waf.services.inference_service_simple:app --host 0.0.0.0 --port 8000 --workers 1

Example (quick test)
curl -sS -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"raw":"GET /product?id=1 UNION SELECT password FROM users HTTP/1.1"}' | jq


Service should return JSON similar to:

{
  "score": 2.2938,
  "rules": [
    {"rule_id":"sqli_union_select","description":"SQLi - UNION SELECT","match":"UNION SELECT","severity":9}
  ],
  "anomaly": true,
  "threshold": 0.0004771026541129632,
  "action": "alert"
}

Where normalization & tokenization happen (precise)

Normalization

Implemented in src/normalize.py.

parse_access_log_line(line) parses a raw access-log line (method, URI, user-agent, etc.).

normalize_request(method, uri, ua) produces the canonical string the system uses (e.g., replaces IDs with placeholders like <<SYM>>, decodes percent-encoding, canonicalizes query order). Normalization must be identical between training and inference.

Tokenization

Training: src/train_tokenizer.py trains and saves the tokenizer (e.g., Byte-Pair / SentencePiece) into models/req_tokenizer.

Inference: services/inference_service_simple.py loads the tokenizer at startup (AutoTokenizer.from_pretrained(MODEL_DIR)) and tokenizes the normalized text right before model inference (tokenizer(text, return_tensors="pt", ...)).

Rule engine integration

Add deterministic rules in waf/rules/rules.py with a function check_rules(raw: str) -> list[dict]. Example match object:

{"rule_id":"sqli_union_select","description":"SQLi - UNION SELECT","severity":9,"match":"UNION SELECT","start":123,"end":134}


Scoring service should import and call check_rules(normalized_text) and include rules in the /score response.

Ingest pipeline (src/ingest_stream.py) should persist rules alongside score and anomaly in logs/stream_scores.log.