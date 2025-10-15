#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-http://localhost:8081}"

echo "[*] Benign requests"
curl -sS "$HOST/search?q=shoes" | head -c 80; echo
curl -sS "$HOST/products/123/details" | head -c 80; echo

echo "[*] Suspicious requests"
curl -sS "$HOST/search?q=<script>alert(1)</script>" | head -c 80; echo
curl -sS "$HOST/login?user=admin' OR '1'='1" | head -c 80; echo
