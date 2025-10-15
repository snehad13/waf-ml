#!/usr/bin/env bash
URL="http://localhost:8000/score"

REQ() {
  local raw="$1"
  echo "===="
  echo "Request: $raw"
  curl -s -X POST "$URL" -H "Content-Type: application/json" -d "{\"raw\":\"$raw\"}" | jq .
  echo
}

REQ "GET /home HTTP/1.1"

REQ "GET /search?q=1 OR 1=1 HTTP/1.1"
REQ "GET /product?id=100' OR '1'='1 HTTP/1.1"

REQ "GET /comment?msg=<script>alert(1)</script> HTTP/1.1"
REQ "GET /profile?bio=<img src=x onerror=alert(1)> HTTP/1.1"

REQ "GET /download?file=../../etc/passwd HTTP/1.1"
REQ "GET /view?file=../../../var/log/auth.log HTTP/1.1"

REQ "GET /run?cmd=;ls -la; HTTP/1.1"

echo "Done."
