#!/usr/bin/env python
import argparse, re, sys
from urllib.parse import urlparse, parse_qsl, unquote

# Regexes for normalization
UUID_RE = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}(-[0-9a-fA-F]{4}){2}-[0-9a-fA-F]{12}')
INT_RE = re.compile(r'\b\d+\b')
BASE64_RE = re.compile(r'(?:[A-Za-z0-9+/]{4}){2,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')

def normalize_value(v: str) -> str:
    v = unquote(v)
    v = UUID_RE.sub('<UUID>', v)
    v = EMAIL_RE.sub('<EMAIL>', v)
    v = BASE64_RE.sub('<TOKEN>', v)
    v = INT_RE.sub('<INT>', v)
    v = re.sub(r'/[0-9a-zA-Z]{6,}', '/<SEG>', v)
    return v

def normalize_request(request_line: str) -> str:
    """
    Normalize a single HTTP request string like:
    'GET /api/user/123?session=abc123 HTTP/1.1'
    Returns normalized string with placeholders.
    """
    try:
        method, path, _ = request_line.split()
        parsed = urlparse(path)
        path_norm = normalize_value(parsed.path)
        qs_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        qs = "&".join(f"{k}={normalize_value(v)}" for k,v in qs_pairs)
        return f"[METHOD] {method} [PATH] {path_norm} [QS] {qs}"
    except Exception:
        return ""

def parse_apache_line(line: str) -> str:
    """
    Parse a full Apache log line and normalize request + user-agent.
    """
    try:
        quoted = line.split('"')
        req = quoted[1]
        ua = quoted[-2] if len(quoted) >= 3 else ""
        method, path, _ = req.split()
        parsed = urlparse(path)
        path_norm = normalize_value(parsed.path)
        qs_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        qs = "&".join(f"{k}={normalize_value(v)}" for k,v in qs_pairs)
        ua_norm = normalize_value(ua)
        return f"[METHOD] {method} [PATH] {path_norm} [QS] {qs} [UA] {ua_norm}"
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n_in, n_out = 0, 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            n_in += 1
            s = parse_apache_line(line)
            if s:
                fout.write(s+"\n")
                n_out += 1
    print(f"Parsed {n_in} lines → {n_out} normalized")

if __name__ == "__main__":
    main()
