#!/usr/bin/env python
# waf/src/ingest_batch.py
import argparse, os
from normalize import parse_access_log_line, normalize_request

def process_file(infile, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    count = 0
    with open(infile, "r", encoding="utf-8", errors="ignore") as inf, open(outfile, "a", encoding="utf-8") as outf:
        for line in inf:
            parsed = parse_access_log_line(line)
            if not parsed:
                continue
            method, uri, ua = parsed
            norm = normalize_request(method, uri, ua)
            outf.write(norm + "\n")
            count += 1
    print(f"Wrote {count} normalized lines to {outfile}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logfile", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    process_file(args.logfile, args.out)

if __name__ == "__main__":
    main()
