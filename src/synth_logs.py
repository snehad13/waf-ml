#!/usr/bin/env python
import argparse, random, datetime
from faker import Faker

fake = Faker()

METHODS = ["GET","POST"]
TEMPLATES = [
    "/",
    "/products",
    "/products/{id}/details",
    "/login",
    "/search?q={q}",
    "/api/user/{id}/profile",
    "/cart/add?id={id}&qty={qty}",
    "/order/{id}/status",
]

UAS = [
    "Mozilla/5.0",
    "curl/7.68.0",
    "PostmanRuntime/7.26.8",
    "Chrome/122.0 Safari/537.36",
]

def make_line():
    ip = fake.ipv4_public()
    ts = datetime.datetime.utcnow().strftime('%d/%b/%Y:%H:%M:%S +0000')
    method = random.choice(METHODS)
    path = random.choice(TEMPLATES)
    if "{id}" in path:
        path = path.format(id=random.randint(1,20000), qty=random.randint(1,100))

    if "{q}" in path:
        path = path.format(q=random.choice(["shoes","laptop","phone","books","toys"]))
    if "{qty}" in path:
        path = path.format(qty=random.randint(1,4))
    protocol = "HTTP/1.1"
    status = random.choice([200]*92 + [302]*4 + [404]*4)
    size = random.randint(100,5000)
    ref = "-"
    ua = random.choice(UAS)
    req = f'{method} {path} {protocol}'
    return f'{ip} - - [{ts}] "{req}" {status} {size} "{ref}" "{ua}"'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120000)
    args = ap.parse_args()
    with open(args.out, "w") as f:
        for _ in range(args.n):
            f.write(make_line()+"\n")
    print(f"Wrote {args.n} lines to {args.out}")

if __name__ == "__main__":
    main()
