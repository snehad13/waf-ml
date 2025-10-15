#!/usr/bin/env python
import argparse, os
from tokenizers import ByteLevelBPETokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=8000)
    args = ap.parse_args()

    os.makedirs(args.tok_dir, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train(files=[args.inp], vocab_size=args.vocab_size, min_frequency=2,
              special_tokens=["<pad>","<s>","</s>","<unk>"])
    tok.save_model(args.tok_dir, "tokenizer")
    print(f"Saved tokenizer to {args.tok_dir}")

if __name__ == "__main__":
    main()
