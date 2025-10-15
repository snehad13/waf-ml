# --- paste/replace token loading section in waf/src/sanity_check.py ---
import os, json
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers, decoders, normalizers

tok_dir = "waf/models/tokenizer_hf"
print("token dir:", tok_dir)
files = os.listdir(tok_dir)
print("files:", files)

tok = None

# 1) Try HF fast loader (best effort)
try:
    print("Trying PreTrainedTokenizerFast.from_pretrained(...)")
    tok = PreTrainedTokenizerFast.from_pretrained(tok_dir)
    print("Loaded via PreTrainedTokenizerFast")
except Exception as e:
    print("PreTrainedTokenizerFast failed:", repr(e))

# 2) If HF loader failed, try building from vocab.json + merges.txt
if tok is None:
    vocab_path = os.path.join(tok_dir, "vocab.json")
    merges_path = os.path.join(tok_dir, "merges.txt")
    # Some tokenizers also store files named tokenizer-vocab.json / tokenizer-merges.txt
    if not os.path.exists(vocab_path) and os.path.exists(os.path.join(tok_dir, "tokenizer-vocab.json")):
        vocab_path = os.path.join(tok_dir, "tokenizer-vocab.json")
    if not os.path.exists(merges_path) and os.path.exists(os.path.join(tok_dir, "tokenizer-merges.txt")):
        merges_path = os.path.join(tok_dir, "tokenizer-merges.txt")

    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print("Found vocab and merges:", vocab_path, merges_path)
        # load vocab.json
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)  # dict: token -> id

        # load merges: ignore first line if it starts with '#version' or empty
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0 and (line.startswith("#") or line == ""):
                    continue
                if line == "":
                    continue
                # merges lines are like: "t h\n" or "Ġ t h"? usually "a b"
                parts = line.split()
                if len(parts) >= 2:
                    merges.append(tuple(parts[:2]))
        print(f"Loaded {len(vocab)} vocab tokens and {len(merges)} merges")

        # Build a tokenizers Tokenizer using BPE model with provided vocab and merges
        try:
            bpe_model = models.BPE(vocab=vocab, merges=merges)
            tk = Tokenizer(bpe_model)
            # ByteLevel pre-tokenizer & decoder are common for tokenizers created by tokenizers library
            tk.pre_tokenizer = pre_tokenizers.ByteLevel()
            tk.decoder = decoders.ByteLevel()
            # (optional) normalizer: ByteLevel works without extra normalizer, skip if not present
            tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>")
            print("Built PreTrainedTokenizerFast wrapper from vocab+merges")
        except Exception as ex:
            print("Failed to build BPE tokenizer programmatically:", repr(ex))
            tok = None
    else:
        print("No vocab+merges pair found in tokenizer dir; files present:", files)

if tok is None:
    raise RuntimeError(f"Failed to load tokenizer from {tok_dir}. Files present: {files}")

# Ensure pad token exists
if tok.pad_token is None:
    print("[auto-fix] adding pad token '<pad>'")
    tok.add_special_tokens({"pad_token": "<pad>"})

print("Tokenizer ready. vocab_size:", getattr(tok, "vocab_size", None))
# --- end replacement block ---
