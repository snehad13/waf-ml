#!/usr/bin/env python
"""
Robust training script for encoder-decoder reconstruction (AE-style).
Supports T5/BART (seq2seq) or encoder+decoder combos.
Example:
  python waf/src/train_ae.py \
    --data waf/data/tokenized/benign.txt \
    --tok waf/models/tokenizer_hf \
    --out waf/models/ae_out \
    --model t5-small \
    --epochs 1 \
    --bsz 4
Or to use an encoder+causal-decoder pair:
  python waf/src/train_ae.py --data ... --tok ... --out ... --encoder distilbert-base-uncased --decoder gpt2 ...
"""
import argparse
import os
import json
import torch
from datasets import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    EncoderDecoderModel,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


def load_tokenizer(tok_dir: str) -> PreTrainedTokenizerFast:
    """Robust tokenizer loader: HF fast loader or build from vocab+merges."""
    print(f"[load_tokenizer] token dir: {tok_dir}")
    try:
        files = os.listdir(tok_dir)
    except Exception as e:
        raise RuntimeError(f"Cannot read tokenizer dir {tok_dir}: {e}")

    tok = None
    try:
        print("[load_tokenizer] Trying PreTrainedTokenizerFast.from_pretrained(...)")
        tok = PreTrainedTokenizerFast.from_pretrained(tok_dir)
        print("[load_tokenizer] Loaded via PreTrainedTokenizerFast")
    except Exception as e:
        print("[load_tokenizer] PreTrainedTokenizerFast.from_pretrained failed:", repr(e))

    if tok is None:
        vocab_path = os.path.join(tok_dir, "vocab.json")
        merges_path = os.path.join(tok_dir, "merges.txt")
        if not os.path.exists(vocab_path) and os.path.exists(os.path.join(tok_dir, "tokenizer-vocab.json")):
            vocab_path = os.path.join(tok_dir, "tokenizer-vocab.json")
        if not os.path.exists(merges_path) and os.path.exists(os.path.join(tok_dir, "tokenizer-merges.txt")):
            merges_path = os.path.join(tok_dir, "tokenizer-merges.txt")

        if os.path.exists(vocab_path) and os.path.exists(merges_path):
            print(f"[load_tokenizer] Found vocab and merges: {vocab_path} {merges_path}")
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            merges = []
            with open(merges_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if i == 0 and (line.startswith("#") or line == ""):
                        continue
                    if line == "":
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        merges.append(tuple(parts[:2]))
            try:
                from tokenizers import Tokenizer
                from tokenizers import models, pre_tokenizers, decoders

                bpe_model = models.BPE(vocab=vocab, merges=merges)
                tk = Tokenizer(bpe_model)
                tk.pre_tokenizer = pre_tokenizers.ByteLevel()
                tk.decoder = decoders.ByteLevel()
                tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>")
                print("[load_tokenizer] Built PreTrainedTokenizerFast wrapper from vocab+merges")
            except Exception as ex:
                print("[load_tokenizer] Failed to build BPE tokenizer programmatically:", repr(ex))
                tok = None
        else:
            print("[load_tokenizer] vocab+merges not present; files:", files)

    if tok is None:
        raise RuntimeError(f"Failed to load tokenizer from {tok_dir}. Files present: {files}")

    if tok.pad_token is None:
        print("[load_tokenizer] pad token missing — adding '<pad>'")
        tok.add_special_tokens({"pad_token": "<pad>"})

    print("[load_tokenizer] tokenizer ready. vocab_size:", getattr(tok, "vocab_size", None))
    return tok


def choose_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU for training")
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS GPU for training")
        return torch.device("mps")
    print("Using CPU for training")
    return torch.device("cpu")


def prepare_dataset(texts, tokenizer, max_len):
    """
    Build a datasets.Dataset where each example has 'input_ids', 'attention_mask', 'labels'
    and remove keys that some models don't accept (e.g., token_type_ids).
    """
    ds = Dataset.from_dict({"text": texts})

    def prep(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)
        # Remove token_type_ids if present (T5/BART don't accept it).
        if "token_type_ids" in enc:
            enc.pop("token_type_ids")
        # Ensure labels are copies of input_ids (lists of ints)
        enc["labels"] = [list(x) for x in enc["input_ids"]]
        return enc

    ds = ds.map(prep, batched=True, remove_columns=["text"])
    return ds



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tok", required=True, help="tokenizer directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="seq2seq model id (e.g., t5-small or facebook/bart-base)")
    ap.add_argument("--encoder", default=None, help="encoder model id (if using encoder+decoder combo)")
    ap.add_argument("--decoder", default=None, help="decoder model id (if using encoder+decoder combo)")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = choose_device()

    with open(args.data, "r", encoding="utf-8") as f:
        texts = [t.strip() for t in f if t.strip()]
    if len(texts) == 0:
        raise RuntimeError("No texts found in data file")

    tokenizer = load_tokenizer(args.tok)
    ds = prepare_dataset(texts, tokenizer, args.max_len)

    model = None
    # 1) If user provided a seq2seq model id (like t5-small or bart-base), try to load it directly
    if args.model:
        print(f"[main] trying to load seq2seq model id: {args.model}")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
            print("[main] Loaded seq2seq model via AutoModelForSeq2SeqLM")
        except Exception as e:
            print("[main] AutoModelForSeq2SeqLM.from_pretrained failed:", repr(e))
            model = None

    # 2) If not provided or failed, try encoder+decoder combo
    if model is None:
        if args.encoder and args.decoder:
            print(f"[main] building EncoderDecoderModel from encoder={args.encoder}, decoder={args.decoder}")
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder, args.decoder)
        else:
            raise RuntimeError("No model could be loaded. Provide --model <seq2seq-id> or both --encoder and --decoder.")

    # safe token id fallbacks
    cls_id = getattr(tokenizer, "cls_token_id", None) or getattr(tokenizer, "bos_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if cls_id is not None:
        try:
            model.config.decoder_start_token_id = cls_id
        except Exception:
            pass
    if sep_id is not None:
        try:
            model.config.eos_token_id = sep_id
        except Exception:
            pass
    if pad_id is not None:
        try:
            model.config.pad_token_id = pad_id
        except Exception:
            pass

    # resize embeddings to tokenizer vocab size
    try:
        model.resize_token_embeddings(len(tokenizer))
        print("[main] resized model token embeddings to", len(tokenizer))
    except Exception as e:
        print("[main] warning: resize_token_embeddings failed:", e)

    model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        fp16=False,
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=data_collator)

    trainer.train()

    trainer.save_model(args.out)
    try:
        tokenizer.save_pretrained(args.out)
    except Exception as e:
        print("[main] tokenizer.save_pretrained failed:", e)
    print(f"Model and tokenizer saved to {args.out}")


if __name__ == "__main__":
    main()
