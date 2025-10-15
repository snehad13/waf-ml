# waf/src/lo_ra_finetune.py
import argparse
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="waf/models/ae_out")
    ap.add_argument("--data", required=True)  # incremental benign lines
    ap.add_argument("--out", default="waf/models/ae_out_lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model = prepare_model_for_kbit_training(model) if hasattr(model, "get_input_embeddings") else model

    # LORA config (small)
    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    texts = load_texts(args.data)
    ds = Dataset.from_dict({"text": texts})

    def prep(batch):
        enc = tok(batch["text"], truncation=True, padding="max_length", max_length=128)
        enc["labels"] = [list(x) for x in enc["input_ids"]]
        return enc

    ds = ds.map(prep, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        fp16=False,
        logging_steps=50,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=data_collator)
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Saved LoRA model to", args.out)

if __name__ == "__main__":
    main()
