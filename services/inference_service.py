#!/usr/bin/env python
import asyncio
from fastapi import FastAPI, Request
import torch
from transformers import PreTrainedTokenizerFast, EncoderDecoderModel

TOK_DIR = "models/req_tokenizer"
MODEL_DIR = "models/ae_out"

app = FastAPI()
tokenizer = None
model = None
queue = asyncio.Queue(maxsize=1024)

async def worker():
    global tokenizer, model
    while True:
        seq, fut = await queue.get()
        try:
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            fut.set_result({"status":"ok", "score": loss})
        except Exception as e:
            fut.set_result({"status":"error", "error": str(e)})
        finally:
            queue.task_done()

@app.on_event("startup")
async def startup():
    global tokenizer, model
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOK_DIR)
    model = EncoderDecoderModel.from_pretrained(MODEL_DIR)
    model.eval()
    for _ in range(4):
        asyncio.create_task(worker())

@app.post("/detect")
async def detect(req: Request):
    payload = await req.json()
    seq = payload.get("normalized","").strip()
    if not seq:
        return {"status":"bad_request", "msg":"normalized string missing"}
    fut = asyncio.get_event_loop().create_future()
    try:
        queue.put_nowait((seq, fut))
    except asyncio.QueueFull:
        return {"status":"drop", "score": None}
    try:
        res = await asyncio.wait_for(fut, timeout=0.15)
        return res
    except asyncio.TimeoutError:
        return {"status":"timeout", "score": None}
