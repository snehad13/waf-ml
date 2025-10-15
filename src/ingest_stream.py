#!/usr/bin/env python3
# waf/src/ingest_stream.py
"""
Tail a logfile and POST each new normalized request to the scoring endpoint.
Writes out JSON lines including model score, rule matches and anomaly flag.

Usage:
  python src/ingest_stream.py --logfile data/sample_access.log --url http://localhost:8000/score --out logs/stream_scores.log --concurrency 4
"""

import argparse
import asyncio
import json
import os
import signal
from datetime import datetime
from typing import Optional, Set

import aiohttp
import aiofiles
import asyncio.subprocess as subprocess

# adjust import path if needed; this matches your original code
from normalize import parse_access_log_line, normalize_request

STOP = False


def _on_signal(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


async def post_score(session: aiohttp.ClientSession, url: str, text: str, sem: asyncio.Semaphore, out_fh):
    """
    Post one normalized text to the scoring endpoint and append result JSON to out_fh (an aiofiles handle).
    The response is expected to be JSON and may contain keys: 'score', 'rules', 'anomaly'.
    """
    async with sem:
        payload = {"raw": text}
        score = None
        rules = []
        is_anomaly = None

        try:
            async with session.post(url, json=payload, timeout=10) as resp:
                # try reading JSON; fall back gracefully if something unexpected happens
                try:
                    j = await resp.json()
                except Exception:
                    # if response isn't JSON, try reading text for logging but keep going
                    txt = await resp.text()
                    j = {}
                    # optionally include raw response text under a debug key
                    j["_raw_response_text"] = txt

                score = j.get("score")
                rules = j.get("rules", [])
                is_anomaly = j.get("anomaly", None)
        except asyncio.TimeoutError:
            # timeout: keep score None
            score = None
        except aiohttp.ClientError:
            # connection issues etc.
            score = None
        except Exception:
            # unexpected error; don't crash the loop
            score = None

        ts = datetime.utcnow().isoformat()
        out_obj = {
            "ts": ts,
            "raw": text,
            "score": score,
            "rules": rules,
            "anomaly": is_anomaly,
        }

        # write asynchronously
        try:
            await out_fh.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            await out_fh.flush()
        except Exception:
            # if write fails, we don't want to crash the worker; print to stderr
            print("Failed to write output for line:", out_obj, flush=True)


async def tail_and_post(logfile: str, url: str, outpath: str, concurrency: int, poll_interval: float = 0.1):
    """
    Tail logfile (via `tail -F`), normalize each parsed log line, and post concurrently.
    """
    # ensure output directory exists
    outdir = os.path.dirname(outpath) or "."
    os.makedirs(outdir, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)

    # start tail -F as subprocess
    proc = await subprocess.create_subprocess_exec(
        "tail", "-F", logfile, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    tasks: Set[asyncio.Task] = set()

    async with aiohttp.ClientSession() as session, aiofiles.open(outpath, "a", encoding="utf-8") as out_fh:
        global STOP
        while not STOP:
            # read next line (bytes) from tail stdout
            try:
                line = await proc.stdout.readline()
            except Exception:
                await asyncio.sleep(poll_interval)
                continue

            if not line:
                # no new data; allow other coroutines to run and poll again
                await asyncio.sleep(poll_interval)
                # prune finished tasks occasionally
                if tasks:
                    done = {t for t in tasks if t.done()}
                    tasks -= done
                continue

            try:
                line = line.decode("utf-8", errors="ignore")
            except Exception:
                # skip undecodable lines
                continue

            parsed = parse_access_log_line(line)
            if not parsed:
                # skip lines that don't parse as access logs
                continue

            method, uri, ua = parsed
            norm = normalize_request(method, uri, ua)

            # schedule the post; track the task so we can wait on shutdown
            task = asyncio.create_task(post_score(session, url, norm, sem, out_fh))
            tasks.add(task)

            # prune finished tasks immediately to keep set small
            if tasks:
                done = {t for t in tasks if t.done()}
                tasks -= done

        # STOP requested: wait briefly for outstanding tasks to finish
        if tasks:
            try:
                await asyncio.wait(tasks, timeout=15)
            except Exception:
                pass

        # try to terminate tail process
        try:
            proc.terminate()
            await proc.wait()
        except Exception:
            pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logfile", required=True, help="Path to input logfile to tail")
    ap.add_argument("--url", default="http://localhost:8000/score", help="Scoring endpoint URL")
    ap.add_argument("--out", default="waf/logs/stream_scores.log", help="Output file (JSON lines)")
    ap.add_argument("--concurrency", type=int, default=16, help="Number of concurrent HTTP posts")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        asyncio.run(tail_and_post(args.logfile, args.url, args.out, args.concurrency))
    except Exception as e:
        print("Fatal error in ingest_stream:", e, flush=True)
        raise


if __name__ == "__main__":
    main()
