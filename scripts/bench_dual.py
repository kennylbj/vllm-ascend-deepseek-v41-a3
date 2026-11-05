#!/usr/bin/env python3
"""Two-prefix benchmark for the DP2 external-LB / prefix-router setup.

bench_2k.py sends one identical prompt to every request. With prefix-affinity
routing that hot prefix can concentrate on one DP rank, so its low-concurrency
aggregate is not a DP2 ceiling. This script alternates two different long
prefixes (A/B): each rank owns one prefix and both DP ranks can be exercised.

Examples:
    # ~2K prompt / 256 output
    python3 scripts/bench_dual.py --concurrency 2,4,8,16,32

    # ~8K prompt / 1024 output
    python3 scripts/bench_dual.py --prompt-repeat 11 --max-tokens 1024 \
        --concurrency 2,4,8,16
"""
from __future__ import annotations

import argparse
import os
import asyncio
import json
import os
import statistics
import time

import aiohttp

PARAGRAPH = (
    "Transformer架构的核心是自注意力机制。对于输入序列X，通过可学习的权重矩阵W_Q、W_K、W_V"
    "将其投影到Query、Key、Value三个表示空间。注意力权重通过Q和K的点积计算，除以√d_k缩放后"
    "经softmax归一化，再与V相乘。多头注意力将QKV投影到多个子空间独立计算后拼接。"
    "位置编码注入位置信息，RoPE对Q和K应用旋转矩阵编码相对位置，外推性好。"
    "每层包含注意力子层、残差连接、LayerNorm、前馈网络FFN。现代变体用RMSNorm替代LayerNorm，"
    "用SwiGLU替代ReLU。GPT系列只用Decoder，BERT只用Encoder。大语言模型都基于Transformer。"
)


async def one_request(session, url, payload):
    started = time.perf_counter()
    ttft = None
    last = started
    usage = 0
    chunks = 0
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=900)) as resp:
        if resp.status != 200:
            return {"error": f"HTTP {resp.status}: {(await resp.text())[:160]}"}
        async for raw in resp.content:
            line = raw.strip()
            if not line or line == b"data: [DONE]" or not line.startswith(b"data: "):
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"].get("completion_tokens", usage)
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - started
                last = time.perf_counter()
                chunks += 1
    tokens = usage or chunks
    decode = max(last - started - (ttft or 0.0), 1e-9)
    return {"ttft": ttft or 0.0, "tokens": tokens,
            "decode_tps": (tokens - 1) / decode if tokens > 1 else 0.0,
            "total": last - started}


async def run_level(base_url, model, prompts, max_tokens, concurrency, headers, repeats):
    url = base_url.rstrip("/") + "/v1/completions"
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    aggregates, streams, ttfts = [], [], []
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        for _ in range(repeats):
            payloads = []
            for i in range(concurrency):
                payloads.append({
                    "model": model,
                    "prompt": prompts[i % len(prompts)],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                })
            started = time.perf_counter()
            results = await asyncio.gather(*[one_request(session, url, p) for p in payloads])
            wall = time.perf_counter() - started
            ok = [r for r in results if "error" not in r]
            if not ok:
                raise RuntimeError(results[0]["error"])
            total = sum(r["tokens"] for r in ok)
            aggregates.append(total / wall)
            streams.append(statistics.median(r["decode_tps"] for r in ok))
            ttfts.append(statistics.median(r["ttft"] for r in ok))
    return {"concurrency": concurrency, "aggregate": statistics.median(aggregates),
            "stream": statistics.median(streams), "ttft": statistics.median(ttfts)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8021")
    ap.add_argument("--model", default=os.environ.get("DSV41_SERVED_NAME", "dsv41"))
    ap.add_argument("--api-key", default=os.environ.get("DSV41_API_KEY", ""))
    ap.add_argument("--concurrency", default="2,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-repeat", type=int, default=3,
                    help="≈2K tokens at 3; ≈8K at 11")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    # Distinct first blocks force the router to keep A and B on separate ranks.
    prompt_a = "ALPHA-7f3c9a2e unique workload block. " * 20 + PARAGRAPH * args.prompt_repeat
    prompt_b = "BETA-91d4e6b0 unique workload block. " * 20 + PARAGRAPH * args.prompt_repeat
    prompts = [prompt_a, prompt_b]
    url = args.base_url.rstrip("/") + "/v1/completions"

    print(f"workload : prompt≈{args.prompt_repeat}×paragraph, max_tokens={args.max_tokens}, "
          f"two prefixes A/B, temperature=0, streaming")
    print("warm-up A/B ...")
    async with aiohttp.ClientSession(headers=headers) as session:
        warm = [{"model": args.model, "prompt": p, "max_tokens": 8,
                 "temperature": 0, "stream": True,
                 "stream_options": {"include_usage": True}} for p in prompts]
        await asyncio.gather(*[one_request(session, url, p) for p in warm])
    await asyncio.sleep(2)

    rows = []
    for level in [int(x) for x in args.concurrency.split(",") if x.strip()]:
        row = await run_level(args.base_url, args.model, prompts, args.max_tokens,
                              level, headers, args.repeats)
        rows.append(row)
        print(f"concurrency {level:>3}: aggregate {row['aggregate']:8.1f} tok/s | "
              f"stream {row['stream']:6.1f} tok/s | TTFT {row['ttft'] * 1000:7.1f} ms")
    print()
    print("| concurrency | aggregate tok/s | stream tok/s | TTFT (ms) |")
    print("|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['concurrency']} | {r['aggregate']:.1f} | {r['stream']:.1f} | {r['ttft']*1000:.0f} |")


if __name__ == "__main__":
    asyncio.run(main())
