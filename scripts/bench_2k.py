#!/usr/bin/env python3
"""Throughput benchmark for an OpenAI-compatible vLLM endpoint.

Measures, for each concurrency level:
  * aggregate output throughput (tok/s)
  * per-stream decode throughput (median tok/s)
  * median TTFT

The workload is a ~2K-token Chinese prompt with 256 output tokens, greedy
decoding, streaming. This mirrors the reference numbers in docs/performance.md.

Usage:
    python3 scripts/bench_2k.py
    python3 scripts/bench_2k.py --base-url http://127.0.0.1:8021 \
        --model dsv41 --concurrency 1,4,8,16,32

API key: --api-key or DSV41_API_KEY / API_KEY.
Requires: pip install aiohttp
"""

from __future__ import annotations

import argparse
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


async def one_request(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    started = time.perf_counter()
    ttft = None
    last_token_at = started
    usage_tokens = 0
    chunks = 0
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=900)) as resp:
        if resp.status != 200:
            body = await resp.text()
            return {"error": f"HTTP {resp.status}: {body[:160]}"}
        async for raw in resp.content:
            line = raw.strip()
            if not line or line == b"data: [DONE]":
                continue
            if not line.startswith(b"data: "):
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens", usage_tokens)
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - started
                last_token_at = time.perf_counter()
                chunks += 1
    tokens = usage_tokens or chunks
    decode_time = max(last_token_at - started - (ttft or 0.0), 1e-9)
    return {
        "ttft": ttft or 0.0,
        "tokens": tokens,
        "decode_tps": (tokens - 1) / decode_time if tokens > 1 else 0.0,
        "total": last_token_at - started,
    }


async def run_level(base_url: str, model: str, prompt: str, max_tokens: int, concurrency: int,
                    headers: dict, repeats: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = base_url.rstrip("/") + "/v1/completions"
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    aggregates, streams, ttfts = [], [], []
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        for _ in range(repeats):
            started = time.perf_counter()
            results = await asyncio.gather(
                *[one_request(session, url, payload) for _ in range(concurrency)]
            )
            wall = time.perf_counter() - started
            ok = [r for r in results if "error" not in r]
            if not ok:
                err = next((r["error"] for r in results if "error" in r), "unknown")
                raise RuntimeError(err)
            total_tokens = sum(r["tokens"] for r in ok)
            aggregates.append(total_tokens / wall)
            streams.append(statistics.median(r["decode_tps"] for r in ok))
            ttfts.append(statistics.median(r["ttft"] for r in ok))
    return {
        "concurrency": concurrency,
        "aggregate": statistics.median(aggregates),
        "stream": statistics.median(streams),
        "ttft": statistics.median(ttfts),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("DSV41_BASE_URL", "http://127.0.0.1:8021"))
    parser.add_argument("--model", default=os.environ.get("DSV41_SERVED_NAME", "dsv41"))
    parser.add_argument("--api-key", default=(os.environ.get("DSV41_API_KEY") or os.environ.get("DSV41_W8A8_API_KEY")
                        or os.environ.get("API_KEY", "")))
    parser.add_argument("--concurrency", default="1,2,4,8,16,32",
                        help="comma-separated concurrency levels")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-repeat", type=int, default=3, help="≈2K tokens at repeat=3")
    parser.add_argument("--repeats", type=int, default=3, help="repetitions per level (median)")
    args = parser.parse_args()

    prompt = PARAGRAPH * args.prompt_repeat
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    print(f"endpoint : {args.base_url}")
    print(f"model    : {args.model}")
    print(f"workload : prompt≈2K tokens, max_tokens={args.max_tokens}, temperature=0, streaming")
    print()

    rows = []
    for level in levels:
        row = await run_level(args.base_url, args.model, prompt, args.max_tokens, level, headers, args.repeats)
        rows.append(row)
        print(f"concurrency {level:>3}: aggregate {row['aggregate']:8.1f} tok/s | "
              f"stream {row['stream']:6.1f} tok/s | TTFT {row['ttft'] * 1000:7.1f} ms")

    print()
    print("| concurrency | aggregate tok/s | stream tok/s | TTFT (ms) |")
    print("|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['concurrency']} | {row['aggregate']:.1f} | {row['stream']:.1f} | "
              f"{row['ttft'] * 1000:.0f} |")


if __name__ == "__main__":
    asyncio.run(main())
