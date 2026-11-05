#!/usr/bin/env python3
"""8K-prompt speculative-decoding benchmark with acceptance-rate accounting.

Reports per-step latency, single-stream throughput, aggregate throughput and
TTFT for 8K-prompt → 512/1024-token generations at several concurrency levels.

Single-stream throughput formula used by the reference run:
    tokens_per_step = 1 + acceptance_rate * num_speculative_tokens
    step_ms         = decode_time_ms / num_forward_passes
    stream_tok/s    = (1000 / step_ms) * tokens_per_step
    aggregate       = stream_tok/s * concurrency

Acceptance rate is read from /metrics after a warm-up request.

Usage:
    python3 scripts/bench_8k.py
    python3 scripts/bench_8k.py --base-url http://127.0.0.1:8021 --max-tokens 512
Requires: pip install aiohttp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import urllib.request

import aiohttp

PARAGRAPH = (
    "Transformer架构的核心是自注意力机制。对于输入序列X，通过可学习的权重矩阵W_Q、W_K、W_V"
    "将其投影到Query、Key、Value三个表示空间。注意力权重通过Q和K的点积计算，除以√d_k缩放后"
    "经softmax归一化，再与V相乘：Attention(Q,K,V)=softmax(Q·K^T/√d_k)·V。多头注意力将QKV"
    "投影到h个子空间独立计算注意力后拼接，每个头可关注语法依赖、语义关系、长距离依赖。"
    "位置编码注入位置信息，原始论文用正弦余弦函数，后续有可学习位置编码、相对位置编码T5、"
    "旋转位置编码RoPE。RoPE对Q和K应用旋转矩阵编码相对位置，外推性好。"
    "每层包含多头注意力子层、残差连接、LayerNorm、前馈网络FFN。FFN通常是两层线性变换夹ReLU"
    "或GELU。现代变体用RMSNorm替代LayerNorm，用SwiGLU替代ReLU。残差连接解决梯度消失。"
    "GPT系列只用Decoder，BERT只用Encoder。深度学习革命始于AlexNet，随后VGG ResNet Inception"
    "改进。注意力机制最早在Bahdanau 2014用于机器翻译，Transformer 2017完全基于注意力。"
    "大语言模型GPT BERT T5 LLaMA都基于Transformer。训练用大规模文本语料自监督学习。"
    "Scaling law表明性能随参数量数据量计算量幂律提升。涌现能力在足够规模后出现。"
    "指令微调RLHF对齐人类偏好。"
)


def read_acceptance(base_url: str, headers: dict[str, str], spec_tokens: int) -> tuple[float, float]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/metrics",
        headers=headers,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        text = opener.open(request, timeout=10).read().decode()
    except Exception:
        return 0.0, 0.0
    accepted = drafted = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if "spec_decode_num_accepted_tokens_total" in line:
            accepted += float(line.split()[-1])
        elif "spec_decode_num_draft_tokens_total" in line:
            drafted += float(line.split()[-1])
    if drafted <= 0:
        return 0.0, 0.0
    rate = accepted / drafted
    return rate, 1.0 + rate * spec_tokens


async def one_request(session: aiohttp.ClientSession, url: str, payload: dict) -> tuple[float, float, int]:
    started = time.perf_counter()
    ttft = None
    last_token_at = None
    usage_tokens = 0
    chunks = 0
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=900)) as resp:
        async for raw in resp.content:
            line = raw.strip()
            if not line or line == b"data: [DONE]":
                continue
            if not line.startswith(b"data: "):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens", usage_tokens)
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - started
                last_token_at = time.perf_counter()
                chunks += 1
    tokens = usage_tokens or chunks
    decode_time = (last_token_at - started - ttft) if (ttft and last_token_at) else 0.0
    return ttft or 0.0, decode_time, tokens


async def run_level(base_url: str, model: str, prompt: str, max_tokens: int, concurrency: int,
                    headers: dict[str, str], tokens_per_step: float) -> None:
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
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        results = await asyncio.gather(
            *[one_request(session, url, payload) for _ in range(concurrency)],
            return_exceptions=True,
        )
    ok = [r for r in results if not isinstance(r, Exception)]
    if not ok:
        print(f"  concurrency {concurrency}: all requests failed ({results[0]})")
        return

    decode = statistics.median(r[1] for r in ok if r[1] > 0)
    tokens = statistics.median(r[2] for r in ok)
    ttft = statistics.median(r[0] for r in ok)
    forward_passes = (tokens - 1) / tokens_per_step if tokens > 1 and tokens_per_step > 0 else 1
    step_ms = decode * 1000 / forward_passes if forward_passes > 0 else 0
    single = (1000 / step_ms) * tokens_per_step if step_ms > 0 else 0
    aggregate = single * concurrency
    print(f"  concurrency {concurrency:>3}: step {step_ms:7.1f} ms | "
          f"stream {single:7.1f} tok/s | aggregate {aggregate:8.1f} tok/s | "
          f"TTFT {ttft * 1000:6.0f} ms")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("DSV41_BASE_URL", "http://127.0.0.1:8021"))
    parser.add_argument("--model", default=os.environ.get("DSV41_SERVED_NAME", "dsv41"))
    parser.add_argument("--api-key", default=(os.environ.get("DSV41_API_KEY") or os.environ.get("DSV41_W8A8_API_KEY")
                        or os.environ.get("API_KEY", "")))
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--prompt-repeat", type=int, default=11, help="≈8K tokens at repeat=11")
    parser.add_argument("--spec-tokens", type=int, default=5, help="DSpark K")
    args = parser.parse_args()

    prompt = PARAGRAPH * args.prompt_repeat
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    print(f"endpoint : {args.base_url}")
    print(f"model    : {args.model}")
    print("workload : prompt≈8K tokens, streaming, temperature=0, DSpark K=%d" % args.spec_tokens)

    print("\n=== warm-up (populate speculative-decoding metrics) ===")
    async with aiohttp.ClientSession(headers=headers) as session:
        await one_request(session, args.base_url.rstrip("/") + "/v1/completions", {
            "model": args.model, "prompt": prompt, "max_tokens": 64,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        })
    rate, per_step = read_acceptance(args.base_url, headers, args.spec_tokens)
    print(f"  acceptance rate = {rate * 100:.1f}%  tokens/step = {per_step:.2f}")

    for max_tokens in (512, 1024):
        print(f"\n=== 8K prompt → {max_tokens} output ===")
        for level in levels:
            await run_level(args.base_url, args.model, prompt, max_tokens, level, headers, per_step)
            await asyncio.sleep(3)
        rate, per_step = read_acceptance(args.base_url, headers, args.spec_tokens)
        print(f"  (updated) acceptance rate = {rate * 100:.1f}%  tokens/step = {per_step:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
