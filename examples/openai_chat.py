#!/usr/bin/env python3
"""OpenAI SDK chat example with streaming and reasoning output.

    pip install 'openai>=1.0'
    DSV41_API_KEY=... python3 examples/openai_chat.py "你好"
    DSV41_API_KEY=... python3 examples/openai_chat.py --stream "写一段快排"
"""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="用三句话解释什么是张量并行。")
    parser.add_argument("--base-url", default=os.environ.get("DSV41_BASE_URL", "http://127.0.0.1:8021"))
    parser.add_argument("--model", default=os.environ.get("DSV41_SERVED_NAME", "dsv41"))
    parser.add_argument("--api-key", default=os.environ.get("DSV41_API_KEY") or os.environ.get("API_KEY", ""))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url.rstrip("/") + "/v1", api_key=args.api_key or "EMPTY")
    kwargs = dict(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=0,
        stream=args.stream,
    )

    if not args.stream:
        message = client.chat.completions.create(**kwargs).choices[0].message
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        if reasoning:
            print("=== reasoning ===")
            print(reasoning)
        print("=== content ===")
        print(message.content)
        return 0

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if piece:
            reasoning_parts.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
        if delta.content:
            if reasoning_parts and not content_parts:
                print("\n=== content ===")
            content_parts.append(delta.content)
            sys.stdout.write(delta.content)
            sys.stdout.flush()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
