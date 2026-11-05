#!/usr/bin/env python3
"""Minimal deployment smoke test (standard library only).

Checks:
  1. GET /health returns 200
  2. GET /v1/models lists the served model
  3. POST /v1/chat/completions returns a non-empty answer (streaming off)

Usage:
    python3 tests/smoke_test.py
    python3 tests/smoke_test.py --base-url http://127.0.0.1:8021 \
        --model dsv41 --api-key "$DSV41_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url: str, headers: dict[str, str], payload: dict | None = None) -> tuple[int, bytes]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    try:
        with OPENER.open(req, timeout=600) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("DSV41_BASE_URL", "http://127.0.0.1:8021"))
    parser.add_argument("--model", default=os.environ.get("DSV41_SERVED_NAME", "dsv41"))
    parser.add_argument("--api-key", default=(os.environ.get("DSV41_API_KEY") or os.environ.get("DSV41_W8A8_API_KEY")
                        or os.environ.get("API_KEY", "")))
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    status, _ = request(base + "/health", headers)
    print(f"[1/3] /health              HTTP {status}")
    if status != 200:
        return 1

    status, body = request(base + "/v1/models", headers)
    models = []
    if status == 200:
        models = [item.get("id") for item in json.loads(body).get("data", [])]
    print(f"[2/3] /v1/models           HTTP {status} models={models}")
    if status != 200 or args.model not in models:
        print(f"      ⚠ served model '{args.model}' not in model list")
        return 1

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "用一句话说明什么是张量并行。"}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    status, body = request(base + "/v1/chat/completions", headers, payload)
    print(f"[3/3] /v1/chat/completions HTTP {status}")
    if status != 200:
        print(body[:500].decode(errors="replace"))
        return 1

    result = json.loads(body)
    message = result["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
    print(f"      reasoning chars={len(reasoning)} content chars={len(content)}")
    print(f"      content preview: {content[:120]}")
    if not content and not reasoning:
        print("      ❌ empty answer")
        return 1
    prefix = "[PASS]"
    if reasoning:
        prefix += " (reasoning parser active)"
    print(prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
