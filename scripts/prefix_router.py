#!/usr/bin/env python3
"""Prefix/KV-aware router for an externally load-balanced vLLM DP deployment.

Architecture
------------
    client -> this router (:8021) -> vLLM API rank0 (:8022)
                                   -> vLLM API rank1 (:8023)

Each vLLM rank publishes KV cache events over ZMQ. The router tracks the
sequence of prefix blocks cached by each rank and scores every rank for each
request:

    score(rank) = uncached prompt tokens on that rank
                  + prefill_weight * prompt tokens already queued for prefill there
                  + sat_penalty    if the rank is KV-saturated or has a queue

and sends the request to the lowest score (policy ``cost``, the default).
Cached prefix, prefill backlog and KV pressure are all expressed in tokens, so
a request is only pulled away from the rank holding its prefix when the saving
is smaller than the cost of waiting there. The previous request-count policy
(prefix affinity unless the active-request gap exceeds --max-skew) is kept as
``--policy legacy``.

Why tokens and not request counts: with DP+EP the two engines step in lockstep,
a 16K-token prefill chunk on either rank stalls decode on both, and KV capacity
(not --max-num-seqs) limits concurrency. Two ranks can hold the same number of
requests while one is KV-saturated and queueing.

The router is OpenAI-compatible: it transparently proxies chat/completions,
completions, models, metrics and streaming SSE responses. It does not require
duplicated model output logic.

Dependencies (host Python):
    aiohttp, pyzmq, msgspec

Usage:
    python3 scripts/prefix_router.py \
        --listen 8021 \
        --upstream http://127.0.0.1:8022 \
        --upstream http://127.0.0.1:8023 \
        --kv-events tcp://127.0.0.1:5557 \
        --kv-events tcp://127.0.0.1:5558 \
        --model dsv41
"""

from __future__ import annotations

import argparse
import array
import asyncio
import hashlib
import itertools
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import msgspec
import zmq
import zmq.asyncio
from aiohttp import web

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
COMPLETION_PATHS = {"/v1/chat/completions", "/v1/completions"}
BLOCK_SIZE = 128
# Upper bound on LIVE prefix digests per rank. A rank normally holds a few
# thousand (KV pool / block size); the bound is a safety net against leaked
# digests (lost BlockRemoved events), so overflow evicts the oldest entries
# instead of forgetting everything.
MAX_DIGESTS = 200_000
EVICT_FRACTION = 0.10
MAX_STICKY = 50_000
MAX_TOKEN_CACHE_ENTRIES = 1_024
MAX_TOKEN_CACHE_TOKENS = 2_000_000
MAX_CACHED_PROMPT_TOKENS = 16_384
LOAD_STALE_S = 10.0          # ignore scraped load older than this

# Idle keep-alive connections are closed by uvicorn after 5 s by default; keep
# the router's pool below that or a request can race a closing socket and fail
# with "Broken pipe" (seen in production).
KEEPALIVE_TIMEOUT_S = 4

# Errors that mean "this pooled connection was already dead" - safe to retry on
# the same rank once before failing over (nothing was sent to the engine).
STALE_CONNECTION_ERRORS = (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) + tuple(
    getattr(aiohttp, name) for name in ("ClientConnectionResetError",) if hasattr(aiohttp, name))

logger = logging.getLogger("prefix_router")

_METRIC_LINE = re.compile(
    r"^(vllm:kv_cache_usage_perc|vllm:num_requests_waiting|vllm:num_requests_running)"
    r"(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN|nan)\s*$"
)


def block_digest(previous: bytes, tokens: list[int]) -> bytes:
    payload = array.array("I", tokens).tobytes()
    return hashlib.blake2b(previous + payload, digest_size=16).digest()


def prefix_digests(tokens: list[int], block_size: int = BLOCK_SIZE) -> list[bytes]:
    out: list[bytes] = []
    previous = b""
    blocks = len(tokens) // block_size
    for i in range(blocks):
        previous = block_digest(previous, tokens[i * block_size:(i + 1) * block_size])
        out.append(previous)
    return out


def parse_load_metrics(text: str) -> dict[str, float]:
    """Pull KV usage (0-1), queue depth and running count out of Prometheus text.

    The /metrics of one API server only describes its own engine, but if several
    series are present the largest value wins (the conservative reading).
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name, raw = m.group(1), m.group(2)
        try:
            value = float(raw)
        except ValueError:
            continue
        if value != value:  # NaN
            continue
        key = {"vllm:kv_cache_usage_perc": "kv_usage",
               "vllm:num_requests_waiting": "waiting",
               "vllm:num_requests_running": "running"}[name]
        if key == "kv_usage" and value > 1.5:  # reported as a percentage
            value /= 100.0
        out[key] = max(out.get(key, value), value)
    return out


@dataclass
class Rank:
    index: int
    url: str
    active: int = 0
    healthy: bool = False  # set by the first /health probe
    # Token-level load, maintained by the router itself (instant, no scrape lag).
    inflight_tokens: int = 0     # prompt tokens of requests currently on this rank
    pending_prefill: int = 0     # uncached prompt tokens not yet at first response byte
    # Scraped from the rank's /metrics (None until the first successful scrape).
    kv_usage: float | None = None
    waiting: int | None = None
    load_ts: float = 0.0
    # Insertion-ordered set of live prefix digests (dict preserves order).
    digests: dict[bytes, None] = field(default_factory=dict)
    digest_counts: dict[bytes, int] = field(default_factory=dict)
    hash_to_digest: dict[str, bytes] = field(default_factory=dict)
    evicted: int = 0   # digests dropped by the MAX_DIGESTS safety bound
    resets: int = 0    # AllBlocksCleared events (vLLM reset its prefix cache)

    def add_digest(self, digest: bytes) -> None:
        # The same block hash is published once per KV-cache group. Count
        # references so BlockRemoved for one group does not hide the prefix
        # while other groups still hold it.
        self.digest_counts[digest] = self.digest_counts.get(digest, 0) + 1
        if digest in self.digests:
            return
        self.digests[digest] = None
        if len(self.digests) > MAX_DIGESTS:
            self._evict_oldest()

    def remove_digest(self, digest: bytes) -> None:
        count = self.digest_counts.get(digest, 0) - 1
        if count > 0:
            self.digest_counts[digest] = count
            return
        self.digest_counts.pop(digest, None)
        self.digests.pop(digest, None)

    def _evict_oldest(self) -> None:
        n = max(1, int(MAX_DIGESTS * EVICT_FRACTION))
        for digest in list(itertools.islice(self.digests, n)):
            del self.digests[digest]
            self.digest_counts.pop(digest, None)
        self.evicted += n
        logger.warning("rank %d: digest table over %d live entries, evicted %d oldest "
                       "(leaked BlockRemoved events?)", self.index, MAX_DIGESTS, n)

    def prune_hash_map(self) -> None:
        """Drop hash -> digest entries whose digest is no longer tracked."""
        self.hash_to_digest = {h: d for h, d in self.hash_to_digest.items() if d in self.digests}

    def clear(self) -> None:
        self.digests.clear()
        self.digest_counts.clear()
        self.hash_to_digest.clear()

    def reset(self) -> None:
        """vLLM cleared its whole prefix cache (AllBlocksCleared)."""
        self.clear()
        self.resets += 1

    def matched_blocks(self, digests: list[bytes]) -> int:
        matched = 0
        for digest in digests:
            if digest not in self.digests:
                break
            matched += 1
        return matched

    def saturated(self, kv_high: float, now: float) -> bool:
        """Scraped load says new work would queue here (fresh data only)."""
        if not self.load_ts or now - self.load_ts > LOAD_STALE_S:
            return False
        if self.waiting is not None and self.waiting > 0:
            return True
        return self.kv_usage is not None and self.kv_usage >= kv_high


@dataclass
class Decision:
    """Where a request goes, plus the load it adds to whichever rank serves it."""
    rank: int
    prompt_tokens: int = 0
    uncached: list[int] = field(default_factory=list)  # per rank

    def uncached_for(self, idx: int) -> int:
        return self.uncached[idx] if idx < len(self.uncached) else 0


class _Load:
    """Books one request's load on a rank; released exactly once."""

    __slots__ = ("rank", "tokens", "pending", "released")

    def __init__(self, rank: Rank, tokens: int, pending: int) -> None:
        self.rank = rank
        self.tokens = tokens
        self.pending = pending
        self.released = False
        rank.active += 1
        rank.inflight_tokens += tokens
        rank.pending_prefill += pending

    def first_byte(self) -> None:
        """Prefill is over once the engine produces output."""
        if self.pending:
            self.rank.pending_prefill -= self.pending
            self.pending = 0

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.first_byte()
        self.rank.active -= 1
        self.rank.inflight_tokens -= self.tokens


class Router:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.block_size = args.block_size
        self.ranks = [Rank(i, url.rstrip("/")) for i, url in enumerate(args.upstream)]
        if len(self.ranks) < 2:
            raise SystemExit("at least two --upstream endpoints are required")
        self.policy = getattr(args, "policy", "cost")
        if self.policy not in ("cost", "legacy"):
            raise SystemExit("--policy must be 'cost' or 'legacy'")
        self.kv_high = float(getattr(args, "kv_high", 0.90))
        self.sat_penalty = float(getattr(args, "sat_penalty", 32768))
        self.prefill_weight = float(getattr(args, "prefill_weight", 0.5))
        self.load_interval = float(getattr(args, "load_interval", 2.0))
        self.session: aiohttp.ClientSession | None = None
        self.zmq_ctx: zmq.asyncio.Context | None = None
        self.sticky: OrderedDict[bytes, tuple[int, float]] = OrderedDict()
        self.token_cache: OrderedDict[bytes, list[int]] = OrderedDict()
        self.token_cache_tokens = 0
        self._tok_rr = 0
        self._tasks: list[asyncio.Task] = []
        self._load_warned = False
        self.stats = {"requests": 0, "affinity_hits": 0, "fallback": 0, "events": 0,
                      "spills": 0, "sat_avoid": 0}

    async def start(self) -> None:
        connector = aiohttp.TCPConnector(
            limit=256, limit_per_host=128, keepalive_timeout=KEEPALIVE_TIMEOUT_S,
            enable_cleanup_closed=True, ttl_dns_cache=300,
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=3600),
            trust_env=False,
            auto_decompress=False,
        )
        self.zmq_ctx = zmq.asyncio.Context.instance()
        for rank, endpoint in zip(self.ranks, self.args.kv_events or []):
            self._tasks.append(asyncio.create_task(
                self._kv_event_loop(rank, endpoint), name="kv-events-%d" % rank.index))
        self._tasks.append(asyncio.create_task(self._health_loop(), name="health"))
        if self.load_interval > 0:
            self._tasks.append(asyncio.create_task(self._load_loop(), name="load"))
        logger.info("routing policy=%s kv_high=%.2f sat_penalty=%d prefill_weight=%.2f "
                    "load_interval=%.1fs max_skew=%s",
                    self.policy, self.kv_high, self.sat_penalty, self.prefill_weight,
                    self.load_interval, getattr(self.args, "max_skew", 8))

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()
        if self.session:
            await self.session.close()

    # ---- KV cache events -------------------------------------------------
    async def _kv_event_loop(self, rank: Rank, endpoint: str) -> None:
        assert self.zmq_ctx is not None
        socket = self.zmq_ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVHWM, 100_000)
        socket.connect(endpoint)
        logger.info("rank %d subscribing KV events at %s", rank.index, endpoint)
        try:
            while True:
                frames = await socket.recv_multipart()
                if len(frames) < 3:
                    continue
                try:
                    batch = msgspec.msgpack.decode(frames[-1])
                except Exception as exc:  # noqa: BLE001
                    logger.debug("bad KV event payload: %s", exc)
                    continue
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                for event in batch[1] or []:
                    self._handle_event(rank, event)
                    self.stats["events"] += 1
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("KV event loop for rank %d stopped: %s", rank.index, exc)
        finally:
            socket.close(linger=0)

    def _handle_event(self, rank: Rank, event: Any) -> None:
        if isinstance(event, dict):
            kind = event.get("type")
        else:
            return
        if kind == "AllBlocksCleared":
            rank.reset()
            return
        if kind == "BlockRemoved":
            for block_hash in event.get("block_hashes") or []:
                digest = rank.hash_to_digest.get(block_hash)
                if digest is None:
                    continue
                rank.remove_digest(digest)
                # Keep the hash -> digest mapping until every cache group has
                # removed it; otherwise the remaining group events cannot
                # decrement the reference count.
                if rank.digest_counts.get(digest, 0) <= 0:
                    rank.hash_to_digest.pop(block_hash, None)
            return
        if kind != "BlockStored":
            return

        hashes = event.get("block_hashes") or []
        tokens = event.get("token_ids") or []
        block_size = int(event.get("block_size") or self.block_size)
        parent = event.get("parent_block_hash")
        previous = rank.hash_to_digest.get(parent, b"") if parent else b""
        if not hashes:
            return
        chunks = [tokens[i:i + block_size] for i in range(0, len(tokens), block_size)]
        # Null/masked blocks can make lengths differ; pair what we can safely.
        pair_count = min(len(hashes), len(chunks))
        for block_hash, chunk in zip(hashes[:pair_count], chunks[:pair_count]):
            if len(chunk) != block_size:
                break
            previous = block_digest(previous, chunk)
            rank.hash_to_digest[block_hash] = previous
            rank.add_digest(previous)
        # Bound the hash table as well as the digest set.
        if len(rank.hash_to_digest) > MAX_DIGESTS * 2:
            rank.prune_hash_map()

    # ---- health / load ---------------------------------------------------
    async def _health_loop(self) -> None:
        while True:
            await asyncio.gather(*(self._check_health(rank) for rank in self.ranks))
            await asyncio.sleep(self.args.health_interval)

    async def _check_health(self, rank: Rank) -> None:
        assert self.session is not None
        try:
            async with self.session.get(rank.url + "/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                rank.healthy = resp.status == 200
        except Exception:  # noqa: BLE001
            rank.healthy = False

    async def _load_loop(self) -> None:
        while True:
            await asyncio.gather(*(self._poll_load(rank) for rank in self.ranks))
            await asyncio.sleep(self.load_interval)

    async def _poll_load(self, rank: Rank) -> None:
        """Scrape KV usage and queue depth; failures just leave the data to go stale."""
        assert self.session is not None
        try:
            async with self.session.get(rank.url + "/metrics",
                                        timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status != 200:
                    return
                text = await resp.text()
        except Exception:  # noqa: BLE001
            return
        load = parse_load_metrics(text)
        if "kv_usage" not in load and "waiting" not in load:
            if not self._load_warned:
                self._load_warned = True
                logger.warning("rank %d /metrics has no vllm:kv_cache_usage_perc / "
                               "vllm:num_requests_waiting; load-based avoidance is inactive",
                               rank.index)
            return
        rank.kv_usage = load.get("kv_usage")
        rank.waiting = int(load["waiting"]) if "waiting" in load else None
        rank.load_ts = time.monotonic()

    # ---- tokenization / routing -----------------------------------------
    def _tokenize_payload(self, body: dict[str, Any]) -> dict[str, Any] | None:
        payload: dict[str, Any] = {}
        if body.get("model"):
            payload["model"] = body["model"]
        if "messages" in body:
            payload["messages"] = body["messages"]
            for key in ("tools", "chat_template", "chat_template_kwargs",
                        "continue_final_message", "add_generation_prompt"):
                if key in body:
                    payload[key] = body[key]
        elif "prompt" in body:
            payload["prompt"] = body["prompt"]
            if "add_special_tokens" in body:
                payload["add_special_tokens"] = body["add_special_tokens"]
        else:
            return None
        return payload

    @staticmethod
    def _body_key(payload: dict[str, Any]) -> bytes:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")).encode()
        return hashlib.blake2b(canonical, digest_size=16).digest()

    def _tokenize_targets(self) -> list[Rank]:
        """Healthy ranks in round-robin order so /tokenize load is shared."""
        healthy = [r for r in self.ranks if r.healthy] or list(self.ranks)
        start = self._tok_rr % len(healthy)
        self._tok_rr += 1
        return healthy[start:] + healthy[:start]

    async def _tokenize(self, payload: dict[str, Any], headers: dict[str, str],
                        key: bytes | None = None) -> list[int] | None:
        if key is None:
            key = self._body_key(payload)
        cached = self.token_cache.get(key)
        if cached is not None:
            self.token_cache.move_to_end(key)
            return cached
        assert self.session is not None
        tokens = None
        for rank in self._tokenize_targets()[:2]:   # one failover attempt
            try:
                async with self.session.post(rank.url + "/tokenize", json=payload, headers=headers,
                                             timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
            except Exception:  # noqa: BLE001
                continue
            tokens = data.get("tokens")
            if isinstance(tokens, list):
                break
            tokens = None
        if tokens is None:
            return None
        # Never cache huge prompts: a Python list of 1M ints costs ~36 MB and
        # would quickly exhaust the router process.
        if len(tokens) <= MAX_CACHED_PROMPT_TOKENS:
            old = self.token_cache.pop(key, None)
            if old is not None:
                self.token_cache_tokens -= len(old)
            self.token_cache[key] = tokens
            self.token_cache_tokens += len(tokens)
            while (len(self.token_cache) > MAX_TOKEN_CACHE_ENTRIES
                   or self.token_cache_tokens > MAX_TOKEN_CACHE_TOKENS):
                _, old = self.token_cache.popitem(last=False)
                self.token_cache_tokens -= len(old)
        return tokens

    def _candidates(self) -> list[Rank]:
        return [r for r in self.ranks if r.healthy] or list(self.ranks)

    def _least_active(self) -> int:
        return min(self._candidates(), key=lambda r: (r.active, r.index)).index

    def _choose_rank(self, body_key: bytes, matched: list[int], prompt_tokens: int = 0) -> int:
        if self.policy == "legacy":
            return self._choose_legacy(body_key, matched)
        return self._choose_cost(matched, prompt_tokens)

    def _choose_cost(self, matched: list[int], prompt_tokens: int) -> int:
        now = time.monotonic()
        candidates = self._candidates()
        best = max(matched) if matched else 0
        # A match can never exceed the prompt; trust the larger of the two.
        total = max(prompt_tokens, best * self.block_size)

        def score(rank: Rank, with_saturation: bool) -> tuple[float, int, int]:
            uncached = max(total - matched[rank.index] * self.block_size, 0)
            s = uncached + self.prefill_weight * rank.pending_prefill
            if with_saturation and rank.saturated(self.kv_high, now):
                s += self.sat_penalty
            return (s, rank.active, rank.index)

        choice = min(candidates, key=lambda r: score(r, True))
        if choice is not min(candidates, key=lambda r: score(r, False)):
            self.stats["sat_avoid"] += 1
        if best > 0:
            if matched[choice.index] == best:
                self.stats["affinity_hits"] += 1
            else:
                self.stats["spills"] += 1
        else:
            self.stats["fallback"] += 1
        return choice.index

    def _choose_legacy(self, body_key: bytes, matched: list[int]) -> int:
        """Previous policy: longest prefix wins unless the request-count gap exceeds --max-skew."""
        best = max(matched) if matched else 0
        candidates = self._candidates()
        ranked = [r.index for r in candidates if matched[r.index] == best] if best > 0 else []
        if ranked:
            choice = min(ranked, key=lambda i: self.ranks[i].active)
            self.stats["affinity_hits"] += 1
        else:
            sticky = self.sticky.get(body_key)
            if sticky and self.ranks[sticky[0]].healthy:
                choice = sticky[0]
            else:
                choice = min(candidates, key=lambda r: r.active).index
                self.stats["fallback"] += 1
        # Avoid dumping a hot prefix onto one rank forever.
        other = min((r for r in candidates if r.index != choice),
                    key=lambda r: r.active, default=None)
        max_skew = getattr(self.args, "max_skew", 8)
        if other is not None and self.ranks[choice].active > other.active + max_skew:
            choice = other.index
            self.stats["fallback"] += 1
        self.sticky[body_key] = (choice, time.monotonic())
        self.sticky.move_to_end(body_key)
        while len(self.sticky) > MAX_STICKY:
            self.sticky.popitem(last=False)
        return choice

    async def route(self, request: web.Request, body: bytes) -> Decision:
        n = len(self.ranks)
        try:
            data = json.loads(body)
        except Exception:  # noqa: BLE001
            return Decision(self._least_active())
        if not isinstance(data, dict):
            return Decision(self._least_active())
        payload = self._tokenize_payload(data)
        if payload is None:
            return Decision(self._least_active())
        key = self._body_key(payload)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        tokens = await self._tokenize(payload, headers, key)
        matched = [0] * n
        prompt_tokens = 0
        if tokens:
            prompt_tokens = len(tokens)
            digests = prefix_digests(tokens, self.block_size)
            matched = [rank.matched_blocks(digests) for rank in self.ranks]
        rank_index = self._choose_rank(key, matched, prompt_tokens)
        uncached = [max(prompt_tokens - m * self.block_size, 0) for m in matched]
        return Decision(rank_index, prompt_tokens, uncached)

    # ---- proxies ---------------------------------------------------------
    async def proxy(self, request: web.Request, forced_rank: int | None = None) -> web.StreamResponse:
        body = await request.read()
        if forced_rank is not None:
            decision = Decision(forced_rank)
        elif request.path in COMPLETION_PATHS and request.method == "POST":
            decision = await self.route(request, body)
        else:
            decision = Decision(0)
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP]
        order = [decision.rank] + [r.index for r in self.ranks if r.index != decision.rank]
        last_error: Exception | None = None
        for idx in order:
            rank = self.ranks[idx]
            if not rank.healthy and len(order) > 1:
                continue
            self.stats["requests"] += 1
            for attempt in (0, 1):
                load = _Load(rank, decision.prompt_tokens, decision.uncached_for(idx))
                try:
                    return await self._forward(request, body, rank, headers, load)
                except STALE_CONNECTION_ERRORS as exc:
                    last_error = exc
                    if attempt == 0:
                        logger.warning("upstream rank %d stale connection (%s); retrying same rank",
                                       idx, exc)
                        continue
                    logger.warning("upstream rank %d failed: %s", idx, exc)
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    last_error = exc
                    logger.warning("upstream rank %d failed: %s", idx, exc)
                    break
                finally:
                    load.release()
        return web.Response(status=502, text="all upstreams failed: %s" % last_error)

    async def _forward(self, request: web.Request, body: bytes, rank: Rank,
                       headers: list[tuple[str, str]], load: _Load) -> web.StreamResponse:
        """Proxy one request.

        Exceptions escape only BEFORE any byte reached the client, which is what
        makes failing over to another rank safe. Once the response has started,
        errors end the stream instead of re-running the request elsewhere.
        """
        assert self.session is not None
        url = rank.url + request.rel_url.raw_path_qs
        upstream = await self.session.request(
            request.method, url, headers=headers, data=body if body else None,
        )
        out = web.StreamResponse(status=upstream.status, reason=upstream.reason)
        for key, value in upstream.headers.items():
            if key.lower() not in HOP_BY_HOP:
                out.headers[key] = value
        if "text/event-stream" in upstream.headers.get("Content-Type", ""):
            out.headers.setdefault("Cache-Control", "no-cache")
        try:
            await out.prepare(request)
        except ConnectionResetError:
            # The client is already gone; do not replay the request on another rank.
            upstream.close()
            return out
        try:
            async for chunk in upstream.content.iter_any():
                load.first_byte()
                await out.write(chunk)
        except ConnectionResetError:
            pass  # client went away; closing upstream below aborts the request
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("upstream rank %d dropped mid-stream: %s", rank.index, exc)
        finally:
            upstream.close()
        try:
            await out.write_eof()
        except Exception:  # noqa: BLE001
            pass
        return out

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/health":
            healthy = all(rank.healthy for rank in self.ranks)
            return web.Response(status=200 if healthy else 503,
                                text="ok" if healthy else "not ready")
        if request.path == "/router/stats":
            if self.args.stats_token:
                auth = request.headers.get("Authorization", "")
                if auth != "Bearer %s" % self.args.stats_token:
                    return web.Response(status=401, text="unauthorized")
            return web.json_response(self.stats_snapshot())
        if request.path == "/metrics":
            return await self._aggregate_metrics()
        return await self.proxy(request)

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "ranks": [{"url": r.url, "active": r.active, "healthy": r.healthy,
                       "cached_prefix_digests": len(r.digests),
                       "inflight_tokens": r.inflight_tokens,
                       "pending_prefill": r.pending_prefill,
                       "kv_usage": r.kv_usage, "waiting": r.waiting} for r in self.ranks],
            "digest_evicted": sum(r.evicted for r in self.ranks),
            "digest_resets": sum(r.resets for r in self.ranks),
            **self.stats,
        }

    async def _aggregate_metrics(self) -> web.Response:
        assert self.session is not None
        texts: list[str] = []
        for rank in self.ranks:
            try:
                async with self.session.get(rank.url + "/metrics",
                                            timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        texts.append(await resp.text())
            except Exception:  # noqa: BLE001
                pass
        if not texts:
            return web.Response(status=503, text="metrics unavailable")
        seen_meta: set[str] = set()
        lines: list[str] = []
        for text in texts:
            for line in text.splitlines():
                if line.startswith("# HELP") or line.startswith("# TYPE"):
                    if line in seen_meta:
                        continue
                    seen_meta.add(line)
                lines.append(line)
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def build_app(router: Router) -> web.Application:
    async def startup(app: web.Application) -> None:
        await router.start()

    async def cleanup(app: web.Application) -> None:
        await router.close()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_route("*", "/{tail:.*}", router.handle)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", type=int, default=8021)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--upstream", action="append", required=True,
                        help="vLLM rank endpoint, repeat for each rank")
    parser.add_argument("--kv-events", action="append", default=[],
                        help="ZMQ KV event endpoint per rank, same order as --upstream")
    parser.add_argument("--model", default=os.environ.get("DSV41_SERVED_NAME", ""))
    parser.add_argument("--policy", choices=("cost", "legacy"),
                        default=os.environ.get("DSV41_ROUTER_POLICY", "cost"),
                        help="cost: token-level score (cached prefix, prefill backlog, KV saturation); "
                             "legacy: longest prefix unless the active-request gap exceeds --max-skew")
    parser.add_argument("--kv-high", type=float,
                        default=float(os.environ.get("DSV41_ROUTER_KV_HIGH", "0.90")),
                        help="cost policy: a rank at or above this KV usage (0-1) counts as saturated")
    parser.add_argument("--sat-penalty", type=float,
                        default=float(os.environ.get("DSV41_ROUTER_SAT_PENALTY", "32768")),
                        help="cost policy: tokens added to the score of a saturated or queueing rank")
    parser.add_argument("--prefill-weight", type=float,
                        default=float(os.environ.get("DSV41_ROUTER_PREFILL_WEIGHT", "0.5")),
                        help="cost policy: weight of prompt tokens already queued for prefill on a rank")
    parser.add_argument("--load-interval", type=float,
                        default=float(os.environ.get("DSV41_ROUTER_LOAD_INTERVAL", "2.0")),
                        help="seconds between /metrics scrapes of each rank (0 disables)")
    parser.add_argument("--max-skew", type=int, default=8,
                        help="legacy policy: break prefix affinity when one rank has this many "
                             "more active requests")
    parser.add_argument("--block-size", type=int, default=128,
                        help="KV cache block size; must match vLLM --block-size")
    parser.add_argument("--stats-token", default=None,
                        help="require this bearer token for /router/stats "
                             "(default: $DSV41_ROUTER_STATS_TOKEN or $DSV41_API_KEY)")
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    args.stats_token = (args.stats_token
                        or os.environ.get("DSV41_ROUTER_STATS_TOKEN")
                        or os.environ.get("DSV41_API_KEY"))

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    logger.info("router listening on %s:%d, upstreams=%s, kv_events=%s",
                args.host, args.listen, args.upstream, args.kv_events)
    router = Router(args)
    web.run_app(build_app(router), host=args.host, port=args.listen, access_log=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
