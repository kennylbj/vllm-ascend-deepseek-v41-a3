#!/usr/bin/env python3
"""End-to-end tests of the router's proxy path against in-process fake vLLM ranks.

No vLLM / NPU / network needed - only aiohttp, pyzmq and msgspec (skipped if absent).

Run:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import aiohttp
    from aiohttp import web
    import prefix_router
except ImportError as exc:  # pragma: no cover - optional host dependencies
    raise unittest.SkipTest(f"router dependencies unavailable: {exc}") from exc


class FakeRank:
    """Minimal stand-in for one vLLM API server (/health /metrics /tokenize /v1/completions)."""

    def __init__(self, tokens: int = 1000, waiting: int = 0, kv: float = 0.2) -> None:
        self.tokens = tokens
        self.waiting = waiting
        self.kv = kv
        self.completions = 0
        self.first_token_delay = 0.0
        self.close_next = 0          # drop the connection before answering, N times
        self.midstream_close = False  # send one chunk, then drop the connection
        self.runner: web.AppRunner | None = None
        self.url = ""

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_post("/tokenize", self._tokenize)
        app.router.add_post("/v1/completions", self._completions)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = "http://127.0.0.1:%d" % self.runner.addresses[0][1]

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()

    async def _health(self, request):
        return web.Response(text="ok")

    async def _metrics(self, request):
        text = ('vllm:num_requests_waiting{engine="0"} %s\n'
                'vllm:kv_cache_usage_perc{engine="0"} %s\n' % (float(self.waiting), self.kv))
        return web.Response(text=text, content_type="text/plain")

    async def _tokenize(self, request):
        return web.json_response({"tokens": list(range(self.tokens)), "count": self.tokens})

    async def _completions(self, request):
        self.completions += 1
        body = await request.json()
        if self.close_next > 0:
            self.close_next -= 1
            request.transport.close()
            return web.Response(status=500)
        if body.get("stream", True):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await asyncio.sleep(self.first_token_delay)
            await resp.write(b"data: first\n\n")
            if self.midstream_close:
                request.transport.close()
                return resp
            await asyncio.sleep(0.05)
            await resp.write(b"data: [DONE]\n\n")
            return resp
        await asyncio.sleep(self.first_token_delay)
        return web.json_response({"ok": True})


class Harness:
    def __init__(self, ranks: list, **router_args) -> None:
        self.ranks = ranks
        self.router_args = router_args
        self.runner: web.AppRunner | None = None
        self.router: prefix_router.Router | None = None
        self.base = ""

    async def __aenter__(self) -> "Harness":
        for r in self.ranks:
            await r.start()
        args = SimpleNamespace(
            upstream=[r.url for r in self.ranks], kv_events=[], block_size=128, max_skew=8,
            health_interval=0.1, stats_token=None, model="m", policy="cost",
            kv_high=0.90, sat_penalty=32768, prefill_weight=0.5, load_interval=0.1,
        )
        for k, v in self.router_args.items():
            setattr(args, k, v)
        self.router = prefix_router.Router(args)
        self.runner = web.AppRunner(prefix_router.build_app(self.router))
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        self.base = "http://127.0.0.1:%d" % self.runner.addresses[0][1]
        for _ in range(100):                        # wait for health + first load scrape
            if all(r.healthy and r.load_ts for r in self.router.ranks):
                break
            await asyncio.sleep(0.05)
        return self

    async def __aexit__(self, *exc) -> None:
        if self.runner:
            await self.runner.cleanup()
        for r in self.ranks:
            await r.stop()

    async def stats(self) -> dict:
        async with aiohttp.ClientSession() as s:
            async with s.get(self.base + "/router/stats") as resp:
                return await resp.json()


def run(coro) -> None:
    asyncio.run(coro)


BODY = {"model": "m", "prompt": "hello", "max_tokens": 8, "stream": True}


class ProxyTest(unittest.TestCase):
    def test_load_is_booked_until_first_byte_then_released(self) -> None:
        async def go() -> None:
            a, b = FakeRank(tokens=1000), FakeRank(tokens=1000)
            a.first_token_delay = 0.6
            async with Harness([a, b]) as h:
                async with aiohttp.ClientSession() as s:
                    async with s.post(h.base + "/v1/completions", json=BODY) as resp:
                        self.assertEqual(resp.status, 200)
                        st = await h.stats()                      # headers sent, no token yet
                        r0 = st["ranks"][0]
                        self.assertEqual((r0["active"], r0["inflight_tokens"], r0["pending_prefill"]),
                                         (1, 1000, 1000))
                        await resp.content.readany()              # first body bytes
                        r0 = (await h.stats())["ranks"][0]
                        self.assertEqual((r0["inflight_tokens"], r0["pending_prefill"]), (1000, 0))
                        await resp.read()
                await asyncio.sleep(0.1)
                r0 = (await h.stats())["ranks"][0]
                self.assertEqual((r0["active"], r0["inflight_tokens"], r0["pending_prefill"]), (0, 0, 0))
        run(go())

    def test_cold_request_avoids_queueing_rank_from_scraped_metrics(self) -> None:
        async def go() -> None:
            a, b = FakeRank(waiting=3, kv=0.95), FakeRank(waiting=0, kv=0.30)
            async with Harness([a, b]) as h:
                async with aiohttp.ClientSession() as s:
                    async with s.post(h.base + "/v1/completions", json=BODY) as resp:
                        await resp.read()
                st = await h.stats()
                self.assertEqual((a.completions, b.completions), (0, 1))
                self.assertEqual(st["sat_avoid"], 1)
                self.assertEqual(st["ranks"][0]["waiting"], 3)
                self.assertAlmostEqual(st["ranks"][0]["kv_usage"], 0.95)
                self.assertEqual(st["policy"], "cost")
        run(go())

    def test_stale_keepalive_is_retried_on_the_same_rank(self) -> None:
        async def go() -> None:
            a, b = FakeRank(), FakeRank()
            a.close_next = 1                       # first attempt: connection dropped
            async with Harness([a, b]) as h:
                async with aiohttp.ClientSession() as s:
                    async with s.post(h.base + "/v1/completions", json=BODY) as resp:
                        self.assertEqual(resp.status, 200)
                        await resp.read()
                self.assertEqual(a.completions, 2)  # dropped + retried
                self.assertEqual(b.completions, 0)  # prefix-holding rank kept; no re-prefill elsewhere
        run(go())

    def test_failure_after_response_started_is_not_replayed(self) -> None:
        async def go() -> None:
            a, b = FakeRank(), FakeRank()
            a.midstream_close = True
            async with Harness([a, b]) as h:
                async with aiohttp.ClientSession() as s:
                    try:
                        async with s.post(h.base + "/v1/completions", json=BODY) as resp:
                            await resp.read()
                    except aiohttp.ClientError:
                        pass                       # a truncated stream is acceptable
                await asyncio.sleep(0.2)
                self.assertEqual(a.completions, 1)
                self.assertEqual(b.completions, 0)  # no second generation elsewhere
                r0 = (await h.stats())["ranks"][0]
                self.assertEqual((r0["active"], r0["inflight_tokens"], r0["pending_prefill"]), (0, 0, 0))
        run(go())

    def test_client_leaving_does_not_trigger_a_second_generation(self) -> None:
        """Regression: after the client timed out, the failed response write was
        treated as an upstream error and the request was re-run on the other rank."""
        async def go() -> None:
            a, b = FakeRank(), FakeRank()
            a.first_token_delay = 0.8
            async with Harness([a, b]) as h:
                async with aiohttp.ClientSession() as s:
                    try:
                        async with s.post(h.base + "/v1/completions",
                                          json=dict(BODY, stream=False),
                                          timeout=aiohttp.ClientTimeout(total=0.3)) as resp:
                            await resp.read()
                    except asyncio.TimeoutError:
                        pass
                await asyncio.sleep(1.5)           # let the upstream answer into the void
                self.assertEqual(a.completions, 1)
                self.assertEqual(b.completions, 0)
                st = await h.stats()
                self.assertEqual([(r["active"], r["inflight_tokens"], r["pending_prefill"])
                                  for r in st["ranks"]], [(0, 0, 0), (0, 0, 0)])
        run(go())

    def test_stats_expose_new_fields(self) -> None:
        async def go() -> None:
            async with Harness([FakeRank(), FakeRank()]) as h:
                st = await h.stats()
                for key in ("policy", "spills", "sat_avoid", "digest_evicted", "digest_resets",
                            "requests", "affinity_hits", "fallback", "events"):
                    self.assertIn(key, st)
                for key in ("inflight_tokens", "pending_prefill", "kv_usage", "waiting",
                            "cached_prefix_digests"):
                    self.assertIn(key, st["ranks"][0])
        run(go())


if __name__ == "__main__":
    unittest.main()
