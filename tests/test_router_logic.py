#!/usr/bin/env python3
"""Unit tests for the prefix/KV-aware router (no network required).

Run:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import pathlib
import random
import sys
import time
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import msgspec
    import prefix_router
except ImportError as exc:  # pragma: no cover - optional host dependencies
    raise unittest.SkipTest(f"router dependencies unavailable: {exc}") from exc


def make_router(policy: str = "cost", max_skew: int = 8, **extra) -> "prefix_router.Router":
    args = SimpleNamespace(
        upstream=["http://127.0.0.1:18022", "http://127.0.0.1:18023"],
        block_size=128,
        max_skew=max_skew,
        policy=policy,
        **extra,
    )
    router = prefix_router.Router(args)
    for rank in router.ranks:
        rank.healthy = True
    return router


def toks(seed: int, n_blocks: int, block: int = 128) -> list:
    return [(seed * 1_000_003 + i * 7919) % 120_000 + 10 for i in range(n_blocks * block)]


def chain_hashes(tokens: list, block: int, tag: bytes = b"t") -> list:
    """Stand-in for vLLM's chained, opaque external block hashes."""
    out, prev = [], b""
    for i in range(len(tokens) // block):
        prev = hashlib.sha256(prev + tag + repr(tokens[i * block:(i + 1) * block]).encode()).digest()
        out.append(prev)
    return out


class DigestTest(unittest.TestCase):
    def test_block_boundaries(self) -> None:
        tokens = list(range(128 * 3))
        digests = prefix_router.prefix_digests(tokens, 128)
        self.assertEqual(len(digests), 3)
        self.assertEqual(len(set(digests)), 3)
        # Changing a token before the last block must change that block and all
        # following digests, but not earlier ones.
        changed = tokens.copy()
        changed[0] = 999
        changed_digests = prefix_router.prefix_digests(changed, 128)
        self.assertNotEqual(digests[0], changed_digests[0])
        self.assertNotEqual(digests[1], changed_digests[1])

    def test_short_prompt_has_no_full_block(self) -> None:
        self.assertEqual(prefix_router.prefix_digests([1, 2, 3], 128), [])


class RankTest(unittest.TestCase):
    def test_longest_prefix_stops_at_first_miss(self) -> None:
        rank = prefix_router.Rank(0, "http://127.0.0.1:1")
        digests = [b"a", b"b", b"c"]
        for d in digests:
            rank.add_digest(d)
        self.assertEqual(rank.matched_blocks(digests), 3)
        self.assertEqual(rank.matched_blocks([b"a", b"x", b"c"]), 1)

    def test_refcount_keeps_digest_until_all_groups_removed(self) -> None:
        rank = prefix_router.Rank(0, "http://127.0.0.1:1")
        digest = b"x" * 16
        # vLLM publishes the same block once per KV-cache group.
        rank.add_digest(digest)
        rank.add_digest(digest)
        rank.remove_digest(digest)
        self.assertIn(digest, rank.digests)
        rank.remove_digest(digest)
        self.assertNotIn(digest, rank.digests)

    def test_churn_does_not_wipe_live_prefix(self) -> None:
        """Regression: the bound used to count cumulative inserts, so ~200k
        store/evict cycles wiped a rank even with only 40 live digests."""
        old = prefix_router.MAX_DIGESTS
        prefix_router.MAX_DIGESTS = 100
        try:
            rank = prefix_router.Rank(0, "http://127.0.0.1:1")
            live = [b"live-%03d" % i for i in range(40)]
            for d in live:
                rank.add_digest(d)
            for i in range(5_000):          # 50x the bound in cumulative inserts
                d = b"churn-%05d" % i
                rank.add_digest(d)
                rank.remove_digest(d)
            self.assertEqual(rank.evicted, 0)
            self.assertEqual(rank.matched_blocks(live), 40)
        finally:
            prefix_router.MAX_DIGESTS = old

    def test_overflow_evicts_oldest_not_everything(self) -> None:
        old = prefix_router.MAX_DIGESTS
        prefix_router.MAX_DIGESTS = 100
        try:
            rank = prefix_router.Rank(0, "http://127.0.0.1:1")
            for i in range(101):            # one over the bound
                rank.add_digest(b"d-%03d" % i)
            self.assertEqual(rank.evicted, 10)
            self.assertEqual(len(rank.digests), 91)
            self.assertNotIn(b"d-000", rank.digests)   # oldest gone
            self.assertIn(b"d-100", rank.digests)      # newest kept
            self.assertEqual(rank.resets, 0)
        finally:
            prefix_router.MAX_DIGESTS = old

    def test_all_blocks_cleared_counts_reset(self) -> None:
        router = make_router()
        rank = router.ranks[0]
        rank.add_digest(b"a")
        router._handle_event(rank, {"type": "AllBlocksCleared"})
        self.assertEqual(len(rank.digests), 0)
        self.assertEqual(rank.resets, 1)

    def test_prune_hash_map_drops_dead_entries(self) -> None:
        rank = prefix_router.Rank(0, "http://127.0.0.1:1")
        rank.add_digest(b"keep")
        rank.hash_to_digest = {"h1": b"keep", "h2": b"gone"}
        rank.prune_hash_map()
        self.assertEqual(rank.hash_to_digest, {"h1": b"keep"})

    def test_saturated_needs_fresh_data(self) -> None:
        rank = prefix_router.Rank(0, "http://127.0.0.1:1")
        now = time.monotonic()
        self.assertFalse(rank.saturated(0.9, now))                 # never scraped
        rank.kv_usage, rank.waiting, rank.load_ts = 0.5, 0, now
        self.assertFalse(rank.saturated(0.9, now))
        rank.waiting = 2
        self.assertTrue(rank.saturated(0.9, now))                  # queueing
        rank.waiting, rank.kv_usage = 0, 0.95
        self.assertTrue(rank.saturated(0.9, now))                  # KV nearly full
        rank.load_ts = now - prefix_router.LOAD_STALE_S - 1
        self.assertFalse(rank.saturated(0.9, now))                 # stale -> ignored


class LoadAccountingTest(unittest.TestCase):
    def test_load_lifecycle(self) -> None:
        rank = prefix_router.Rank(0, "http://127.0.0.1:1")
        load = prefix_router._Load(rank, tokens=80_000, pending=30_000)
        self.assertEqual((rank.active, rank.inflight_tokens, rank.pending_prefill), (1, 80_000, 30_000))
        load.first_byte()                       # prefill finished
        self.assertEqual((rank.inflight_tokens, rank.pending_prefill), (80_000, 0))
        load.first_byte()                       # idempotent
        load.release()
        load.release()                          # idempotent
        self.assertEqual((rank.active, rank.inflight_tokens, rank.pending_prefill), (0, 0, 0))

    def test_release_without_first_byte_returns_pending(self) -> None:
        rank = prefix_router.Rank(0, "http://127.0.0.1:1")
        prefix_router._Load(rank, tokens=10, pending=7).release()
        self.assertEqual((rank.active, rank.inflight_tokens, rank.pending_prefill), (0, 0, 0))


class LoadMetricsParseTest(unittest.TestCase):
    SAMPLE = "\n".join([
        '# HELP vllm:kv_cache_usage_perc KV-cache usage.',
        'vllm:num_requests_running{engine="0",model_name="dsv41"} 9.0',
        'vllm:num_requests_waiting{engine="0",model_name="dsv41"} 3.0',
        'vllm:num_requests_waiting_by_reason{engine="0",model_name="dsv41",reason="capacity"} 3.0',
        'vllm:kv_cache_usage_perc{engine="0",model_name="dsv41"} 0.872',
        'vllm:prefix_cache_hits_total{engine="0",model_name="dsv41"} 5.2e+07',
    ])

    def test_parses_exact_names_only(self) -> None:
        got = prefix_router.parse_load_metrics(self.SAMPLE)
        self.assertEqual(got, {"running": 9.0, "waiting": 3.0, "kv_usage": 0.872})

    def test_percentage_is_normalised(self) -> None:
        got = prefix_router.parse_load_metrics('vllm:kv_cache_usage_perc{engine="0"} 87.2')
        self.assertAlmostEqual(got["kv_usage"], 0.872)

    def test_missing_metrics_give_empty(self) -> None:
        self.assertEqual(prefix_router.parse_load_metrics("foo 1\nbar{a=\"b\"} 2"), {})


class CostPolicyTest(unittest.TestCase):
    PROMPT = 80_000

    def setUp(self) -> None:
        self.router = make_router("cost")
        self.now = time.monotonic()

    def saturate(self, idx: int, waiting: int = 3, kv: float = 0.9) -> None:
        r = self.router.ranks[idx]
        r.waiting, r.kv_usage, r.load_ts = waiting, kv, time.monotonic()

    def test_prefers_longest_prefix(self) -> None:
        self.assertEqual(self.router._choose_rank(b"k", [3, 1], 3 * 128), 0)
        self.assertEqual(self.router.stats["affinity_hits"], 1)

    def test_tie_uses_least_active(self) -> None:
        self.router.ranks[0].active = 4
        self.router.ranks[1].active = 1
        self.assertEqual(self.router._choose_rank(b"k", [2, 2], self.PROMPT), 1)

    def test_request_count_alone_does_not_override_affinity(self) -> None:
        """The legacy skew rule is gone: 10 vs 0 active requests is not a reason to
        recompute a 3-block prefix (the old behaviour is --policy legacy)."""
        self.router.ranks[0].active = 10
        self.assertEqual(self.router._choose_rank(b"k", [3, 0], 3 * 128 + 100), 0)

    def test_cold_request_avoids_saturated_rank(self) -> None:
        self.saturate(0)
        self.assertEqual(self.router._choose_rank(b"k", [0, 0], self.PROMPT), 1)
        self.assertEqual(self.router.stats["sat_avoid"], 1)
        self.assertEqual(self.router.stats["fallback"], 1)

    def test_strong_match_survives_saturation(self) -> None:
        self.saturate(0)
        matched = [600, 0]          # 76,800 of 80,000 tokens cached on rank 0
        self.assertEqual(self.router._choose_rank(b"k", matched, self.PROMPT), 0)

    def test_weak_match_loses_to_saturation(self) -> None:
        self.saturate(0)
        matched = [100, 0]          # 12,800 tokens cached: less than the saturation penalty
        self.assertEqual(self.router._choose_rank(b"k", matched, self.PROMPT), 1)
        self.assertEqual(self.router.stats["spills"], 1)

    def test_prefill_backlog_moves_cold_requests(self) -> None:
        self.router.ranks[0].pending_prefill = 200_000
        self.assertEqual(self.router._choose_rank(b"k", [0, 0], self.PROMPT), 1)

    def test_prefill_backlog_does_not_beat_a_big_match(self) -> None:
        self.router.ranks[0].pending_prefill = 30_000
        self.assertEqual(self.router._choose_rank(b"k", [600, 0], self.PROMPT), 0)

    def test_stale_load_data_is_ignored(self) -> None:
        self.saturate(0)
        self.router.ranks[0].load_ts = time.monotonic() - prefix_router.LOAD_STALE_S - 1
        self.assertEqual(self.router._choose_rank(b"k", [0, 0], self.PROMPT), 0)

    def test_unknown_prompt_length_still_decides(self) -> None:
        self.saturate(0)
        self.assertEqual(self.router._choose_rank(b"k", [0, 0], 0), 1)

    def test_unhealthy_best_rank_does_not_raise(self) -> None:
        """Regression: min() over an empty list -> HTTP 500 when the rank holding
        the prefix was marked unhealthy."""
        self.router.ranks[0].healthy = False
        self.assertEqual(self.router._choose_rank(b"k", [5, 0], 5 * 128 + 10), 1)

    def test_all_unhealthy_falls_back_to_all_ranks(self) -> None:
        for r in self.router.ranks:
            r.healthy = False
        self.assertIn(self.router._choose_rank(b"k", [0, 0], self.PROMPT), (0, 1))

    def test_concurrent_cold_requests_alternate_via_backlog(self) -> None:
        """A burst of cold requests must spread out even before any scrape lands."""
        picks = []
        for _ in range(6):
            idx = self.router._choose_rank(b"k", [0, 0], 40_000)
            picks.append(idx)
            self.router.ranks[idx].pending_prefill += 40_000
        self.assertEqual(sorted(set(picks)), [0, 1])
        self.assertLessEqual(abs(picks.count(0) - picks.count(1)), 1)


class LegacyPolicyTest(unittest.TestCase):
    def test_prefers_longest_prefix(self) -> None:
        self.assertEqual(make_router("legacy")._choose_rank(b"key", [3, 1]), 0)

    def test_tie_uses_least_active(self) -> None:
        router = make_router("legacy")
        router.ranks[0].active = 4
        router.ranks[1].active = 1
        self.assertEqual(router._choose_rank(b"key", [2, 2]), 1)

    def test_skew_breaks_sticky_affinity(self) -> None:
        router = make_router("legacy", max_skew=2)
        router.ranks[0].active = 10
        router.ranks[1].active = 0
        self.assertEqual(router._choose_rank(b"key", [3, 0]), 1)

    def test_unhealthy_best_rank_does_not_raise(self) -> None:
        router = make_router("legacy")
        router.ranks[0].healthy = False
        self.assertEqual(router._choose_rank(b"key", [5, 0]), 1)


class TokenizePayloadTest(unittest.TestCase):
    def test_tokenize_payload_excludes_sampling_fields(self) -> None:
        router = make_router()
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
            "max_tokens": 32,
            "temperature": 0,
        }
        payload = router._tokenize_payload(body)
        assert payload is not None
        self.assertEqual(payload["messages"], body["messages"])
        self.assertEqual(payload["tools"], body["tools"])
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)

    def test_tokenize_targets_rotate_over_healthy_ranks(self) -> None:
        router = make_router()
        first = [router._tokenize_targets()[0].index for _ in range(4)]
        self.assertEqual(first, [0, 1, 0, 1])
        router.ranks[0].healthy = False
        self.assertEqual({router._tokenize_targets()[0].index for _ in range(4)}, {1})


# ---- KV event wire format + bookkeeping ---------------------------------------
class BlockStored(msgspec.Struct, omit_defaults=True, gc=False, tag=True):
    block_hashes: list
    parent_block_hash: object
    token_ids: list
    block_size: int
    lora_id: object = None
    medium: object = None
    lora_name: object = None
    group_idx: object = None


class BlockRemoved(msgspec.Struct, omit_defaults=True, gc=False, tag=True):
    block_hashes: list
    medium: object = None
    group_idx: object = None


class _Batch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list
    data_parallel_rank: object = None


def wire(events: list) -> list:
    """Encode like vLLM's ZMQ publisher, decode like Router._kv_event_loop."""
    payload = msgspec.msgpack.Encoder().encode(_Batch(ts=0.0, events=events, data_parallel_rank=0))
    return msgspec.msgpack.decode(payload)[1]


class KvEventTest(unittest.TestCase):
    BS = 128

    def feed(self, router, rank_idx, events) -> None:
        for ev in wire(events):
            router._handle_event(router.ranks[rank_idx], ev)

    def test_wire_format_is_tagged_dict_and_matches(self) -> None:
        router = make_router()
        n = 12
        t = toks(1, n)
        h = chain_hashes(t, self.BS)
        evs = wire([BlockStored(block_hashes=h, parent_block_hash=None, token_ids=t,
                                block_size=self.BS, group_idx=0)])
        self.assertIsInstance(evs[0], dict)
        self.assertEqual(evs[0]["type"], "BlockStored")
        self.feed(router, 0, [BlockStored(block_hashes=h, parent_block_hash=None, token_ids=t,
                                          block_size=self.BS, group_idx=0)])
        probe = prefix_router.prefix_digests(t + [9] * 10, self.BS)
        self.assertEqual(router.ranks[0].matched_blocks(probe), n)
        self.assertEqual(router.ranks[1].matched_blocks(probe), 0)

    def test_twelve_groups_fuzz_matches_oracle(self) -> None:
        """12 KV-cache groups, all block_size 128 (the production layout): chunked
        prefill, per-group eviction under pressure. The router's belief must equal
        'a block is cached if any group still holds it, contiguous from block 0'."""
        rng = random.Random(7)
        groups = 12
        router = make_router()
        rank = router.ranks[0]
        hlist = []           # one entry per physical (group, hash) block
        hash_refs = {}       # hash -> physical blocks across groups
        reqs = []
        shared = toks(1, 6)
        checks = mismatches = 0

        def evict_one() -> None:
            i = rng.randrange(len(hlist))
            g, hh = hlist[i]
            hlist[i] = hlist[-1]
            hlist.pop()
            self.feed(router, 0, [BlockRemoved(block_hashes=[hh], group_idx=g)])
            hash_refs[hh] -= 1
            if hash_refs[hh] == 0:
                del hash_refs[hh]

        for step in range(500):
            if rng.random() < 0.4 or not reqs:
                tail = toks(1000 + step, rng.randint(1, 8))
                t = shared + tail if rng.random() < 0.6 else tail
                h = chain_hashes(t, self.BS, b"R")
                reqs.append((t, h))
                i = 0
                while i < len(h):
                    j = min(len(h), i + rng.randint(1, 5))       # chunked prefill
                    for g in range(groups):
                        self.feed(router, 0, [BlockStored(
                            block_hashes=h[i:j], parent_block_hash=(h[i - 1] if i else None),
                            token_ids=t[i * self.BS:j * self.BS], block_size=self.BS, group_idx=g)])
                        for hh in h[i:j]:
                            hlist.append((g, hh))
                            hash_refs[hh] = hash_refs.get(hh, 0) + 1
                    i = j
            elif hlist:
                evict_one()
            while len(hlist) > 900:                              # pool capacity
                evict_one()
            if step % 10 == 0:
                for t, h in rng.sample(reqs, min(3, len(reqs))):
                    truth = 0
                    for hh in h:
                        if hh not in hash_refs:
                            break
                        truth += 1
                    got = rank.matched_blocks(prefix_router.prefix_digests(t + [3], self.BS))
                    checks += 1
                    mismatches += got != truth
        self.assertGreater(checks, 100)
        self.assertEqual(mismatches, 0)
        self.assertEqual(len(rank.digests), len(hash_refs))      # no leaked digests


if __name__ == "__main__":
    unittest.main()
