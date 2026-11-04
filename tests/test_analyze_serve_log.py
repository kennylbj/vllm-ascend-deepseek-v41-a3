#!/usr/bin/env python3
"""Tests for scripts/analyze_serve_log.py on a synthetic two-engine log.

The log has 30 healthy windows followed by 30 stalled ones, so every statistic
has a known answer. Pure standard library.

Run:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_serve_log as asl  # noqa: E402


def _hms(window: int, base_hour: int = 8) -> str:
    t = base_hour * 3600 + window * 10
    return "%02d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)


def stat_line(engine: int, window: int, gen: float, running: int, kv: float,
              prompt: float = 0.0, waiting: int = 0, hit: float = 70.0, drafted: int | None = None) -> list:
    ts = "11-04 %s" % _hms(window)
    pre = "(APIServer_DP%d pid=1) INFO %s" % (engine, ts)
    lines = ["%s [loggers.py:315] Engine %03d: Avg prompt throughput: %.1f tokens/s, "
             "Avg generation throughput: %.1f tokens/s, Running: %d reqs, Waiting: %d reqs, "
             "NPU KV cache usage: %.1f%%, Prefix cache hit rate: %.1f%%"
             % (pre, engine, prompt, gen, running, waiting, kv, hit)]
    if drafted is not None:
        lines.append("%s [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.00, "
                     "Accepted throughput: 1.00 tokens/s, Drafted throughput: 2.00 tokens/s, "
                     "Accepted: 10 tokens, Drafted: %d tokens, Per-position acceptance rate: 0.5" % (pre, drafted))
    return lines


def synthetic_log(healthy: int = 30, stalled: int = 30) -> list:
    lines = ["(EngineCore_DP0 pid=2) INFO 11-04 07:59:00 [kv_cache_utils.py:2395] "
             "NPU KV cache size: 848,184 tokens, Maximum concurrency for 360,448 tokens per request: 2.35x"]
    kv = {0: 40.0, 1: 40.0}
    for w in range(healthy + stalled):
        is_stalled = w >= healthy
        for e in (0, 1):
            if is_stalled:
                kv[e] += 1.5 if e == 0 else 0.0     # rank 0 prefills, rank 1 does not
            lines += stat_line(e, w, gen=10.0 if is_stalled else 300.0, running=10, kv=kv[e],
                               prompt=500.0 if is_stalled else 0.0,
                               drafted=200 if is_stalled else 4650)   # 0.4 vs 9.3 steps/s per request
    return lines


class AnalyzeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples, self.kv_tokens = asl.parse_log(synthetic_log())
        self.rep = asl.analyze(self.samples, kv_tokens=self.kv_tokens)

    def test_parse_picks_up_kv_capacity_and_both_engines(self) -> None:
        self.assertEqual(self.kv_tokens, 848184)
        self.assertEqual(sorted(self.samples), [0, 1])
        self.assertEqual(len(self.samples[0]), 60)

    def test_stall_and_healthy_shares(self) -> None:
        self.assertEqual(self.rep["active"], 60)
        self.assertAlmostEqual(self.rep["stalled_share"], 0.5)
        self.assertAlmostEqual(self.rep["healthy_share"], 0.5)
        self.assertAlmostEqual(self.rep["gen_total_both_healthy_mean"], 600.0)

    def test_ranks_stall_together(self) -> None:
        self.assertAlmostEqual(self.rep["p_other_stalled_given_stalled"], 1.0)
        self.assertGreater(self.rep["corr"], 0.99)

    def test_step_time_from_spec_counters(self) -> None:
        self.assertAlmostEqual(self.rep["step_ms"]["H"], 1000 / 9.3, delta=2)
        self.assertAlmostEqual(self.rep["step_ms"]["S"], 2500, delta=10)

    def test_stall_follows_kv_growth(self) -> None:
        g = self.rep["by_kv_growth"]
        self.assertEqual(g[">=1%"]["windows"], 30)
        self.assertAlmostEqual(g[">=1%"]["stalled"], 1.0)
        self.assertAlmostEqual(g["<0.3%"]["stalled"], 0.0)

    def test_no_overlap_ratio_without_both_rank_windows(self) -> None:
        # only rank 0 grows, so there are no "both ranks" windows -> no overlap ratio
        self.assertNotIn("prefill_overlap", self.rep)

    def test_episode_detection(self) -> None:
        ep = self.rep["episodes"]
        self.assertEqual(ep["count"], 1)
        self.assertEqual(ep["max_s"], 300)

    def test_since_until_filter(self) -> None:
        rep = asl.analyze(self.samples, kv_tokens=self.kv_tokens, since="08:00", until="08:04")
        self.assertEqual(rep["windows"], 25)       # 08:00:00 .. 08:04:00 inclusive
        self.assertAlmostEqual(rep["stalled_share"], 0.0)

    def test_buckets_cover_the_run(self) -> None:
        rows = self.rep["buckets"]
        self.assertEqual(rows[0]["bucket"], "08:00")
        self.assertAlmostEqual(rows[0]["r0"]["uncached_prefill_M"], 500.0 * 10 * 30 / 1e6)

    def test_single_engine_log_reports_error(self) -> None:
        only0 = {0: self.samples[0]}
        rep = asl.analyze(only0)
        self.assertIn("error", rep)
        self.assertTrue(asl.render(rep).startswith("ERROR"))

    def test_render_contains_all_sections(self) -> None:
        text = asl.render(self.rep)
        for needle in ("both stalled 50.0%", "lockstep", "per bucket", "848,184"):
            self.assertIn(needle, text)

    def test_both_ranks_prefilling_ratio(self) -> None:
        lines = []
        kv = {0: 10.0, 1: 10.0}
        for w in range(40):
            for e in (0, 1):
                kv[e] += 2.0 if w % 2 == 0 else 0.0
                if w < 20 and e == 1:
                    kv[e] -= 2.0 if w % 2 == 0 else 0.0          # first half: only rank 0 grows
                lines += stat_line(e, w, gen=20.0, running=10, kv=kv[e])
        samples, _ = asl.parse_log(lines)
        rep = asl.analyze(samples, kv_tokens=1_000_000)
        po = rep["prefill_overlap"]
        self.assertGreater(po["ratio"], 1.5)       # both-rank windows move ~2x the tokens


if __name__ == "__main__":
    unittest.main()
