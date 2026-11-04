#!/usr/bin/env python3
"""Read-only analysis of a DP2 vLLM serve log: why does output tok/s swing?

vLLM's engine logger prints one line per engine every 10 s (``Avg generation
throughput``, ``Running``, ``Waiting``, ``KV cache usage`` ...). With DP+EP the
engines step in lockstep, so a long prefill chunk on either rank slows decode on
both. This script turns the log into the numbers needed to see that and to compare
two runs (before/after a router or scheduler change):

  * output tok/s distribution and the share of "stalled" windows
  * whether the two ranks stall together (lockstep) and the step time in each state
  * stall rate vs. prefill activity (KV growth is used as the prefill proxy)
  * per-bucket KV usage, queue depth, hit rate, uncached prefill per rank

    python3 scripts/analyze_serve_log.py logs/serve_*.log
    python3 scripts/analyze_serve_log.py logs/serve_*.log --since 09:30 --until 10:45

Pure standard library, Python >= 3.7, never writes anything.
"""

from __future__ import annotations

import argparse
import collections
import math
import re
import statistics
import sys

WINDOW_S = 10.0   # vLLM's default stat logging interval

_TS = r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)"
STAT_RE = re.compile(
    _TS + r" \[loggers\.py:\d+\] Engine (\d+): Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"\w+ KV cache usage: ([\d.]+)%(?:, Prefix cache hit rate: ([\d.]+)%)?")
SPEC_RE = re.compile(
    r"\(\w+_DP(\d+)[^)]*\) " + _TS + r" \[metrics\.py:\d+\] SpecDecoding metrics: .*?Drafted: (\d+) tokens")
KV_SIZE_RE = re.compile(r"KV cache size: ([\d,]+) tokens")


class Sample(object):
    __slots__ = ("prompt", "gen", "running", "waiting", "kv", "hit", "drafted")

    def __init__(self, prompt, gen, running, waiting, kv, hit):
        self.prompt, self.gen, self.running, self.waiting = prompt, gen, running, waiting
        self.kv, self.hit, self.drafted = kv, hit, None


def _secs(mo, dd, hh, mi, ss):
    # Month/day ordinal keeps multi-day logs ordered; the year is irrelevant here.
    return (int(mo) * 31 + int(dd)) * 86400 + int(hh) * 3600 + int(mi) * 60 + int(ss)


def parse_log(lines):
    """Return ({engine: {window: Sample}}, kv_cache_tokens or None)."""
    samples = collections.defaultdict(dict)
    kv_tokens = None
    for line in lines:
        m = STAT_RE.search(line)
        if m:
            mo, dd, hh, mi, ss, eng = m.groups()[:6]
            prompt, gen, run, wait, kv, hit = m.groups()[6:]
            k = int(round(_secs(mo, dd, hh, mi, ss) / WINDOW_S))
            samples[int(eng)][k] = Sample(float(prompt), float(gen), int(run), int(wait),
                                          float(kv), float(hit) if hit else None)
            continue
        m = SPEC_RE.search(line)
        if m:
            eng, mo, dd, hh, mi, ss, drafted = m.groups()
            k = int(round(_secs(mo, dd, hh, mi, ss) / WINDOW_S))
            s = samples.get(int(eng), {}).get(k)
            if s is not None:
                s.drafted = int(drafted)
            continue
        if kv_tokens is None:
            m = KV_SIZE_RE.search(line)
            if m:
                kv_tokens = int(m.group(1).replace(",", ""))
    return samples, kv_tokens


def _quantile(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


def _corr(x, y):
    if len(x) < 2:
        return float("nan")
    sx, sy = statistics.pstdev(x), statistics.pstdev(y)
    if sx == 0 or sy == 0:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / len(x) / (sx * sy)


def _tod(k):
    return int(k * WINDOW_S) % 86400


def _parse_hhmm(text):
    if text is None:
        return None
    hh, mm = text.split(":")[:2]
    return int(hh) * 3600 + int(mm) * 60


def _state(gen, stall, healthy):
    return "S" if gen <= stall else ("H" if gen >= healthy else "M")


def analyze(samples, kv_tokens=None, since=None, until=None, stall=40.0, healthy=150.0,
            min_running=5, bucket_min=20, kv_growth_pct=1.0):
    """Compute the report as a plain dict (rendering is separate, for tests)."""
    engines = sorted(samples)
    rep = {"engines": engines, "kv_tokens": kv_tokens}
    if len(engines) < 2:
        rep["error"] = "need two engines (DP2) in the log; found %s" % engines
        return rep
    e0, e1 = engines[0], engines[1]
    lo, hi = _parse_hhmm(since), _parse_hhmm(until)

    def in_range(k):
        t = _tod(k)
        return (lo is None or t >= lo) and (hi is None or t <= hi)

    ks = sorted(k for k in samples[e0] if k in samples[e1] and in_range(k))
    rep["windows"] = len(ks)
    if not ks:
        rep["error"] = "no windows where both engines logged (check --since/--until)"
        return rep
    act = [k for k in ks if samples[e0][k].running >= min_running and samples[e1][k].running >= min_running]
    rep["active"] = len(act)
    rep["thresholds"] = {"stall": stall, "healthy": healthy, "min_running": min_running}

    def dist(xs):
        if not xs:
            return None
        return {"mean": statistics.mean(xs), "sd": statistics.pstdev(xs),
                "p5": _quantile(xs, .05), "p50": _quantile(xs, .5), "p95": _quantile(xs, .95), "max": max(xs)}

    rep["gen_total_all"] = dist([samples[e0][k].gen + samples[e1][k].gen for k in ks])
    rep["gen_total_active"] = dist([samples[e0][k].gen + samples[e1][k].gen for k in act])
    rep["gen_rank"] = {e: dist([samples[e][k].gen for k in act]) for e in (e0, e1)}
    if not act:
        return rep

    # ---- joint stall state / lockstep -----------------------------------------------------
    joint = collections.Counter((_state(samples[e0][k].gen, stall, healthy),
                                 _state(samples[e1][k].gen, stall, healthy)) for k in act)
    rep["joint"] = {"%s%s" % key: n for key, n in joint.items()}
    rep["stalled_share"] = joint[("S", "S")] / float(len(act))
    rep["healthy_share"] = joint[("H", "H")] / float(len(act))
    s0 = set(k for k in act if samples[e0][k].gen <= stall)
    s1 = set(k for k in act if samples[e1][k].gen <= stall)
    rep["p_other_stalled_given_stalled"] = len(s0 & s1) / float(len(s0)) if s0 else float("nan")
    rep["p_other_stalled"] = len(s1) / float(len(act))
    rep["corr"] = _corr([samples[e0][k].gen for k in act], [samples[e1][k].gen for k in act])
    both_h = [samples[e0][k].gen + samples[e1][k].gen for k in act
              if _state(samples[e0][k].gen, stall, healthy) == "H"
              and _state(samples[e1][k].gen, stall, healthy) == "H"]
    rep["gen_total_both_healthy_mean"] = statistics.mean(both_h) if both_h else None

    # ---- engine step time from speculative-decoding counters (drafted / K per request) -------
    step = {}
    for label in ("H", "S"):
        vals = []
        for k in act:
            for e in (e0, e1):
                s = samples[e][k]
                if s.drafted and _state(s.gen, stall, healthy) == label and s.running:
                    vals.append(s.drafted / 5.0 / WINDOW_S / s.running)   # steps/s per request (K=5)
        if vals:
            step[label] = 1000.0 / statistics.mean(vals)
    rep["step_ms"] = step

    # ---- stall vs prefill activity (KV growth proxy) ------------------------------------------
    recs = []
    for k in act:
        p = k - 1
        if p in samples[e0] and p in samples[e1]:
            recs.append((k, samples[e0][k].kv - samples[e0][p].kv, samples[e1][k].kv - samples[e1][p].kv))
    by_growth = collections.OrderedDict()
    for name, f in (("<0.3%", lambda g: g < 0.3), ("0.3-%g%%" % kv_growth_pct, lambda g: 0.3 <= g < kv_growth_pct),
                    (">=%g%%" % kv_growth_pct, lambda g: g >= kv_growth_pct)):
        sel = [k for k, d0, d1 in recs if f(max(d0, d1))]
        both_s = [k for k in sel if samples[e0][k].gen <= stall and samples[e1][k].gen <= stall]
        by_growth[name] = {"windows": len(sel),
                           "stalled": len(both_s) / float(len(sel)) if sel else float("nan"),
                           "gen_total": statistics.mean([samples[e0][k].gen + samples[e1][k].gen for k in sel])
                           if sel else float("nan")}
    rep["by_kv_growth"] = by_growth
    if kv_tokens:
        unit = kv_tokens / 100.0
        one, both = [], []
        for k, d0, d1 in recs:
            g = [d0 >= kv_growth_pct, d1 >= kv_growth_pct]
            if g[0] and g[1]:
                both.append((d0 + d1) * unit)
            elif g[0]:
                one.append(d0 * unit)
            elif g[1]:
                one.append(d1 * unit)
        if one and both:
            rep["prefill_overlap"] = {"one_rank_windows": len(one), "one_rank_tokens": statistics.mean(one),
                                      "both_rank_windows": len(both), "both_rank_tokens": statistics.mean(both),
                                      "ratio": statistics.mean(both) / statistics.mean(one)}

    # ---- stall episodes -------------------------------------------------------------------------
    eps, run, prev = [], 0, None
    for k in act:
        is_stall = samples[e0][k].gen <= stall and samples[e1][k].gen <= stall
        if is_stall:
            run = run + 1 if (prev is not None and k == prev + 1 and run) else 1
        else:
            if run:
                eps.append(run)
            run = 0
        prev = k
    if run:
        eps.append(run)
    rep["episodes"] = ({"count": len(eps), "median_s": statistics.median(eps) * WINDOW_S,
                        "p90_s": sorted(eps)[int(.9 * (len(eps) - 1))] * WINDOW_S,
                        "max_s": max(eps) * WINDOW_S} if eps else None)

    # ---- per-bucket table -------------------------------------------------------------------------
    # Per-engine statistics use each engine's OWN windows (a missed log line on one rank
    # must not drop the other rank's data from the totals).
    buckets = {}
    for e in (e0, e1):
        for k in sorted(samples[e]):
            if not in_range(k):
                continue
            t = int(k * WINDOW_S)
            label = "%02d:%02d" % ((t % 86400) // 3600, ((t % 3600) // 60) // bucket_min * bucket_min)
            d = buckets.setdefault((t // 86400, label), {}).setdefault(e, collections.defaultdict(list))
            s = samples[e][k]
            d["kv"].append(s.kv)
            d["run"].append(s.running)
            d["wait"].append(s.waiting)
            d["gen"].append(s.gen)
            d["prompt"].append(s.prompt * WINDOW_S)
            if s.hit is not None:
                d["hit"].append(s.hit)
    table = []
    for (day, label) in sorted(buckets):
        row = {"bucket": label}
        for i, e in enumerate((e0, e1)):
            d = buckets[(day, label)].get(e)
            row["r%d" % i] = None if not d else {
                "kv_mean": statistics.mean(d["kv"]), "kv_p90": _quantile(d["kv"], .9),
                "running": statistics.mean(d["run"]), "waiting": statistics.mean(d["wait"]),
                "gen": statistics.mean(d["gen"]), "uncached_prefill_M": sum(d["prompt"]) / 1e6,
                "hit_end": d["hit"][-1] if d["hit"] else None}
        table.append(row)
    rep["buckets"] = table
    return rep


def _f(x, spec="%.1f"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else spec % x


def render(rep):
    out = []
    w = out.append
    if rep.get("error"):
        return "ERROR: %s" % rep["error"]
    th = rep["thresholds"]
    w("windows (10 s): %d with both engines, %d active (>=%d running on both)" % (
        rep["windows"], rep["active"], th["min_running"]))
    if rep.get("kv_tokens"):
        w("KV cache per rank: %s tokens" % format(rep["kv_tokens"], ","))
    for title, key in (("total output tok/s, all windows", "gen_total_all"),
                       ("total output tok/s, active windows", "gen_total_active")):
        d = rep.get(key)
        if d:
            w("%-38s mean %7.1f  sd %7.1f  p5 %6.1f  p50 %6.1f  p95 %7.1f  max %7.1f" % (
                title, d["mean"], d["sd"], d["p5"], d["p50"], d["p95"], d["max"]))
    if "joint" not in rep:
        return "\n".join(out)
    w("")
    w("== lockstep: per-rank state (H healthy >= %g tok/s, S stalled <= %g, M between), active windows" % (
        th["healthy"], th["stall"]))
    for key in sorted(rep["joint"]):
        w("   rank0=%s rank1=%s  %5d  %5.1f%%" % (key[0], key[1], rep["joint"][key],
                                                  100.0 * rep["joint"][key] / rep["active"]))
    w("   both stalled %.1f%% | both healthy %.1f%% (mean total %s tok/s)" % (
        100 * rep["stalled_share"], 100 * rep["healthy_share"], _f(rep["gen_total_both_healthy_mean"], "%.0f")))
    w("   P(other rank stalled | this rank stalled) = %s  vs  P(other stalled) = %s ; corr(gen r0, gen r1) = %s" % (
        _f(rep["p_other_stalled_given_stalled"], "%.2f"), _f(rep["p_other_stalled"], "%.2f"),
        _f(rep["corr"], "%.2f")))
    if rep.get("step_ms"):
        w("   engine step time (spec-decode counters): " + ", ".join(
            "%s ~%.0f ms" % (k, v) for k, v in sorted(rep["step_ms"].items())))
    w("")
    w("== stall vs prefill activity (max KV growth of either rank in the window)")
    for name, d in rep["by_kv_growth"].items():
        w("   KV growth %-8s windows %5d  both-stalled %5s  mean total gen %6s tok/s" % (
            name, d["windows"], _f(100 * d["stalled"] if d["windows"] else None, "%.1f%%"),
            _f(d["gen_total"] if d["windows"] else None, "%.0f")))
    po = rep.get("prefill_overlap")
    if po:
        w("   prompt KV moved per window: one rank prefilling %.0f tok (n=%d), both ranks %.0f tok each-sum (n=%d) -> x%.2f" % (
            po["one_rank_tokens"], po["one_rank_windows"], po["both_rank_tokens"], po["both_rank_windows"], po["ratio"]))
    ep = rep.get("episodes")
    if ep:
        w("   stall episodes: %d, median %.0f s, p90 %.0f s, max %.0f s" % (ep["count"], ep["median_s"], ep["p90_s"], ep["max_s"]))
    w("")
    w("== per bucket   (KV mean/p90 %, running, waiting, gen tok/s, uncached prefill M tok, hit % at end)")
    w("   bucket |        rank0                              |        rank1")
    for row in rep["buckets"]:
        cells = []
        for r in ("r0", "r1"):
            d = row[r]
            if d is None:
                cells.append("-")
                continue
            cells.append("%5.1f/%5.1f  run %4.1f  wait %4.1f  gen %6.1f  pf %5.2fM  hit %5s" % (
                d["kv_mean"], d["kv_p90"], d["running"], d["waiting"], d["gen"], d["uncached_prefill_M"],
                _f(d["hit_end"], "%.1f")))
        w("   %s | %s | %s" % (row["bucket"], cells[0], cells[1]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="+", help="serve log file(s)")
    ap.add_argument("--since", help="HH:MM, ignore earlier windows")
    ap.add_argument("--until", help="HH:MM, ignore later windows")
    ap.add_argument("--stall-threshold", type=float, default=40.0,
                    help="per-rank output tok/s at or below this counts as stalled (default 40)")
    ap.add_argument("--healthy-threshold", type=float, default=150.0,
                    help="per-rank output tok/s at or above this counts as healthy (default 150)")
    ap.add_argument("--min-running", type=int, default=5,
                    help="a window is 'active' when both ranks run at least this many requests (default 5)")
    ap.add_argument("--bucket-minutes", type=int, default=20)
    ap.add_argument("--kv-tokens", type=int, default=None,
                    help="KV cache tokens per rank (default: parsed from the log's 'KV cache size' line)")
    args = ap.parse_args(argv)

    lines = []
    for path in args.log:
        with open(path, errors="replace") as fh:
            lines.extend(fh)
    samples, kv_tokens = parse_log(lines)
    rep = analyze(samples, kv_tokens=args.kv_tokens or kv_tokens, since=args.since, until=args.until,
                  stall=args.stall_threshold, healthy=args.healthy_threshold,
                  min_running=args.min_running, bucket_min=args.bucket_minutes)
    print(render(rep))
    return 1 if rep.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
