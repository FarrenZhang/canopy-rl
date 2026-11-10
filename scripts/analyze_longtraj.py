#!/usr/bin/env python3
# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize the long-trajectory experiments (``lp_lc*``): trajectories that branch late.

Each workload ran as several jobs (``<workload>_r<k><s>``); all but the static-cap
jobs compare a few policies with ``sticky`` on the same servers (see docs/REPRODUCE.md).  Per workload and
policy this prints the median rollout time over all runs, the speedup over
``sticky`` (median ``sticky`` time over every job divided by the policy's
median), the in-job ratios to ``sticky``, the speedup over verl (``vanilla``),
and prefill computed as a multiple of the minimum.  Writes
``<results-dir>/longtraj.json``.

Usage:
    python scripts/analyze_longtraj.py [--results-dir DIR]
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

POLICIES = ["vanilla", "par", "sticky", "rv", "rv_hold", "canopy_fx", "canopy", "canopy_full",
            "stickycap1", "stickycap2", "stickycap3"]


def med(xs):
    return statistics.median(xs) if xs else None


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def main(results_dir):
    runs = defaultdict(lambda: defaultdict(list))
    min_prefill = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "lp_lc*", "results.json"))):
        exp = os.path.basename(os.path.dirname(path))
        workload = exp.rsplit("_r", 1)[0]
        with open(path) as f:
            d = json.load(f)
        min_prefill[workload] = (d.get("workload_stats") or {}).get("min_prefill_tokens")
        for r in d.get("runs", []):
            if r.get("makespan_s") is None or r.get("error"):
                continue
            runs[workload][r["policy"]].append({"t": r["makespan_s"], "pf": r.get("prefill_computed_tokens"),
                                                "pre": r.get("preemptions") or 0, "job": exp})
    if not runs:
        raise SystemExit(f"no lp_lc* results under {results_dir}")

    table = {}
    for w in sorted(runs):
        pols = runs[w]
        sticky = med([x["t"] for x in pols.get("sticky", [])])
        sticky_by_job = defaultdict(list)
        for x in pols.get("sticky", []):
            sticky_by_job[x["job"]].append(x["t"])
        rows = {}
        for p in POLICIES + sorted(set(pols) - set(POLICIES)):
            xs = pols.get(p)
            if not xs:
                continue
            t = med([x["t"] for x in xs])
            pf = med([x["pf"] for x in xs if x["pf"]])
            in_job = sorted(round(med(sticky_by_job[x["job"]]) / x["t"], 3)
                            for x in xs if sticky_by_job.get(x["job"]))
            rows[p] = {"n": len(xs), "median_s": t, "min_s": min(x["t"] for x in xs),
                       "max_s": max(x["t"] for x in xs),
                       "vs_sticky": (sticky / t) if sticky else None,
                       "in_job_vs_sticky": in_job if p != "sticky" else None,
                       "prefill_x_min": (pf / min_prefill[w]) if (pf and min_prefill.get(w)) else None,
                       "preemptions": med([x["pre"] for x in xs])}
        base = rows.get("vanilla", {}).get("vs_sticky")
        for row in rows.values():
            row["vs_verl"] = (row["vs_sticky"] / base) if (base and row["vs_sticky"]) else None
        table[w] = rows

        print(f"== {w}  (minimum prefill {(min_prefill.get(w) or 0) / 1e6:.2f}M tokens)")
        print(f"   {'policy':12s} {'n':>2s} {'time_s':>8s} {'vs_sticky':>9s} {'vs_verl':>7s} "
              f"{'in-job vs sticky':>17s} {'prefill/min':>11s} {'preempt':>7s}")
        for p, x in rows.items():
            ij = "-" if not x["in_job_vs_sticky"] else \
                f"{min(x['in_job_vs_sticky']):.3f}-{max(x['in_job_vs_sticky']):.3f}"
            print(f"   {p:12s} {x['n']:2d} {x['median_s']:8.1f} {fmt(x['vs_sticky'], '9.3f')} "
                  f"{fmt(x['vs_verl'], '7.2f')} {ij:>17s} {fmt(x['prefill_x_min'], '11.2f')} "
                  f"{x['preemptions']:7.0f}")
    out = os.path.join(results_dir, "longtraj.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summarize the long-trajectory experiments (lp_lc*).")
    ap.add_argument("--results-dir", default="./results")
    main(ap.parse_args().results_dir)
