#!/usr/bin/env python3
# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Admission delay and retained KV per run, as in the paper's table of the main runs.

wait: how long a request waited beyond its tool latency (admission or pause delay), from
``req_rec``; last start: when the last trajectory started, as a share of rollout time; peak refs:
blocks held by shared-prefix references (sum of per-instance peaks reported by the engine hooks)
as a share of all KV blocks.  Medians over runs, per experiment family and policy.  Writes
``<results-dir>/admission_wait.json``.

Usage:
    python scripts/admission_wait.py [--results-dir DIR] [PREFIX ...]     (default prefix: m_)
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from canopy.bench.runner import build_workload  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def family(exp):
    return re.sub(r"_r\d+$", "", exp) if re.search(r"_r\d+$", exp) and not exp.startswith("sc_repl") else exp


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(v, spec):
    return "n/a" if v is None else format(v, spec)


def main(results_dir, prefixes):
    runs = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "results.json"))):
        exp = os.path.basename(os.path.dirname(path))
        if not any(exp.startswith(p) for p in prefixes):
            continue
        with open(path) as f:
            res = json.load(f)
        w = res["workload"]
        if w.get("kind") == "mixed":
            continue
        by = {n["id"]: n for n in build_workload(w, res["args"]["seed"])}
        total_blocks = sum(b or 0 for b in (res.get("num_gpu_blocks") or []))
        for run in res.get("runs", []):
            rec = run.get("req_rec") or {}
            if not rec or run.get("makespan_s") is None:
                continue
            waits, traj_first = [], {}
            for nid, r in rec.items():
                n = by.get(nid)
                if n is None:
                    continue
                if n["parents"]:
                    parent_end = max(rec[q][2] for q in n["parents"] if q in rec)
                    waits.append(max(0.0, r[1] - parent_end - n["delay_ms"] / 1000.0))
                traj_first[n["traj"]] = min(traj_first.get(n["traj"], 1e18), r[1])
            pinned = (run.get("engine_totals") or {}).get("max_pinned_blocks")
            runs[family(exp)][run["policy"]].append({
                "wait_p99_s": pct(waits, 0.99) if waits else None,
                "wait_max_s": max(waits) if waits else None,
                "last_start": max(traj_first.values()) / run["makespan_s"] if traj_first else None,
                "preemptions": run.get("preemptions"),
                "peak_refs": (pinned / total_blocks) if (pinned is not None and total_blocks) else None})
    if not runs:
        raise SystemExit(f"no results with prefixes {prefixes} under {results_dir}")
    table = {}
    for fam, pols in sorted(runs.items()):
        print(f"== {fam}")
        print(f"   {'policy':14s} {'n':>2s} {'wait p99 s':>10s} {'wait max s':>10s} {'last start':>10s} "
              f"{'preempt.':>8s} {'peak refs':>9s}")
        table[fam] = {}
        for pol, xs in pols.items():
            row = {k: med([x[k] for x in xs]) for k in xs[0]}
            row["n"] = len(xs)
            table[fam][pol] = row
            print(f"   {pol:14s} {row['n']:2d} {fmt(row['wait_p99_s'], '10.1f')} {fmt(row['wait_max_s'], '10.0f')} "
                  f"{fmt(row['last_start'], '10.0%')} {fmt(row['preemptions'], '8.0f')} {fmt(row['peak_refs'], '9.0%')}")
    out = os.path.join(results_dir, "admission_wait.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Admission delay and retained KV per experiment family and policy.")
    ap.add_argument("prefixes", nargs="*", default=["m_"], help="experiment name prefixes (default: m_)")
    ap.add_argument("--results-dir", default="./results")
    a = ap.parse_args()
    main(a.results_dir, a.prefixes)
