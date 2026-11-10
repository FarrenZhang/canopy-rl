#!/usr/bin/env python3
# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize and audit benchmark results under results/.

Reads ``results/<experiment>/results.json`` (written by canopy.bench.runner)
and prints, per experiment family (split repeats ``*_r0``, ``*_r1``, ... merged):
median rollout time [min-max] per policy, speedup over verl (the ``vanilla``
policy), prefill as a multiple of the minimum, preemptions; and audit flags:
errors, cache-reset failures, unequal work across policies, asymmetric
first-issue offsets.  Writes ``results/table.json`` for plotting.

Usage:
    python scripts/analyze.py [--results-dir DIR] [PREFIX ...]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict


def load(results_dir):
    exps = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "results.json"))):
        with open(p) as f:
            e = json.load(f)
        exps[e["exp"]] = e
    return exps


def family(exp):
    return re.sub(r"_r\d+$", "", exp) if re.search(r"_r\d+$", exp) and not exp.startswith("sc_repl") else exp


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main(results_dir, prefixes):
    exps = load(results_dir)
    if not exps:
        raise SystemExit(f"no results found under {results_dir}/*/results.json")
    fams = defaultdict(list)
    for name, e in sorted(exps.items()):
        if prefixes and not any(name.startswith(p) for p in prefixes):
            continue
        fams[family(name)].append(e)
    table = {}
    for fam, es in sorted(fams.items()):
        runs = [r for e in es for r in e.get("runs", [])]
        ws = es[0].get("workload_stats") or {}
        minp = ws.get("min_prefill_tokens") or 0
        by = defaultdict(list)
        flags = []
        for r in runs:
            if r.get("error"):
                flags.append(f"error {r['policy']} rep{r.get('repeat')}: {str(r['error'])[:80]}")
                continue
            if r.get("makespan_s") is None:
                continue
            if r.get("reset_ok") is False:
                flags.append(f"reset failed {r['policy']} rep{r.get('repeat')}")
            by[r["policy"]].append(r)
        # equal work across policies (per experiment, since split jobs share the workload)
        for e in es:
            sig = defaultdict(set)
            for r in e.get("runs", []):
                if r.get("makespan_s") is None or r.get("error"):
                    continue
                sig["requests"].add(r.get("requests"))
                sig["gen"].add(round(r.get("gen_tokens") or 0))
                sig["prompt"].add(round(r.get("prompt_tokens") or 0))
            for k, v in sig.items():
                if len(v) > 1:
                    flags.append(f"{e['exp']}: unequal {k} across policies {sorted(v)[:4]}")
            fi = [r.get("first_issue_s") for r in e.get("runs", []) if r.get("first_issue_s") is not None]
            if fi and max(fi) - min(fi) > 0.5:
                flags.append(f"{e['exp']}: first_issue spread {min(fi):.3f}-{max(fi):.3f}s")
        rows = {}
        base = statistics.median([r["makespan_s"] for r in by["vanilla"]]) if by.get("vanilla") else None
        for pol, rs in by.items():
            ms = sorted(r["makespan_s"] for r in rs)
            pf = statistics.median([(r.get("prefill_computed_tokens") or 0) for r in rs])
            rows[pol] = {
                "n": len(rs), "median": statistics.median(ms), "min": ms[0], "max": ms[-1],
                "speedup_vs_verl": (base / statistics.median(ms)) if base else None,
                "prefill_x_min": (pf / minp) if minp else None,
                "preempt": statistics.median([(r.get("preemptions") or 0) for r in rs]),
                "kv": statistics.median([(r.get("kv_usage_mean") or 0) for r in rs]),
                "engine": rs[-1].get("engine_totals"), "sched": rs[-1].get("sched_stats"),
            }
        table[fam] = {"rows": rows, "flags": flags, "workload": es[0].get("workload"), "stats": ws,
                      "num_gpu_blocks": es[0].get("num_gpu_blocks"), "jobs": [e["exp"] for e in es]}
        print(f"== {fam}  jobs={len(es)} reuse={ws.get('reuse_ratio_input', 0):.2f} min_prefill={minp/1e6:.2f}M")
        order = sorted(rows, key=lambda p: rows[p]["median"])
        for pol in order:
            x = rows[pol]
            sp = f"{x['speedup_vs_verl']:.2f}x" if x["speedup_vs_verl"] else "  -  "
            pf = f"{x['prefill_x_min']:.2f}" if x["prefill_x_min"] else " - "
            print(f"   {pol:14s} n={x['n']} {x['median']:8.1f}s [{x['min']:.1f}-{x['max']:.1f}] {sp:>6s} "
                  f"prefill={pf}x preempt={x['preempt']:.0f} kv={x['kv']:.2f}")
        for f in flags:
            print("   ! " + f)
    out = os.path.join(results_dir, "table.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summarize and audit benchmark results.")
    ap.add_argument("prefixes", nargs="*", help="only experiments whose names start with one of these")
    ap.add_argument("--results-dir", default="./results")
    a = ap.parse_args()
    main(a.results_dir, a.prefixes)
