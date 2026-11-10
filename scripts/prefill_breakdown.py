#!/usr/bin/env python3
# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Break the prefill computed in each run into minimum, duplicated, and recomputed parts.

minimum: every shared prefix and history computed once (``workload_stats["min_prefill_tokens"]``).
placement minimum: every instance computes once each shared prefix or history it needs, given where
the run sent each request and in which order (from ``req_rec``), and never recomputes anything.
duplicated = placement minimum - minimum; recomputed = engine-computed prefill - placement minimum.
All are multiples of the minimum.  Per experiment family and policy, the total and the duplicated
part are medians over runs, and recomputed = total - 1 - duplicated (at least 0), so that
total = 1 + duplicated + recomputed.  Writes ``<results-dir>/prefill_breakdown.json``.

Usage:
    python scripts/prefill_breakdown.py [--results-dir DIR] [PREFIX ...]     (default prefix: m_)
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

from canopy.bench.runner import build_workload, topo, workload_stats  # noqa: E402


def graph(w, seed):
    nodes = build_workload(w, seed)
    by = {n["id"]: n for n in nodes}
    order = topo(nodes)
    hist, inp, anchor = {}, {}, {}
    for nid in order:
        n = by[nid]
        if n["base"] == "parent":
            base = hist[n["parents"][0]]
        else:
            base = len(n["fresh"]) + (sum(by[p]["gen"] for p in n["parents"]) if n["join_outputs"] else 0)
        inp[nid] = base + len(n["extra"])
        hist[nid] = inp[nid] + n["gen"]
    for nid in order:
        n = by[nid]
        anchor[nid] = None
        if n["base"] == "parent":
            p = n["parents"][0]
            if n["shared_len"] == hist[p] and by[p]["shared_len"] != n["shared_len"]:
                anchor[nid] = p
            elif anchor.get(p) is not None and n["shared_len"] == by[p]["shared_len"]:
                anchor[nid] = anchor[p]
    return nodes, by, hist, inp, anchor


def placement_extra(by, hist, inp, anchor, rec):
    """Tokens that the run's placement forces an instance to compute although they were computed elsewhere."""
    avail = set()
    seen_group = set()
    extra = 0
    srv = {nid: r[0] for nid, r in rec.items()}
    for nid, r in sorted(rec.items(), key=lambda kv: (kv[1][1], kv[0])):
        n = by[nid]
        s = r[0]
        g = n["group"]
        key = ("root", g) if anchor[nid] is None else ("node", anchor[nid])
        b = n["shared_len"]
        if n["base"] == "parent":
            p = n["parents"][0]
            if srv.get(p) != s:
                have = b if (s, key) in avail else 0
                extra += max(0, hist[p] - have)
        elif g in seen_group and (s, ("root", g)) not in avail:
            extra += min(b, inp[nid])
        seen_group.add(g)
        avail.add((s, ("root", g)))
        avail.add((s, ("node", nid)))
        if anchor[nid] is not None:
            avail.add((s, ("node", anchor[nid])))
    return extra


def family(exp):
    return re.sub(r"_r\d+$", "", exp) if re.search(r"_r\d+$", exp) and not exp.startswith("sc_repl") else exp


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
        _, by, hist, inp, anchor = graph(w, res["args"]["seed"])
        minp = workload_stats(list(by.values()))["min_prefill_tokens"]
        for run in res.get("runs", []):
            rec = run.get("req_rec") or {}
            if not rec or run.get("makespan_s") is None:
                continue
            mismatched = sum(1 for nid, r in rec.items() if nid in inp and r[3] != inp[nid])
            extra = placement_extra(by, hist, inp, anchor, rec)
            computed = run.get("prefill_computed_tokens") or 0
            runs[family(exp)][run["policy"]].append({
                "dup": round(extra / minp, 4), "total": round(computed / minp, 4), "length_mismatches": mismatched})
    if not runs:
        raise SystemExit(f"no results with prefixes {prefixes} under {results_dir}")
    table = {}
    for fam, pols in sorted(runs.items()):
        print(f"== {fam}")
        print(f"   {'policy':14s} {'n':>2s} {'dup.':>6s} {'rec.':>6s} {'total':>6s}")
        table[fam] = {}
        for pol, xs in pols.items():
            row = {k: round(statistics.median(x[k] for x in xs), 3) for k in ("dup", "total")}
            row["rec"] = max(0.0, round(row["total"] - 1 - row["dup"], 3))
            row["n"] = len(xs)
            row["length_mismatches"] = sum(x["length_mismatches"] for x in xs)
            table[fam][pol] = row
            flag = "  ! regenerated input lengths differ from the log" if row["length_mismatches"] else ""
            print(f"   {pol:14s} {row['n']:2d} {row['dup']:6.2f} {row['rec']:6.2f} {row['total']:6.2f}{flag}")
    out = os.path.join(results_dir, "prefill_breakdown.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Break computed prefill into minimum, duplicated, and recomputed parts.")
    ap.add_argument("prefixes", nargs="*", default=["m_"], help="experiment name prefixes (default: m_)")
    ap.add_argument("--results-dir", default="./results")
    a = ap.parse_args()
    main(a.results_dir, a.prefixes)
