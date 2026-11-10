#!/usr/bin/env python3
# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Canopy rollout-replay benchmark (one invocation = one experiment).

Launches R vLLM API servers (one per assigned GPU, or per ``--tp`` GPUs) with
the Canopy engine hooks loaded, and replays one rollout batch -- a DAG of model
requests with fixed input tokens, fixed output lengths and fixed tool delays --
under several serving policies.  Generated tokens are taken from the live
response so later turns reuse exactly the KV the engine produced; every policy
processes the same prefill/decode token counts in the same dependency order.

Example:
    python -m canopy.bench.runner --exp demo --model Qwen/Qwen3-8B --gpus 0 \\
        --workload '{"kind": "tree", "groups": 4, "n": 16, "fanout": 4, "branch_at": 2, "turns": 6,
                     "prefix": [300, 600], "obs": [600, 1500], "gen": [100, 400], "tool_ms": [200, 1500]}' \\
        --policies vanilla,sticky,rv --repeats 1

Results are written to ``<out-root>/<exp>/results.json``.
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from collections import defaultdict

from canopy.engine import HOOKS_DIR

try:
    import orjson

    def jdumps(o):
        return orjson.dumps(o)

    def jloads(b):
        return orjson.loads(b)
except Exception:  # pragma: no cover
    def jdumps(o):
        return json.dumps(o).encode()

    def jloads(b):
        return json.loads(b)

VOCAB_LO, VOCAB_HI = 1000, 150000

# Health checks, metrics and cache resets go straight to the servers, never through an HTTP proxy.
_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ---------------------------------------------------------------------------
# Workload construction
# ---------------------------------------------------------------------------


def _tok(rng, n):
    return [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(int(n))]


def _draw(rng, spec):
    """int | [lo, hi] uniform int | {"choice": [...]} | {"lognorm": [median, sigma, lo, hi]} | {"emp": [...]}"""
    if isinstance(spec, (int, float)):
        return spec
    if isinstance(spec, list):
        return rng.randint(int(spec[0]), int(spec[1]))
    if "choice" in spec:
        return rng.choice(spec["choice"])
    if "emp" in spec:
        return rng.choice(spec["emp"])
    if "lognorm" in spec:
        med, sig, lo, hi = spec["lognorm"]
        v = med * math.exp(rng.gauss(0, sig))
        return max(lo, min(hi, v))
    raise ValueError(spec)


def build_workload(w, seed):
    rng = random.Random(seed)
    kind = w["kind"]
    if kind == "mixed":
        # several task types in one rollout batch; groups are interleaved in a seeded random order
        by_group = defaultdict(list)
        for i, sub in enumerate(w["parts"]):
            for n in build_workload(sub, seed * 1000 + i + 1):
                n["id"], n["group"], n["traj"] = f"p{i}.{n['id']}", f"p{i}.{n['group']}", f"p{i}.{n['traj']}"
                n["parents"] = [f"p{i}.{p}" for p in n["parents"]]
                by_group[n["group"]].append(n)
        order = list(by_group)
        rng.shuffle(order)
        return [n for g in order for n in by_group[g]]
    nodes = []

    def node(**kw):
        d = dict(parents=[], base="fresh", fresh=[], extra=[], join_outputs=False,
                 gen=1, delay_ms=0.0, terminal=True, shared_len=0, role="turn")
        d.update(kw)
        d["gen"] = max(1, int(d["gen"]))
        nodes.append(d)
        return d

    G, N = int(w["groups"]), int(w["n"])
    for g in range(G):
        gid = f"g{g}"
        P = int(_draw(rng, w.get("prefix", 1024)))
        prefix = _tok(rng, P)
        if kind == "single":
            for i in range(N):
                priv = _tok(rng, _draw(rng, w.get("private_in", 16)))
                node(id=f"{gid}.t{i}.0", group=gid, traj=f"{gid}.t{i}", fresh=prefix + priv,
                     gen=_draw(rng, w["gen"]), terminal=True, shared_len=P)
        elif kind in ("multiturn", "tree", "compaction"):
            branch_at = int(w.get("branch_at", -1)) if kind == "tree" else -1
            fanout = int(w.get("fanout", 1)) if kind == "tree" else 1
            roots = N if kind != "tree" else max(1, N // fanout)
            comp = w.get("compact_at") if kind == "compaction" else None
            for i in range(roots):
                T = int(_draw(rng, w.get("turns", 4)))
                traj = f"{gid}.t{i}"
                priv = _tok(rng, _draw(rng, w.get("private_in", 16)))
                first = node(id=f"{traj}.0", group=gid, traj=traj, fresh=prefix + priv,
                             gen=_draw(rng, w["gen"]), terminal=(T == 1), shared_len=P)
                frontier = [(traj, first, P + len(priv) + first["gen"], 1, P)]
                while frontier:
                    ctraj, cprev, clen, t, shared = frontier.pop()
                    if t >= T:
                        continue
                    kids = fanout if (t == branch_at and fanout > 1) else 1
                    kshared = clen if kids > 1 else shared
                    if kids > 1 and w.get("branch_mode", "turn") == "inflight":
                        # entropy-triggered partial branching: the turn after the tool result is cut at a
                        # branch point; every child continues the same in-flight prefix
                        obs_n = int(_draw(rng, w["obs"]))
                        delay = float(_draw(rng, w.get("tool_ms", 0)))
                        g_turn = int(_draw(rng, w["gen"]))
                        frac = float(_draw(rng, w.get("branch_frac", [20, 60]))) / 100.0
                        g1 = max(1, int(round(g_turn * frac)))
                        pre = node(id=f"{ctraj}.{t}p", group=gid, traj=ctraj, parents=[cprev["id"]],
                                   base="parent", extra=_tok(rng, obs_n), gen=g1, delay_ms=delay,
                                   terminal=False, shared_len=shared, role="pre_branch")
                        kshared = clen + obs_n + g1
                        last = (t + 1 >= T)
                        for k in range(kids):
                            ktraj = f"{ctraj}.b{k}"
                            g2 = max(1, g_turn - g1)
                            nxt = node(id=f"{ktraj}.{t}", group=gid, traj=ktraj, parents=[pre["id"]],
                                       base="parent", extra=[], gen=g2, delay_ms=0.0,
                                       terminal=last, shared_len=kshared, role="branch")
                            frontier.append((ktraj, nxt, kshared + g2, t + 1, kshared))
                        continue
                    for k in range(kids):
                        ktraj = ctraj if kids == 1 else f"{ctraj}.b{k}"
                        obs_n = int(_draw(rng, w["obs"]))
                        delay = float(_draw(rng, w.get("tool_ms", 0)))
                        last = (t + 1 >= T)
                        if comp and clen + obs_n > comp:
                            summ = node(id=f"{ktraj}.s{t}", group=gid, traj=ktraj, parents=[cprev["id"]],
                                        base="parent", extra=_tok(rng, 64), gen=_draw(rng, w.get("summary_gen", 512)),
                                        delay_ms=delay, terminal=False, shared_len=kshared, role="summary")
                            keep = min(P, int(w.get("compact_keep", 1024)))
                            nxt = node(id=f"{ktraj}.{t}", group=gid, traj=ktraj, parents=[summ["id"]],
                                       base="fresh", fresh=prefix[:keep], join_outputs=True,
                                       extra=_tok(rng, obs_n), gen=_draw(rng, w["gen"]), delay_ms=0.0,
                                       terminal=last, shared_len=keep, role="post_compaction")
                            nlen = keep + summ["gen"] + obs_n + nxt["gen"]
                            frontier.append((ktraj, nxt, nlen, t + 1, keep))
                        else:
                            nxt = node(id=f"{ktraj}.{t}", group=gid, traj=ktraj, parents=[cprev["id"]],
                                       base="parent", extra=_tok(rng, obs_n), gen=_draw(rng, w["gen"]),
                                       delay_ms=delay, terminal=last, shared_len=kshared)
                            frontier.append((ktraj, nxt, clen + obs_n + nxt["gen"], t + 1, kshared))
        elif kind == "multiagent":
            k = int(w.get("agents", 4))
            rgen = _draw(rng, w.get("reader_gen", 256))
            rd = node(id=f"{gid}.reader", group=gid, traj=f"{gid}.reader", fresh=prefix + _tok(rng, 32),
                      gen=rgen, terminal=False, shared_len=P, role="reader")
            shared_hist = P + 32 + int(rgen)
            subs = []
            for a in range(k):
                subs.append(node(id=f"{gid}.a{a}", group=gid, traj=f"{gid}.a{a}", parents=[rd["id"]],
                                 base="parent", extra=_tok(rng, _draw(rng, w.get("role_in", 64))),
                                 gen=_draw(rng, w["gen"]), terminal=True, shared_len=shared_hist, role="subagent"))
            node(id=f"{gid}.join", group=gid, traj=f"{gid}.join", parents=[s["id"] for s in subs],
                 base="fresh", fresh=prefix[: min(P, 512)] + _tok(rng, 64), join_outputs=True,
                 gen=_draw(rng, w.get("join_gen", 256)), terminal=True, shared_len=0, role="join")
        else:
            raise ValueError(kind)
    return nodes


def topo(nodes):
    by = {n["id"]: n for n in nodes}
    seen, out = set(), []
    for n in nodes:
        stack = [(n["id"], False)]
        while stack:
            i, done = stack.pop()
            if done:
                out.append(i)
                continue
            if i in seen:
                continue
            seen.add(i)
            stack.append((i, True))
            for p in by[i]["parents"]:
                if p not in seen:
                    stack.append((p, False))
    return out


def workload_stats(nodes):
    by = {n["id"]: n for n in nodes}
    hist = {}
    tot_in = tot_gen = reusable = 0
    max_ctx = 0
    traj_final = {}
    depth = {}
    curve = defaultdict(float)  # batch working-set curve by turn index (all trajectories start together)
    for nid in topo(nodes):
        n = by[nid]
        depth[nid] = 1 + max((depth[p] for p in n["parents"]), default=-1)
        if n["base"] == "parent":
            base = hist[n["parents"][0]]
            reusable += base
        else:
            base = len(n["fresh"]) + (sum(by[p]["gen"] for p in n["parents"]) if n["join_outputs"] else 0)
            reusable += min(n.get("shared_len", 0), base)
        inp = base + len(n["extra"])
        hist[nid] = inp + n["gen"]
        max_ctx = max(max_ctx, hist[nid])
        traj_final[n["traj"]] = max(traj_final.get(n["traj"], 0), hist[nid] - n.get("shared_len", 0))
        curve[depth[nid]] += max(0, hist[nid] - n.get("shared_len", 0))
        tot_in += inp
        tot_gen += n["gen"]
    # a group's first request cannot reuse its shared prefix
    first_seen = set()
    for nid in topo(nodes):
        n = by[nid]
        if n["base"] == "fresh" and n["group"] not in first_seen:
            first_seen.add(n["group"])
            reusable -= min(n.get("shared_len", 0), len(n["fresh"]))
    return {"requests": len(nodes), "groups": len({n["group"] for n in nodes}),
            "trajectories": len({n["traj"] for n in nodes}),
            "input_tokens": tot_in, "gen_tokens": tot_gen, "reusable_input_tokens": reusable,
            "min_prefill_tokens": tot_in - reusable, "max_context": max_ctx,
            "mean_traj_private_peak": statistics.mean(traj_final.values()) if traj_final else 0,
            # peak of the batch working-set curve, spread over trajectories
            "curve_peak_per_traj": (max(curve.values()) / max(1, len(traj_final))) if curve else 0,
            "reuse_ratio_input": reusable / max(1, tot_in),
            "reuse_ratio_all": reusable / max(1, tot_in + tot_gen)}


# ---------------------------------------------------------------------------
# Servers
# ---------------------------------------------------------------------------


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def tail(path, n=60):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""


def _hooked_env(base_env, stats_dir):
    """Environment that loads the Canopy engine hooks in every process of a vLLM server."""
    env = dict(base_env)
    env.update({
        "PYTHONPATH": HOOKS_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "CANOPY_ENGINE_HOOKS": "1",
        "CANOPY_ENGINE_STATS_DIR": stats_dir,
        "VLLM_SERVER_DEV_MODE": "1",  # enables POST /reset_prefix_cache
    })
    return env


class Servers:
    def __init__(self, args, gpus, workdir):
        self.args, self.gpus, self.workdir = args, gpus, workdir
        self.procs, self.urls, self.num_blocks = [], [], []
        self.block_size = 16

    def start(self):
        self.stats_dir = os.path.join(self.workdir, "engine_stats")
        for i, g in enumerate(self.gpus):
            port = free_port()
            env = _hooked_env(os.environ, self.stats_dir)
            env.update({
                "CUDA_VISIBLE_DEVICES": str(g),
                "TRITON_CACHE_DIR": os.path.join(self.workdir, f"triton_{i}"),
                "VLLM_CACHE_ROOT": os.path.join(self.workdir, f"vllm_cache_{i}"),
                "TORCHINDUCTOR_CACHE_DIR": os.path.join(self.workdir, f"inductor_{i}"),
            })
            cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                   "--model", self.args.model, "--served-model-name", "m",
                   "--host", "127.0.0.1", "--port", str(port),
                   "--gpu-memory-utilization", str(self.args.gpu_util),
                   "--max-model-len", str(self.args.max_model_len),
                   "--max-num-seqs", str(self.args.max_num_seqs),
                   "--max-num-batched-tokens", str(self.args.max_batched_tokens),
                   "--enable-prefix-caching", "--seed", "7", "--dtype", "bfloat16"]
            if self.args.tp > 1:
                cmd += ["--tensor-parallel-size", str(self.args.tp)]
            if self.args.num_gpu_blocks:
                cmd += ["--num-gpu-blocks-override", str(self.args.num_gpu_blocks)]
            if self.args.enforce_eager:
                cmd += ["--enforce-eager"]
            if self.args.server_extra:
                cmd += json.loads(self.args.server_extra)
            log = open(os.path.join(self.workdir, f"server_{i}.log"), "w")
            self.procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
            self.urls.append(f"http://127.0.0.1:{port}")
        deadline = time.time() + 1200
        for i, u in enumerate(self.urls):
            while True:
                if self.procs[i].poll() is not None:
                    raise RuntimeError(f"server {i} died:\n" + tail(os.path.join(self.workdir, f"server_{i}.log"), 80))
                try:
                    with _HTTP.open(u + "/health", timeout=5) as r:
                        if r.status == 200:
                            break
                except Exception:
                    pass
                if time.time() > deadline:
                    raise RuntimeError("server start timeout:\n" + tail(os.path.join(self.workdir, f"server_{i}.log"), 80))
                time.sleep(3)
        for i, u in enumerate(self.urls):
            m = scrape(u)
            nb = m.get("num_gpu_blocks")
            if nb is None:
                txt = tail(os.path.join(self.workdir, f"server_{i}.log"), 600)
                mm = re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)
                nb = int(mm[-1].replace(",", "")) // 16 if mm else None
            self.num_blocks.append(nb)
            if m.get("block_size"):
                self.block_size = int(m["block_size"])

    def stop(self):
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
        t0 = time.time()
        for p in self.procs:
            try:
                p.wait(timeout=max(1, 40 - (time.time() - t0)))
            except Exception:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    pass


class RayServers(Servers):
    """Multi-node variant: one Ray actor per GPU starts a vLLM server on its own node.

    Needs a running Ray cluster, and the engine hooks on every node: either the
    ``canopy`` package is installed there, or this repository is at the same
    path (e.g. a shared filesystem).
    """

    def __init__(self, args, n, workdir):
        super().__init__(args, [None] * n, workdir)
        self.n = n
        self.actors = []

    def start(self):
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        ray.init(address="auto", ignore_reinit_error=True)
        args = self.args
        self.pg = placement_group([{"GPU": 1, "CPU": 2}] * self.n, strategy="PACK")
        ray.get(self.pg.ready(), timeout=900)

        @ray.remote(num_gpus=1, num_cpus=2)
        class ServerActor:
            def start(self, idx, cmd_tail, fallback_hooks_dir):
                import os as _os, socket as _s, subprocess as _sp, tempfile as _t
                try:
                    from canopy.engine import HOOKS_DIR as hooks_dir
                except Exception:
                    hooks_dir = fallback_hooks_dir
                if not _os.path.isfile(_os.path.join(hooks_dir, "sitecustomize.py")):
                    raise RuntimeError(f"Canopy engine hooks not found on this node ({hooks_dir})")
                self.wd = _t.mkdtemp(prefix=f"canopy_srv{idx}_")
                sk = _s.socket()
                sk.bind(("0.0.0.0", 0))
                port = sk.getsockname()[1]
                sk.close()
                env = dict(_os.environ)
                env.update({"PYTHONPATH": hooks_dir + (_os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                            "CANOPY_ENGINE_HOOKS": "1",
                            "CANOPY_ENGINE_STATS_DIR": _os.path.join(self.wd, "engine_stats"),
                            "VLLM_SERVER_DEV_MODE": "1",
                            "TRITON_CACHE_DIR": _os.path.join(self.wd, "triton"),
                            "VLLM_CACHE_ROOT": _os.path.join(self.wd, "vllm_cache"),
                            "TORCHINDUCTOR_CACHE_DIR": _os.path.join(self.wd, "inductor")})
                cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--host", "0.0.0.0",
                       "--port", str(port)] + cmd_tail
                self.log = open(_os.path.join(self.wd, "server.log"), "w")
                self.p = _sp.Popen(cmd, env=env, stdout=self.log, stderr=_sp.STDOUT, start_new_session=True)
                import ray as _r
                return f"http://{_r.util.get_node_ip_address()}:{port}"

            def alive(self):
                return self.p.poll() is None

            def read_stats(self):
                import os as _os, json as _j
                d = _os.path.join(self.wd, "engine_stats")
                out = []
                if _os.path.isdir(d):
                    for f in sorted(_os.listdir(d)):
                        if f.endswith(".json"):
                            try:
                                with open(_os.path.join(d, f)) as fh:
                                    out.append(_j.load(fh))
                            except Exception:
                                pass
                return out

            def log_tail(self, n=200):
                try:
                    with open(self.log.name, errors="replace") as fh:
                        return "".join(fh.readlines()[-n:])
                except Exception:
                    return ""

            def stop(self):
                import os as _os, signal as _sig
                try:
                    _os.killpg(self.p.pid, _sig.SIGTERM)
                    self.p.wait(timeout=40)
                except Exception:
                    try:
                        _os.killpg(self.p.pid, _sig.SIGKILL)
                    except Exception:
                        pass
                return True

        tail_cmd = ["--model", args.model, "--served-model-name", "m",
                    "--gpu-memory-utilization", str(args.gpu_util), "--max-model-len", str(args.max_model_len),
                    "--max-num-seqs", str(args.max_num_seqs), "--max-num-batched-tokens", str(args.max_batched_tokens),
                    "--enable-prefix-caching", "--seed", "7", "--dtype", "bfloat16"]
        if args.num_gpu_blocks:
            tail_cmd += ["--num-gpu-blocks-override", str(args.num_gpu_blocks)]
        if args.server_extra:
            tail_cmd += json.loads(args.server_extra)
        for i in range(self.n):
            a = ServerActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=i)).remote()
            self.actors.append(a)
        self.urls = ray.get([a.start.remote(i, tail_cmd, HOOKS_DIR) for i, a in enumerate(self.actors)])
        deadline = time.time() + 1200
        for i, u in enumerate(self.urls):
            while True:
                if not ray.get(self.actors[i].alive.remote()):
                    raise RuntimeError(f"server {i} died:\n" + ray.get(self.actors[i].log_tail.remote(80)))
                try:
                    with _HTTP.open(u + "/health", timeout=5) as r:
                        if r.status == 200:
                            break
                except Exception:
                    pass
                if time.time() > deadline:
                    raise RuntimeError("server start timeout")
                time.sleep(3)
        for u in self.urls:
            m = scrape(u)
            self.num_blocks.append(m.get("num_gpu_blocks"))
            if m.get("block_size"):
                self.block_size = int(m["block_size"])
        self.stats_dir = None

    def read_all_stats(self):
        import ray
        out = []
        for i, a in enumerate(self.actors):
            for e in ray.get(a.read_stats.remote()):
                e["srv"] = i  # hostnames can repeat across nodes
                out.append(e)
        return out

    def stop(self):
        import ray
        try:
            ray.get([a.stop.remote() for a in self.actors], timeout=120)
        except Exception:
            pass
        for a in self.actors:
            try:
                ray.kill(a)
            except Exception:
                pass
        try:
            ray.util.remove_placement_group(self.pg)
        except Exception:
            pass


_PROM_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)$')


def scrape(url):
    out = {}
    try:
        with _HTTP.open(url + "/metrics", timeout=15) as r:
            txt = r.read().decode(errors="replace")
    except Exception as e:
        return {"_error": str(e)}
    for line in txt.splitlines():
        if not line or line[0] == "#":
            continue
        m = _PROM_LINE.match(line.strip())
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        if name == "vllm:cache_config_info":
            for k, s in re.findall(r'(\w+)="([^"]*)"', labels):
                if k in ("num_gpu_blocks", "block_size"):
                    try:
                        out[k] = int(s)
                    except ValueError:
                        pass
            continue
        key = name
        src = re.search(r'source="([^"]+)"', labels)
        if src:
            key = f"{name}[{src.group(1)}]"
        out[key] = out.get(key, 0.0) + v
    return out


METRIC_KEYS = [
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
    "vllm:iteration_tokens_total_sum", "vllm:iteration_tokens_total_count",
    "vllm:prompt_tokens_cached_total", "vllm:prompt_tokens_recomputed_total",
    "vllm:request_prefill_kv_computed_tokens_sum",
]


def metric_delta(before, after):
    keys = set(METRIC_KEYS) | {k for k in after if k.startswith("vllm:prompt_tokens_by_source")}
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys if k in after or k in before}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

# route: replica selection; admit: admission control; hint: request hints for the
# engine hooks ("lifetime": Canopy KV lifetime hints, "ttl": time-to-live baseline).
POLICIES = {
    # vLLM as used by verl: least-outstanding routing, each trajectory sticky to one replica
    "vanilla": dict(route="lor", admit="none", hint="none"),
    # prefix-aware routing (SGLang cache-aware router)
    "par": dict(route="par", admit="none", hint="none"),
    # whole group on one replica
    "sticky": dict(route="group", admit="none", hint="none"),
    # pin a trajectory's KV for a fixed time while it waits on a tool
    "ttl": dict(route="lor", admit="none", hint="ttl"),
    "ttl_par": dict(route="par", admit="none", hint="ttl"),
    # working-set budget admission and hint-only variants
    "canopy_budget": dict(route="canopy", admit="budget", hint="lifetime"),
    "canopy_noadmit": dict(route="canopy", admit="none", hint="lifetime"),
    "canopy_budget_nohint": dict(route="canopy", admit="budget", hint="none"),
    "canopy_noplace_budget": dict(route="lor", admit="budget", hint="lifetime"),
    "hint_only": dict(route="lor", admit="none", hint="lifetime"),
    "sticky_hint": dict(route="group", admit="none", hint="lifetime"),
    # Canopy engine-side mechanisms: group placement + shared-prefix retention while the
    # group is live (soft pin) + prefix-preserving preemption; no admission control
    "rv": dict(route="group", admit="none", hint="lifetime", pin=True, pp=True),
    "rv_hold": dict(route="group", admit="none", hint="lifetime", pin=True, pp=True, pin_mode="admit"),
    "rv_nopin": dict(route="group", admit="none", hint="lifetime", pp=True),
    "rv_nopp": dict(route="group", admit="none", hint="lifetime", pin=True),
    "rv_noplace": dict(route="lor", admit="none", hint="lifetime", pin=True, pp=True),
    # program-level schedulers with periodic pause/restore
    # thunder: ThunderAgent policy (per-program context accounting, memory-balanced placement)
    "thunder": dict(route="prog", admit="prog", hint="none", shared_aware=False, group_place=False),
    # program-level policy that counts each group's shared prompt once per replica; it needs group ids,
    # so it is stronger than ThunderAgent's released code, which charges every program its full context
    "thunder_shared": dict(route="prog", admit="prog", hint="none", shared_aware=True, group_place=False),
    # ThunderAgent's released scheduler (commit 7ddc861) with its defaults: 5 s checks, no discount, smallest
    # tool-waiting program paused first, running programs marked for pausing, best-fit-decreasing resume
    "thunder_orig": dict(route="prog", admit="prog", hint="none", shared_aware=False, group_place=False,
                         ta_orig=True, sched_dt=5.0),
    # canopy: same machinery, structure-aware (shared prefix counted once per group,
    # group co-location, exact liveness hints to the engine, previous-step estimate of
    # each trajectory's peak)
    "canopy": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=True,
                   group_life=True),
    "canopy_noprior": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=False,
                           group_life=True),
    # canopy plus shared-prefix retention and prefix-preserving preemption
    "canopy_full": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=True,
                        group_life=True, pin=True, pp=True),
    # canopy whose group charge follows each request's shared length, so a fork after many turns counts the
    # pre-fork history once (canopy keeps the group's shared length from its first request)
    "canopy_fx": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=True,
                      group_life=True, fork_shared=True),
    "canopy_pp": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=True,
                      group_life=True, pp=True),
    "canopy_nohint": dict(route="prog", admit="prog", hint="none", shared_aware=True, group_place=True, prior=True,
                          group_life=True),
    "canopy_noshare": dict(route="prog", admit="prog", hint="lifetime", shared_aware=False, group_place=True, prior=True,
                           group_life=True),
    "canopy_noplace": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=False, prior=True,
                           group_life=True),
    "canopy_nolife": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=True),
    "canopy_gprior": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=True,
                          group_life=True, global_prior=True),
    "canopy_nofut": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=True,
                         group_life=True, reserve_future=False),
    "canopy_allfut": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=True,
                          group_life=True, reserve_future="all"),
    "thunder_hint": dict(route="prog", admit="prog", hint="lifetime", shared_aware=False, group_place=False),
    "thunder_prior": dict(route="prog", admit="prog", hint="none", shared_aware=False, group_place=False, prior=True),
    "canopy_peak": dict(route="prog", admit="prog", hint="lifetime", shared_aware=True, group_place=True, prior=True,
                        prior_kind="peak", group_life=True),
}
for _s, _tag in ((0.25, "025"), (0.5, "05"), (2.0, "2"), (4.0, "4")):
    POLICIES[f"canopy_ps{_tag}"] = dict(route="prog", admit="prog", hint="lifetime", shared_aware=True,
                                        group_place=True, prior=True, prior_scale=_s, group_life=True)
for _l, _tag in ((0.8, "08"), (0.9, "09"), (1.2, "12")):
    POLICIES[f"canopy_l{_tag}"] = dict(route="prog", admit="prog", hint="lifetime", shared_aware=True,
                                       group_place=True, prior=True, group_life=True, lam=_l)
for _l, _tag in ((0.8, "08"), (0.9, "09"), (1.2, "12")):
    POLICIES[f"thunder_l{_tag}"] = dict(route="prog", admit="prog", hint="none", shared_aware=False,
                                        group_place=False, lam=_l)
# static caps on concurrently admitted trajectories per replica
for _k in (1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
    POLICIES[f"cap{_k}"] = dict(route="lor", admit=f"cap{_k}", hint="none")
    POLICIES[f"stickycap{_k}"] = dict(route="group", admit=f"cap{_k}", hint="none")


class Router:
    def __init__(self, kind, n_servers, chunk=256, par_abs=32, par_rel=1.5, par_thresh=0.5):
        self.kind, self.R = kind, n_servers
        self.outstanding = [0] * n_servers
        self.traj_server, self.group_server = {}, {}
        self.chunk = chunk
        self.trees = [set() for _ in range(n_servers)]
        self.tree_size = [0] * n_servers
        self.par_abs, self.par_rel, self.par_thresh = par_abs, par_rel, par_thresh
        self.rr = 0

    def _least(self, loads):
        m = min(loads)
        cands = [i for i, v in enumerate(loads) if v == m]
        self.rr += 1
        return cands[self.rr % len(cands)]

    def _chunks(self, toks):
        out, h = [], 0
        for c in range(len(toks) // self.chunk):
            h = hash((h, tuple(toks[c * self.chunk:(c + 1) * self.chunk])))
            out.append(h)
        return out

    def route(self, node, prompt):
        traj, group = node["traj"], node["group"]
        if self.kind == "lor":
            s = self.traj_server.get(traj)
            if s is None:
                s = self._least(self.outstanding)
                self.traj_server[traj] = s
            return s
        if self.kind in ("group", "canopy"):
            s = self.group_server.get(group)
            if s is None:
                s = self._least(self.outstanding)
                self.group_server[group] = s
            return s
        if self.kind == "par":  # SGLang cache-aware router
            mx, mn = max(self.outstanding), min(self.outstanding)
            ch = self._chunks(prompt)
            if mx - mn > self.par_abs and mx > self.par_rel * max(mn, 1):
                s = self._least(self.outstanding)
            else:
                best, best_m = 0, -1
                for i in range(self.R):
                    m = 0
                    for c in ch:
                        if c in self.trees[i]:
                            m += 1
                        else:
                            break
                    if m > best_m:
                        best, best_m = i, m
                if best_m / max(1, len(ch)) > self.par_thresh:
                    s = best
                else:
                    s = min(range(self.R), key=lambda i: (self.tree_size[i], self.outstanding[i]))
            for c in ch:
                if c not in self.trees[s]:
                    self.trees[s].add(c)
                    self.tree_size[s] += 1
            return s
        raise ValueError(self.kind)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class Runner:
    def __init__(self, servers, nodes, policy_name, args, seed):
        self.servers, self.nodes, self.args, self.seed = servers, nodes, args, seed
        self.by = {n["id"]: n for n in nodes}
        self.pol, self.name = POLICIES[policy_name], policy_name
        self.R = len(servers.urls)
        self.router = Router(self.pol["route"], self.R)
        self.children = defaultdict(list)
        for n in nodes:
            for p in n["parents"]:
                self.children[p].append(n["id"])
        # a trajectory ends at a node without a same-trajectory child
        self.traj_end_node = {}
        for n in nodes:
            if not any(self.by[c]["traj"] == n["traj"] for c in self.children.get(n["id"], [])):
                self.traj_end_node[n["traj"]] = n["id"]
        self.group_trajs = defaultdict(set)
        self.group_roots = defaultdict(list)
        for n in nodes:
            self.group_trajs[n["group"]].add(n["traj"])
            if not n["parents"]:
                self.group_roots[n["group"]].append(n)
        self.traj_closed = set()
        self.group_open = {g: len(t) for g, t in self.group_trajs.items()}
        self.remaining_children = {nid: len(ch) for nid, ch in self.children.items()}
        self.hist, self.gen_tokens, self.rec = {}, {}, {}
        # admission
        self.admit_kind = self.pol["admit"]
        self.kv_cap = [int(b or 0) * servers.block_size for b in servers.num_blocks]
        self.ws = [0] * self.R
        self.traj_ws = {}
        self.group_srv = {}
        self.group_shared = {}
        self.active_groups = [0] * self.R
        self.active_roots = [0] * self.R
        self.reserve = int(args.ws_reserve)
        # expected per-trajectory private peak, taken from the previous rollout
        # step's statistics (a second sample of the workload stands in for it)
        if args.peak_est >= 0:
            base_prior = args.peak_est
        else:
            kind = self.pol.get("prior_kind", args.prior_kind)
            base_prior = getattr(args, "_prior_by_kind", {}).get(kind, getattr(args, "_peak_prior", 0))
        self.peak_est = int(base_prior * args.prior_scale * self.pol.get("prior_scale", 1.0))
        # mixed batches: the controller knows each group's task type and keeps one estimate per type
        self.part_prior = {}
        if not self.pol.get("global_prior") and getattr(args, "_prior_by_part", None):
            kind = self.pol.get("prior_kind", args.prior_kind)
            for part, d in args._prior_by_part.items():
                self.part_prior[part] = int(d.get(kind, 0) * args.prior_scale * self.pol.get("prior_scale", 1.0))
        self.lam = float(self.pol.get("lam", args.lam))
        # output length used for accounting: exact (known), previous-step mean, or the max_tokens bound
        self.gen_acct = getattr(args, "gen_acct", "exact")
        # program-level scheduler (thunder / canopy)
        self.prog_mode = self.admit_kind == "prog"
        self.shared_aware = bool(self.pol.get("shared_aware", False))
        self.group_place = bool(self.pol.get("group_place", False))
        self.use_prior = bool(self.pol.get("prior", False))
        # group lifetime: an admitted group's shared prefix stays accounted on its backend until all of
        # its trajectories close, and trajectories that have not started yet (forks, sub-agents) are
        # reserved at the previous-step estimate
        self.group_life = bool(self.pol.get("group_life", False))
        # trajectories whose first request is issued as soon as its parent finishes (no tool wait in
        # between) cannot be paused before they run, so they are reserved at admission; trajectories
        # that first wait on a tool are admitted optimistically and paused under pressure
        self.reserve_future = self.pol.get("reserve_future", "immediate")
        self.group_fut = defaultdict(set)
        for n in nodes:
            if n["parents"] and not any(self.by[p]["traj"] == n["traj"] for p in n["parents"]):
                if self.reserve_future == "all" or n.get("delay_ms", 0) <= 0:
                    self.group_fut[n["group"]].add(n["traj"])
        self.group_fut_regs = defaultdict(int)
        self.group_nregs = defaultdict(int)
        self.group_backend = {}
        self.backend_groups = [set() for _ in range(self.R)]
        self.progs = {}
        self.prog_queue = []
        self.backend_progs = [set() for _ in range(self.R)]
        self.group_home = {}
        self.group_shared_len = {}
        for g, roots in self.group_roots.items():
            self.group_shared_len[g] = max((n.get("shared_len", 0) for n in roots), default=0)
        self.parent_backend = {}
        self.sched_stats = defaultdict(int)
        self.ta_mode = bool(self.pol.get("ta_orig"))
        self.ta_queue = {}

    # -- admission -------------------------------------------------------------
    def _root_estimate(self, n):
        cur = max(0, len(n["fresh"]) - n.get("shared_len", 0)) + len(n["extra"]) + n["gen"]
        return max(cur + self.reserve, self.peak_est)

    def _pick_server(self, gid):
        r = self.router
        if r.kind in ("group", "canopy"):
            s = r.group_server.get(gid)
            if s is not None:
                return s
            if r.kind == "canopy":
                return min(range(self.R), key=lambda i: (self.ws[i] / max(1, self.kv_cap[i]), self.active_groups[i]))
            return min(range(self.R), key=lambda i: (self.active_roots[i], r.outstanding[i]))
        if self.admit_kind == "budget":
            return min(range(self.R), key=lambda i: self.ws[i] / max(1, self.kv_cap[i]))
        return min(range(self.R), key=lambda i: self.active_roots[i])

    async def admit_group(self, gid):
        roots = self.group_roots[gid]
        async with self.cv:
            while True:
                s = self._pick_server(gid)
                if self.admit_kind.startswith("cap"):
                    K = int(self.admit_kind[3:])
                    ok = self.active_roots[s] + len(roots) <= K or self.active_roots[s] == 0
                else:
                    shared = max((n.get("shared_len", 0) for n in roots), default=0)
                    est = shared + sum(self._root_estimate(n) for n in roots)
                    ok = self.ws[s] + est <= self.args.budget_frac * self.kv_cap[s] or self.active_groups[s] == 0
                if ok:
                    break
                await self.cv.wait()
            self.active_roots[s] += len(roots)
            self.active_groups[s] += 1
            self.group_srv[gid] = s
            if self.admit_kind == "budget":
                shared = max((n.get("shared_len", 0) for n in roots), default=0)
                self.group_shared[gid] = shared
                self.ws[s] += shared
                for n in roots:
                    self.traj_ws[n["traj"]] = self._root_estimate(n)
                    self.ws[s] += self.traj_ws[n["traj"]]
            r = self.router
            if r.kind in ("group", "canopy"):
                r.group_server[gid] = s
            elif r.kind == "lor":
                for n in roots:
                    r.traj_server[n["traj"]] = s

    async def on_request_done(self, n, hist_len):
        if self.admit_kind != "budget":
            return
        gid, traj = n["group"], n["traj"]
        s = self.group_srv.get(gid)
        if s is None:
            return
        async with self.cv:
            if traj in self.traj_closed:
                return
            new = max(max(0, hist_len - self.group_shared.get(gid, 0)) + self.reserve, self.peak_est)
            old = self.traj_ws.get(traj, 0)
            self.traj_ws[traj] = new
            self.ws[s] += new - old

    async def on_traj_end(self, n):
        gid, traj = n["group"], n["traj"]
        self.traj_closed.add(traj)
        self.group_open[gid] -= 1
        if self.admit_kind == "none":
            return
        async with self.cv:
            s = self.group_srv.get(gid)
            if s is not None:
                self.ws[s] -= self.traj_ws.pop(traj, 0)
                if self.group_open[gid] == 0:
                    self.ws[s] -= self.group_shared.pop(gid, 0)
                    self.active_roots[s] -= len(self.group_roots[gid])
                    self.active_groups[s] -= 1
            self.cv.notify_all()

    # -- program-level scheduler (ThunderAgent policy / Canopy) ------------------
    def _weight(self, p, now):
        if p["phase"] != "A":
            return 1.0
        t = now - p["acting_since"]
        if t <= self.args.decay_t0:
            return 1.0
        return 0.5 ** ((t - self.args.decay_t0) / self.args.decay_half)

    def _g(self, n):
        if self.gen_acct == "mean":
            return int(self.args._gen_mean)
        if self.gen_acct == "max":
            return int(self.args._gen_max)
        return n["gen"]

    def _pe(self, g):
        if self.part_prior:
            return self.part_prior.get(g.split(".", 1)[0], self.peak_est)
        return self.peak_est

    def _ceff(self, p, projected):
        # projected: expected peak from the previous rollout step's statistics
        if projected and self.use_prior:
            return max(p["c"], p["shared"] + self._pe(p["group"]))
        return p["c"]

    def _demand(self, b, now, projected=False):
        tot = 0.0
        groups = set()
        for traj in self.backend_progs[b]:
            p = self.progs[traj]
            w = 1.0 if projected else self._weight(p, now)
            c = self._ceff(p, projected)
            if self.shared_aware:
                groups.add(p["group"])
                tot += max(0, c - p["shared"]) * w
            else:
                tot += c * w
        if self.shared_aware:
            if self.group_life:
                groups |= self.backend_groups[b]
            tot += sum(self.group_shared_len.get(g, 0) for g in groups)
        if projected and self.group_life:
            tot += sum(self._future(g) for g in self.backend_groups[b])
        return tot

    def _future(self, g):
        """Reservation for trajectories of group g that have not been issued yet."""
        if not self.use_prior or not self.reserve_future:
            return 0
        n = len(self.group_fut[g]) - self.group_fut_regs[g]
        per = self._pe(g) + (0 if self.shared_aware else self.group_shared_len.get(g, 0))
        return max(0, n) * per

    def _need(self, p, b):
        c = self._ceff(p, True)
        g = p["group"]
        fut = self._future(g) if (self.group_life and g not in self.group_backend) else 0
        if not self.shared_aware:
            return c + fut
        group_active = any(self.progs[t]["group"] == g for t in self.backend_progs[b])
        if self.group_life:
            group_active = group_active or g in self.backend_groups[b]
        return max(0, c - p["shared"]) + (0 if group_active else p["shared"]) + fut

    def _activate(self, p, b):
        p["status"] = "active"
        p["backend"] = b
        self.backend_progs[b].add(p["traj"])
        if self.group_life and p["group"] not in self.group_backend:
            self.group_backend[p["group"]] = b
            self.backend_groups[b].add(p["group"])
        if self.group_place and p["group"] not in self.group_home:
            self.group_home[p["group"]] = b
        p["ev"].set()
        self.sched_stats["restores"] += 1

    def _schedule_tick(self):
        now = time.perf_counter()
        lam = self.lam
        cap = self.kv_cap
        dem = [self._demand(b, now) for b in range(self.R)]
        # pause acting programs on over-committed backends
        for b in range(self.R):
            if dem[b] <= lam * cap[b]:
                continue
            acting = [self.progs[t] for t in self.backend_progs[b] if self.progs[t]["phase"] == "A"]
            acting.sort(key=lambda p: -(p["c"] - (p["shared"] if self.shared_aware else 0)))
            for p in acting:
                if dem[b] <= lam * cap[b]:
                    break
                self.backend_progs[b].discard(p["traj"])
                p["status"] = "paused"
                p["ev"].clear()
                p["prev"] = b
                if not self.group_place:
                    p["backend"] = None
                self.prog_queue.append(p["traj"])
                self.sched_stats["pauses"] += 1
                dem[b] = self._demand(b, now)
        # restore / admit in FIFO order (admission uses projected demand when a prior is available)
        if self.use_prior:
            dem = [self._demand(b, now, projected=True) for b in range(self.R)]
        remaining = []
        for traj in self.prog_queue:
            p = self.progs[traj]
            if p["status"] not in ("wait", "paused"):
                continue
            if self.group_place and p["group"] in self.group_home:
                cands = [self.group_home[p["group"]]]
            else:
                order = sorted(range(self.R), key=lambda i: dem[i] / max(1, cap[i]))
                pref = p.get("prev")
                if pref is not None and pref in order:
                    order.remove(pref)
                    order.insert(0, pref)
                cands = order
            placed = False
            for b in cands:
                need = self._need(p, b)
                if dem[b] + need <= lam * cap[b] or not self.backend_progs[b]:
                    self._activate(p, b)
                    dem[b] += need
                    placed = True
                    break
            if not placed:
                remaining.append(traj)
        self.prog_queue = remaining
        self.sched_stats["ticks"] += 1

    # -- ThunderAgent's released scheduler (ThunderAgent/scheduler/router.py, backend/state.py) ----------
    TA_BUF = 100  # BUFFER_PER_PROGRAM

    def _ta_active_tokens(self, b):
        return sum(self.progs[t]["c"] for t in self.backend_progs[b])

    def _ta_remaining(self, b):
        return self.kv_cap[b] - self._ta_active_tokens(b) - self.TA_BUF * len(self.backend_progs[b])

    def _ta_register(self, p, b):
        p["status"] = "active"
        p["backend"] = b
        self.backend_progs[b].add(p["traj"])
        p["ev"].set()

    def _ta_pause(self, p):
        b = p["backend"]
        self.backend_progs[b].discard(p["traj"])
        p["prev"], p["backend"], p["status"] = b, None, "paused"
        p["ev"].clear()
        self.ta_queue[p["traj"]] = p["c"]
        self.sched_stats["pauses"] += 1

    async def _ta_acquire(self, n, prompt_len):
        traj = n["traj"]
        c = prompt_len + self._g(n)
        p = self.progs.get(traj)
        if p is None:
            p = {"traj": traj, "group": n["group"], "status": "wait", "backend": None, "prev": None, "c": c,
                 "phase": "R", "acting_since": 0.0, "ev": asyncio.Event(), "shared": 0, "steps": 1,
                 "marked": False}
            self.progs[traj] = p
            b = None
            if not self.ta_queue:
                fits = [i for i in range(self.R) if self._ta_remaining(i) >= c + self.TA_BUF]
                if fits:
                    b = min(fits, key=self._ta_active_tokens)
            if b is not None:
                self._ta_register(p, b)
            else:
                self.ta_queue[traj] = c
                self.sched_stats["queued_new"] += 1
        else:
            p["steps"] += 1
            p["c"] = max(p["c"], c)
            p["phase"] = "R"
        if p["status"] != "active":
            await p["ev"].wait()
        p["phase"] = "R"
        return p["backend"]

    def _ta_release(self, n, hist_len, closes):
        p = self.progs.get(n["traj"])
        if p is None:
            return
        self.parent_backend[n["id"]] = p["backend"]
        if closes:
            p["status"], p["phase"], p["marked"] = "done", None, False
            if p["backend"] is not None:
                self.backend_progs[p["backend"]].discard(n["traj"])
            return
        p["c"] = hist_len
        p["phase"] = "A"
        p["acting_since"] = time.perf_counter()
        if p["marked"]:
            p["marked"] = False
            self._ta_pause(p)

    def _ta_tick(self):
        # resume (greedy, best fit decreasing), then pause over-committed backends
        caps, total = [], 0
        for b in range(self.R):
            rem = self._ta_remaining(b)
            if rem > self.TA_BUF:
                caps.append([b, rem])
                total += rem
        if caps and self.ta_queue:
            reasoning, new, acting = [], [], []
            for traj, _ in sorted(self.ta_queue.items(), key=lambda kv: kv[1]):
                p = self.progs[traj]
                if p["steps"] == 1:
                    new.append(p)
                elif p["phase"] == "R":
                    reasoning.append(p)
                else:
                    acting.append(p)
            sel, cum = [], 0
            for p in reasoning + new + acting:
                req = p["c"] + self.TA_BUF
                if cum + req <= total:
                    sel.append(p)
                    cum += req
            sel.sort(key=lambda p: -p["c"])
            caps.sort(key=lambda x: -x[1])
            min_req = (sel[-1]["c"] + self.TA_BUF) if sel else 0
            for p in sel:
                if not caps or min_req > caps[0][1]:
                    break
                req = p["c"] + self.TA_BUF
                if req > caps[0][1]:
                    continue
                del self.ta_queue[p["traj"]]
                self._ta_register(p, caps[0][0])
                self.sched_stats["restores"] += 1
                caps[0][1] -= req
                if caps[0][1] > self.TA_BUF:
                    caps.sort(key=lambda x: -x[1])
                else:
                    caps.pop(0)
        for b in range(self.R):
            while self._ta_remaining(b) < 0:
                acting = [self.progs[t] for t in self.backend_progs[b] if self.progs[t]["phase"] == "A"]
                if acting:
                    self._ta_pause(min(acting, key=lambda p: p["c"]))
                    continue
                running = [self.progs[t] for t in self.backend_progs[b]
                           if self.progs[t]["phase"] == "R" and not self.progs[t]["marked"]]
                if running:
                    min(running, key=lambda p: p["c"])["marked"] = True
                    self.sched_stats["marks"] += 1
                    continue
                break
        self.sched_stats["ticks"] += 1

    async def _scheduler_loop(self):
        if self.ta_mode:
            dt = float(self.pol.get("sched_dt", 5.0))
            while not self._sched_stop:
                await asyncio.sleep(dt)
                self._ta_tick()
            return
        while not self._sched_stop:
            try:
                await asyncio.wait_for(self._kick.wait(), timeout=self.args.sched_dt)
            except asyncio.TimeoutError:
                pass
            self._kick.clear()
            self._schedule_tick()

    async def acquire(self, n, prompt_len):
        if self.ta_mode:
            return await self._ta_acquire(n, prompt_len)
        traj = n["traj"]
        p = self.progs.get(traj)
        if p is None:
            pref = None
            for par in n["parents"]:
                pref = self.parent_backend.get(par, pref)
            if self.pol.get("fork_shared"):
                # the descriptor's b grows at a fork: the history before the fork is shared by every branch
                self.group_shared_len[n["group"]] = max(self.group_shared_len.get(n["group"], 0),
                                                        n.get("shared_len", 0))
            p = {"traj": traj, "group": n["group"], "status": "wait", "backend": None, "prev": pref,
                 "c": prompt_len + self._g(n), "phase": "R", "acting_since": 0.0, "ev": asyncio.Event(),
                 "shared": self.group_shared_len.get(n["group"], 0)}
            self.progs[traj] = p
            self.group_nregs[n["group"]] += 1
            if traj in self.group_fut[n["group"]]:
                self.group_fut_regs[n["group"]] += 1
            self.prog_queue.append(traj)
            self._kick.set()
        else:
            p["c"] = max(p["c"], prompt_len + self._g(n))
        if p["status"] != "active":
            await p["ev"].wait()
        p["phase"] = "R"
        return p["backend"]

    def release_after(self, n, hist_len, closes):
        if self.ta_mode:
            return self._ta_release(n, hist_len, closes)
        p = self.progs.get(n["traj"])
        if p is None:
            return
        self.parent_backend[n["id"]] = p["backend"]
        if closes:
            p["status"] = "done"
            p["phase"] = None
            if p["backend"] is not None:
                self.backend_progs[p["backend"]].discard(n["traj"])
            self._kick.set()
        else:
            p["c"] = hist_len
            p["phase"] = "A"
            p["acting_since"] = time.perf_counter()

    # -- hints -------------------------------------------------------------------
    def _hints(self, n, closes_traj):
        h = self.pol["hint"]
        if h == "none":
            return None
        terminal = closes_traj and not self.children.get(n["id"])
        if h == "lifetime":
            group_done = False
            if terminal:
                group_done = all(t in self.traj_closed or t == n["traj"] for t in self.group_trajs[n["group"]])
            h = {"canopy_mode": "lifetime", "canopy_terminal": int(terminal),
                 "canopy_group_done": int(group_done), "canopy_shared_len": int(n.get("shared_len", 0))}
            if self.pol.get("pin") or self.pol.get("pp"):
                h["canopy_group"] = str(n["group"])
                h["canopy_pin"] = int(bool(self.pol.get("pin")))
                h["canopy_pp"] = int(bool(self.pol.get("pp")))
                h["canopy_pin_ttl"] = float(self.args.pin_ttl)
                h["canopy_pin_mode"] = self.pol.get("pin_mode", "evict")
            return h
        return {"canopy_mode": "ttl", "canopy_terminal": int(terminal), "canopy_traj": n["traj"],
                "canopy_pin_ttl": float(self.args.ttl)}

    # -- request -------------------------------------------------------------------
    def build_prompt(self, n):
        if n["base"] == "parent":
            p = n["parents"][0]
            base = self.hist[p]
        else:
            base = list(n["fresh"])
            if n["join_outputs"]:
                for p in n["parents"]:
                    base = base + self.gen_tokens[p]
        prompt = base + n["extra"]
        for p in n["parents"]:
            self.remaining_children[p] -= 1
            if self.remaining_children[p] == 0:
                self.hist.pop(p, None)
                self.gen_tokens.pop(p, None)
        return prompt

    async def issue(self, session, n, prompt, backend=None):
        s = backend if backend is not None else self.router.route(n, prompt)
        self.router.outstanding[s] += 1
        closes = self.traj_end_node.get(n["traj"]) == n["id"]
        hints = self._hints(n, closes)
        body = {"model": "m", "prompt": prompt, "max_tokens": n["gen"],
                "ignore_eos": True, "temperature": float(self.args.temperature), "top_p": 1.0,
                "return_token_ids": True}
        if hints:
            body["vllm_xargs"] = hints
        data = jdumps(body)
        url = self.servers.urls[s] + "/v1/completions"
        t0 = time.perf_counter()
        out = None
        for attempt in range(3):
            try:
                async with session.post(url, data=data, headers={"Content-Type": "application/json"}) as resp:
                    raw = await resp.read()
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}: {raw[:300]!r}")
                    out = jloads(raw)
                break
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(1.0)
        t1 = time.perf_counter()
        self.router.outstanding[s] -= 1
        toks = out["choices"][0].get("token_ids") or []
        if len(toks) != n["gen"]:
            raise RuntimeError(f"gen mismatch {n['id']}: {len(toks)} vs {n['gen']}")
        if self.children.get(n["id"]):
            self.hist[n["id"]] = prompt + toks
            self.gen_tokens[n["id"]] = toks
        usage = out.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        self.rec[n["id"]] = (s, t0 - self.t_start, t1 - self.t_start, len(prompt), cached)
        return len(prompt) + len(toks), closes

    async def run_node(self, session, nid):
        n = self.by[nid]
        if n["parents"]:
            for p in n["parents"]:
                await self.done[p].wait()
            if n["delay_ms"] > 0:
                await asyncio.sleep(n["delay_ms"] / 1000.0)
        elif self.admit_kind not in ("none", "prog"):
            await self.gate[n["group"]].wait()
        prompt = self.build_prompt(n)
        backend = None
        if self.prog_mode:
            backend = await self.acquire(n, len(prompt))
        hist_len, closes = await self.issue(session, n, prompt, backend)
        if self.prog_mode:
            self.release_after(n, hist_len, closes)
        self.done[nid].set()
        if self.prog_mode:
            if closes:
                self.traj_closed.add(n["traj"])
                self.group_open[n["group"]] -= 1
                if self.group_open[n["group"]] == 0 and n["group"] in self.group_backend:
                    self.backend_groups[self.group_backend.pop(n["group"])].discard(n["group"])
                    self._kick.set()
        elif closes:
            await self.on_traj_end(n)
        else:
            await self.on_request_done(n, hist_len)

    async def run(self):
        import aiohttp
        self.cv = asyncio.Condition()
        self.done = {n["id"]: asyncio.Event() for n in self.nodes}
        order, seen = [], set()
        for n in self.nodes:
            if not n["parents"] and n["group"] not in seen:
                seen.add(n["group"])
                order.append(n["group"])
        self.gate = {g: asyncio.Event() for g in order}
        conn = aiohttp.TCPConnector(limit=0)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=7200)
        self.t_start = time.perf_counter()
        self._kick = asyncio.Event()
        self._sched_stop = False
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            admitter = None
            if self.prog_mode:
                admitter = asyncio.create_task(self._scheduler_loop())
            elif self.admit_kind != "none":
                async def admit_all():
                    for g in order:
                        await self.admit_group(g)
                        self.gate[g].set()
                admitter = asyncio.create_task(admit_all())
            tasks = [asyncio.create_task(self.run_node(session, n["id"])) for n in self.nodes]
            try:
                await asyncio.gather(*tasks)
            finally:
                self._sched_stop = True
                for t in tasks:
                    if not t.done():
                        t.cancel()
                if admitter is not None:
                    admitter.cancel()
        self.makespan = time.perf_counter() - self.t_start
        return self.makespan


async def sample_kv(urls, stop, out):
    while not stop.is_set():
        t = time.time()
        rows = []
        for u in urls:
            m = await asyncio.to_thread(scrape, u)
            rows.append([m.get("vllm:kv_cache_usage_perc"), m.get("vllm:num_requests_running"),
                         m.get("vllm:num_requests_waiting")])
        out.append([t, rows])
        try:
            await asyncio.wait_for(stop.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


def reset_caches(urls):
    ok = True
    for u in urls:
        done = False
        for _ in range(60):
            try:
                req = urllib.request.Request(u + "/reset_prefix_cache", method="POST")
                with _HTTP.open(req, timeout=30) as r:
                    if r.status == 200:
                        done = True
                        break
            except Exception:
                time.sleep(1)
        ok = ok and done
    return ok


def read_engine_stats(d):
    out = []
    if d and os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(d, f)) as fh:
                        out.append(json.load(fh))
                except Exception:
                    pass
    return out


def run_policy(servers, nodes, pol, args, rep, wstats=None):
    reset_ok = reset_caches(servers.urls)
    time.sleep(1.0)
    before = [scrape(u) for u in servers.urls]
    runner = Runner(servers, nodes, pol, args, seed=args.seed * 1000 + max(rep, 0))

    async def main():
        stop = asyncio.Event()
        samples = []
        sampler = asyncio.create_task(sample_kv(servers.urls, stop, samples))
        try:
            ms = await asyncio.wait_for(runner.run(), timeout=args.run_timeout)
        finally:
            stop.set()
            await sampler
        return ms, samples

    ms, samples = asyncio.run(main())
    time.sleep(3.0)
    after = [scrape(u) for u in servers.urls]
    deltas = [metric_delta(b, a) for b, a in zip(before, after)]
    agg = defaultdict(float)
    for d in deltas:
        for k, v in d.items():
            agg[k] += v
    gen = agg.get("vllm:generation_tokens_total", 0.0)
    est = servers.read_all_stats() if hasattr(servers, "read_all_stats") else read_engine_stats(servers.stats_dir)
    latest = {}
    for e in est:
        d = (e.get("srv"), e.get("host"), e.get("dev"))
        if d not in latest or e.get("t", 0) > latest[d].get("t", 0):
            latest[d] = e
    it_sum = sum(e.get("sched_tokens", 0) for e in latest.values())
    per_srv = defaultdict(int)
    traj_end = defaultdict(float)
    for nid, (s, a, b, plen, cached) in runner.rec.items():
        per_srv[s] += 1
        t = runner.by[nid]["traj"]
        traj_end[t] = max(traj_end[t], b)
    te = sorted(traj_end.values())
    first_issue = min((a for (_, a, _, _, _) in runner.rec.values()), default=None)
    req_rec = {nid: [s, round(a, 4), round(b, 4), plen, cached] for nid, (s, a, b, plen, cached) in runner.rec.items()}
    kvs = [row[0] for smp in samples for row in smp[1] if row[0] is not None]
    res = {
        "policy": pol, "repeat": rep, "makespan_s": ms, "reset_ok": reset_ok,
        "requests": len(runner.rec),
        "prefill_computed_tokens": it_sum - gen,
        "prompt_local_compute_tokens": agg.get("vllm:prompt_tokens_by_source_total[local_compute]"),
        "prompt_tokens": agg.get("vllm:prompt_tokens_total"),
        "gen_tokens": gen,
        "preemptions": agg.get("vllm:num_preemptions_total"),
        "prefix_hit_rate": agg.get("vllm:prefix_cache_hits_total", 0) / max(1.0, agg.get("vllm:prefix_cache_queries_total", 0)),
        "metric_deltas_per_server": deltas,
        "requests_per_server": dict(per_srv),
        "traj_end_p50": te[len(te) // 2] if te else None,
        "traj_end_p90": te[int(len(te) * 0.9)] if te else None,
        "kv_usage_mean": statistics.mean(kvs) if kvs else None,
        "kv_samples": samples[:: max(1, len(samples) // 300)],
        "engine_stats": list(latest.values()),
        "engine_preempt_calls": sum(e.get("preempt_calls", 0) for e in latest.values()),
        "dead_blocks_head": sum(e.get("dead_blocks_head", 0) for e in latest.values()),
        "ttl_pins": sum(e.get("ttl_pins", 0) for e in latest.values()),
        "sched_stats": dict(runner.sched_stats),
        "first_issue_s": first_issue,
        "engine_totals": {k: sum(e.get(k, 0) for e in latest.values()) for k in (
            "group_pins", "group_pin_blocks", "group_pin_released_done", "group_pin_expired", "pp_preempts",
            "pp_kept_blocks", "pp_resumed", "pp_released_finished", "soft_releases", "soft_release_blocks",
            "idle_releases", "max_pinned_blocks", "dead_blocks_head", "live_blocks_tail", "preempt_calls")},
        "req_rec": req_rec,
    }
    if wstats:
        res["wasted_prefill_tokens"] = res["prefill_computed_tokens"] - wstats["min_prefill_tokens"]
    return res


def visible_gpus(spec=""):
    """GPU ids from ``spec`` (comma-separated), else CUDA_VISIBLE_DEVICES, else nvidia-smi."""
    if spec:
        return [g.strip() for g in spec.split(",") if g.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g for g in env.split(",") if g != ""]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _load_workload(spec):
    """A workload is a JSON object, given inline or as a path to a JSON file."""
    if os.path.isfile(spec):
        with open(spec) as f:
            return json.load(f)
    return json.loads(spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", required=True, help="experiment name; results go to <out-root>/<exp>/")
    ap.add_argument("--out-root", default="./results")
    ap.add_argument("--model", required=True, help="model path or Hugging Face name, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--workload", required=True, help="workload JSON (inline or a file path)")
    ap.add_argument("--policies", default="vanilla,par,sticky,canopy")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rep-start", type=int, default=0, help="first repeat index (rotates the policy order)")
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids (default: CUDA_VISIBLE_DEVICES or all GPUs)")
    ap.add_argument("--replicas", type=int, default=0, help="number of vLLM replicas (default: all assigned GPUs)")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs per vLLM replica")
    ap.add_argument("--workdir", default="", help="parent directory for server logs and caches (default: system temp)")
    ap.add_argument("--pin-ttl", type=float, default=60.0, help="safety timeout of a group pin (s)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--num-gpu-blocks", type=int, default=0, help="override the KV cache size (blocks per replica)")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--server-extra", default="", help="JSON list of extra vLLM server args")
    ap.add_argument("--multinode", action="store_true", help="start servers as Ray actors (spans nodes); needs --replicas")
    ap.add_argument("--budget-frac", type=float, default=0.9)
    ap.add_argument("--ws-reserve", type=int, default=1024)
    ap.add_argument("--peak-est", type=int, default=-1, help="-1 = derive from a previous-step sample of the workload")
    ap.add_argument("--prior-kind", default="curve", choices=("curve", "peak"),
                    help="curve: peak of the previous step's batch working-set curve per trajectory; peak: mean per-trajectory peak")
    ap.add_argument("--prior-scale", type=float, default=1.0, help="multiply the prior (sensitivity study)")
    ap.add_argument("--ttl", type=float, default=2.0, help="pin duration of the ttl policy (s)")
    ap.add_argument("--lam", type=float, default=1.0, help="memory watermark of the program-level schedulers (ThunderAgent default 1.0)")
    ap.add_argument("--gen-acct", default="exact", choices=["exact", "mean", "max"],
                    help="output length used by admission accounting")
    ap.add_argument("--sched-dt", type=float, default=0.5)
    ap.add_argument("--decay-t0", type=float, default=1.0)
    ap.add_argument("--decay-half", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-timeout", type=float, default=5400)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args(argv)

    w = _load_workload(args.workload)
    unknown = [p for p in args.policies.split(",") if p and p not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown policies: {unknown}")
    if args.multinode:
        if args.tp > 1:
            raise SystemExit("--tp > 1 is only supported for single-node runs")
        if not args.replicas:
            raise SystemExit("--multinode needs --replicas")
        gpus = [f"ray{i}" for i in range(args.replicas)]
    else:
        gpus = visible_gpus(args.gpus)
        if args.replicas:
            gpus = gpus[: args.replicas * args.tp]
        if args.tp > 1:
            gpus = [",".join(gpus[i:i + args.tp]) for i in range(0, len(gpus) - args.tp + 1, args.tp)]
    if not gpus:
        raise SystemExit("no GPUs assigned")
    out_dir = os.path.join(args.out_root, args.exp)
    os.makedirs(out_dir, exist_ok=True)
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix=f"canopy_{args.exp}_", dir=args.workdir or None)
    nodes = build_workload(w, seed=args.seed)
    wstats = workload_stats(nodes)
    # prior from a *different* sample of the same distribution (a previous step)
    prior_stats = workload_stats(build_workload(w, seed=args.seed + 7919))
    key = "curve_peak_per_traj" if args.prior_kind == "curve" else "mean_traj_private_peak"
    args._prior_by_kind = {"curve": prior_stats["curve_peak_per_traj"], "peak": prior_stats["mean_traj_private_peak"]}
    if w["kind"] == "mixed":
        args._prior_by_part = {}
        for i, sub in enumerate(w["parts"]):
            st = workload_stats(build_workload(sub, seed=args.seed + 7919 + i))
            args._prior_by_part[f"p{i}"] = {"curve": st["curve_peak_per_traj"], "peak": st["mean_traj_private_peak"]}
    args._peak_prior = int(prior_stats[key] * args.prior_scale)
    args._gen_mean = prior_stats["gen_tokens"] / max(1, prior_stats["requests"])
    gspec = w.get("gen")
    if gspec is None:
        args._gen_max = 4 * args._gen_mean
    else:
        args._gen_max = gspec["lognorm"][3] if isinstance(gspec, dict) else (
            max(gspec) if isinstance(gspec, (list, tuple)) else gspec)
    wstats["gen_acct"] = args.gen_acct
    wstats["gen_mean_prev"] = args._gen_mean
    wstats["gen_max"] = args._gen_max
    wstats["peak_prior_from_previous_step"] = args._peak_prior
    wstats["prior_kind"] = args.prior_kind
    wstats["prior_scale"] = args.prior_scale
    print("WORKLOAD", json.dumps({"exp": args.exp, **wstats}), flush=True)
    servers = RayServers(args, args.replicas, workdir) if args.multinode else Servers(args, gpus, workdir)
    results = {"exp": args.exp, "args": vars(args), "workload": w, "workload_stats": wstats,
               "gpus": gpus, "host": socket.gethostname(), "started": time.time(), "runs": []}
    policies = [p for p in args.policies.split(",") if p]
    try:
        t0 = time.time()
        servers.start()
        results["server_start_s"] = time.time() - t0
        results["num_gpu_blocks"] = servers.num_blocks
        results["block_size"] = servers.block_size
        print("SERVERS", json.dumps({"blocks": servers.num_blocks, "bs": servers.block_size,
                                     "start_s": round(results["server_start_s"], 1)}), flush=True)
        if not args.no_warmup:
            ww = dict(w)
            if w["kind"] == "mixed":
                ww["parts"] = [dict(p, groups=max(1, min(int(p["groups"]), len(gpus)))) for p in w["parts"]]
            else:
                ww["groups"] = max(1, min(int(w["groups"]), len(gpus) * 2))
            wn = build_workload(ww, seed=12345)
            for pol in policies[:2]:
                run_policy(servers, wn, pol, args, rep=-1)
        for rep in range(args.rep_start, args.rep_start + args.repeats):
            k = rep % len(policies)
            for pol in policies[k:] + policies[:k]:
                try:
                    r = run_policy(servers, nodes, pol, args, rep, wstats)
                except Exception as e:
                    r = {"policy": pol, "repeat": rep, "error": f"{type(e).__name__}: {e}",
                         "tb": traceback.format_exc()[-3000:]}
                results["runs"].append(r)
                brief = {k2: (round(r[k2], 3) if isinstance(r.get(k2), float) else r.get(k2))
                         for k2 in ("policy", "repeat", "makespan_s", "prefill_computed_tokens",
                                    "wasted_prefill_tokens", "preemptions", "engine_preempt_calls", "prefix_hit_rate",
                                    "kv_usage_mean", "dead_blocks_head", "ttl_pins", "traj_end_p50", "error")}
                print("RUN", json.dumps(brief), flush=True)
                with open(os.path.join(out_dir, "results.json"), "w") as f:
                    json.dump(results, f)
    finally:
        results["finished"] = time.time()
        if args.multinode:
            import ray
            for i, a in enumerate(getattr(servers, "actors", [])):
                try:
                    with open(os.path.join(out_dir, f"server_{i}.log.tail"), "w") as f:
                        f.write(ray.get(a.log_tail.remote(300), timeout=60))
                except Exception:
                    pass
        servers.stop()
        if not args.multinode:
            for i in range(len(servers.urls)):
                try:
                    with open(os.path.join(out_dir, f"server_{i}.log.tail"), "w") as f:
                        f.write(tail(os.path.join(workdir, f"server_{i}.log"), 300))
                except Exception:
                    pass
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f)
    summ = defaultdict(list)
    for r in results["runs"]:
        if r.get("makespan_s") is not None:
            summ[r["policy"]].append(r["makespan_s"])
    print("SUMMARY", json.dumps({"exp": args.exp, "median_makespan": {k: round(statistics.median(v), 3) for k, v in summ.items()},
                                 "n": {k: len(v) for k, v in summ.items()}}), flush=True)


if __name__ == "__main__":
    main()
