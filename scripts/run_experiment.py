#!/usr/bin/env python3
# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Run experiments from configs/experiments.json on this machine.

Each experiment starts its own vLLM servers on local GPUs and replays its
workload under the listed policies (see canopy/bench/runner.py).  Results go to
``<out-root>/<experiment>/results.json``.

Examples (from the repository root):
    python scripts/run_experiment.py --list
    python scripts/run_experiment.py m_tree
    python scripts/run_experiment.py 'abr_*' --gpus 0,1,2,3
    python scripts/run_experiment.py m_tree --model /path/to/Qwen3-8B --dry-run
    python scripts/run_experiment.py m_tree -- --enforce-eager   # extra runner arguments after "--"
"""

import argparse
import fnmatch
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from canopy.bench.runner import visible_gpus  # noqa: E402

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "experiments.json")


def runner_args(exp, model, out_root, gpus, extra):
    args = ["--exp", exp["exp"], "--model", model, "--out-root", out_root,
            "--workload", json.dumps(exp["workload"], separators=(",", ":"))]
    for k, v in exp.get("args", {}).items():
        if isinstance(v, bool):
            if v:
                args.append("--" + k)
        else:
            args += ["--" + k, str(v)]
    if gpus:
        args += ["--gpus", ",".join(gpus)]
    return args + list(extra)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="Run experiments from configs/experiments.json on this machine.")
    ap.add_argument("names", nargs="*", help="experiment names or shell-style patterns (e.g. 'm_*')")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--list", action="store_true", help="list the experiments and exit")
    ap.add_argument("--model", default="", help="override the model of every selected experiment (e.g. a local path)")
    ap.add_argument("--out-root", default="./results")
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids to use (default: CUDA_VISIBLE_DEVICES or all GPUs)")
    ap.add_argument("--dry-run", action="store_true", help="print the runner commands without running them")
    ap.add_argument("--force", action="store_true", help="run on fewer GPUs than the experiment specifies")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        exps = json.load(f)
    if args.list:
        for e in exps:
            where = "multi-node (Ray)" if e.get("multinode") else "single node"
            print(f'{e["exp"]:20s} {e["gpus"]:3d} GPUs  {where:17s} {e["model"]:20s} {e["args"].get("policies", "")}')
        return 0
    selected = [e for e in exps if any(fnmatch.fnmatchcase(e["exp"], pat) for pat in args.names)]
    if not selected:
        ap.error("no experiment matches; use --list to see the names")

    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    available = visible_gpus(args.gpus)
    failed = []
    for e in selected:
        need = int(e["gpus"])
        gpus = []
        if e.get("multinode"):
            print(f'[{e["exp"]}] multi-node experiment: needs a running Ray cluster with {need} GPUs '
                  f'(the runner calls ray.init(address="auto")).', flush=True)
        else:
            gpus = available[:need]
            if len(gpus) < need:
                msg = f'[{e["exp"]}] needs {need} GPUs, {len(available)} visible'
                if args.dry_run:
                    print(msg, flush=True)
                elif args.force and gpus:
                    print(msg + "; running on fewer replicas than specified", flush=True)
                else:
                    print(msg + "; skipped (use --force to run on fewer GPUs)", flush=True)
                    failed.append(e["exp"])
                    continue
        cmd = [sys.executable, "-m", "canopy.bench.runner"] + runner_args(
            e, args.model or e["model"], args.out_root, gpus, extra)
        print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        if args.dry_run:
            continue
        if subprocess.call(cmd, env=env) != 0:
            failed.append(e["exp"])
    if failed:
        print("failed or skipped: " + ", ".join(failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
