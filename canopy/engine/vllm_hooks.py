# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this file re-implement internal logic of vLLM v0.20
# (vllm/v1/core/single_type_kv_cache_manager.py, vllm/v1/core/kv_cache_utils.py).
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for Canopy.
"""Canopy engine-side hooks for the vLLM V1 KV cache manager.

Written against vLLM 0.20.1. The hooks
patch these classes when their modules are imported:

  vllm.v1.core.single_type_kv_cache_manager.SingleTypeKVCacheManager.free
  vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots
  vllm.v1.core.sched.scheduler.Scheduler._free_blocks / schedule /
      _preempt_request / reset_prefix_cache

Call :func:`install` in-process, or load the hooks in every Python process of a
vLLM server (the engine core runs in a spawned child) through
``sitecustomize.py`` in this directory: put the directory on ``PYTHONPATH`` and
set ``CANOPY_ENGINE_HOOKS=1``.

The hooks are inert unless a request carries hints in
``SamplingParams.extra_args`` (``vllm_xargs`` in the OpenAI-compatible API).
The hints carry the request descriptor of the RL controller:

  canopy_mode        "lifetime" | "ttl"
  canopy_group       str   -- group id (required by canopy_pin)
  canopy_shared_len  int   -- tokens of the prefix shared with live siblings / children
  canopy_terminal    int   -- the trajectory issues no further request after this one
  canopy_group_done  int   -- no live request of the group remains after this one
  canopy_pin         int   -- keep the shared prefix of a live group out of the eviction queue
  canopy_pp          int   -- on preemption, keep the shared prefix and release only the suffix
  canopy_pin_mode    str   -- "evict" | "admit" (see below)
  canopy_pin_ttl     float -- safety timeout of a group pin since its last touch (lifetime mode);
                              pin duration of a finished request's blocks (ttl mode)
  canopy_traj        str   -- trajectory id (ttl mode: the next turn releases the pin)

Lifetime mode, finished requests: blocks that no future request can use
(private blocks of a terminal trajectory; group-shared blocks once the group is
done) go to the head of the eviction queue; other blocks keep vLLM's LRU order.

Group pin: while a group is live, the first ``canopy_shared_len`` tokens of
its finished requests hold one extra reference, so they never enter the
eviction queue.  The group's last request releases the pin.

Prefix-preserving preemption: before vLLM preempts a running request, the
cached blocks of its shared prefix get one extra reference.  vLLM then frees
the request as usual (suffix blocks become evictable, tail first) and resets
its progress; when the request is scheduled again, prefix caching finds the
retained prefix and only the suffix is recomputed.  The extra reference is
dropped once the request runs again.

Pins are soft.  ``canopy_pin_mode`` "evict" (default): any allocation that
fails releases pins (oldest touch first) and retries, so a pinned prefix is
only the last thing evicted.  "admit": only allocations for RUNNING requests
release pins (a pin never causes a preemption, but it holds memory against
newly admitted requests).  If nothing is running and requests are waiting, all
pins are released.  Group pins also expire after ``canopy_pin_ttl`` seconds
without a touch.

TTL mode is the time-to-live baseline: the blocks of a finished, non-terminal
request stay referenced for ``canopy_pin_ttl`` seconds, or until the next
request of the same trajectory has been admitted.

If ``CANOPY_ENGINE_STATS_DIR`` is set, each engine process writes its counters
(``STATS``) to a JSON file there about once per second and when it goes idle.
"""

import importlib.abc
import json
import os
import socket
import sys
import time

_STATS_DIR = os.environ.get("CANOPY_ENGINE_STATS_DIR", "")
_HOST = socket.gethostname()

STATS = {
    "finished_frees": 0,
    "lifetime_frees": 0,
    "dead_blocks_head": 0,
    "live_blocks_tail": 0,
    "ttl_pins": 0,
    "ttl_pin_blocks": 0,
    "ttl_expired": 0,
    "ttl_released_by_next_turn": 0,
    "ttl_forced_release": 0,
    "group_pins": 0,
    "group_pin_blocks": 0,
    "group_pin_released_done": 0,
    "group_pin_expired": 0,
    "pp_preempts": 0,
    "pp_kept_blocks": 0,
    "pp_resumed": 0,
    "pp_released_finished": 0,
    "soft_releases": 0,
    "soft_release_blocks": 0,
    "idle_releases": 0,
    "max_pinned_blocks": 0,
    "resets": 0,
    "sched_tokens": 0,
    "sched_steps": 0,
    "preempt_calls": 0,
}
_LAST_DUMP = [0.0]
_CUR_HINT = {"rid": None, "hint": None}
_PINS = []  # ttl pins: {"expire": t, "blocks": [...], "traj": str, "pool": BlockPool}
_PIN_BY_TRAJ = {}
_GPINS = {}  # group -> {"pool", "blocks", "ids", "touch", "ttl", "mode"}
_PPPINS = {}  # request_id -> {"pool", "blocks", "touch", "mode"}


def _dump_stats():
    if not _STATS_DIR:
        return
    try:
        os.makedirs(_STATS_DIR, exist_ok=True)
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "x").replace(",", "_")
        path = os.path.join(_STATS_DIR, f"engine_{dev}_{os.getpid()}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"pid": os.getpid(), "dev": dev, "host": _HOST, "t": time.time(), **STATS}, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _prepend_n(queue, blocks):
    """Insert blocks right after the fake head (they are evicted first)."""
    if not blocks:
        return
    head = queue.fake_free_list_head
    first = head.next_free_block
    prev = head
    for b in blocks:
        b.prev_free_block = prev
        prev.next_free_block = b
        prev = b
    prev.next_free_block = first
    first.prev_free_block = prev
    queue.num_free_blocks += len(blocks)


def _hint_of(request):
    sp = getattr(request, "sampling_params", None)
    if sp is None:
        return None
    ea = getattr(sp, "extra_args", None)
    if not ea or "canopy_mode" not in ea:
        return None
    return ea


def _pinned_blocks():
    return sum(len(p["blocks"]) for p in _GPINS.values()) + sum(len(p["blocks"]) for p in _PPPINS.values())


def _unref(pool, blocks, dead=False):
    """Drop one reference; blocks that reach zero go to the queue tail (or head if dead)."""
    freed = []
    for b in blocks:
        b.ref_cnt -= 1
        if b.ref_cnt == 0 and not b.is_null:
            freed.append(b)
    if dead:
        _prepend_n(pool.free_block_queue, freed)
    else:
        freed.reverse()  # suffix first, as vLLM frees
        pool.free_block_queue.append_n(freed)
    return len(freed)


def _release_pin(pin, reason):
    pool = pin["pool"]
    live = []
    for b in pin["blocks"]:
        b.ref_cnt -= 1
        if b.ref_cnt == 0 and not b.is_null:
            live.append(b)
    pool.free_block_queue.append_n(live)
    STATS[reason] += 1
    _PIN_BY_TRAJ.pop(pin.get("traj"), None)


def _release_group(g, reason, dead=False):
    pin = _GPINS.pop(g, None)
    if pin is None:
        return 0
    STATS[reason] += 1
    return _unref(pin["pool"], pin["blocks"], dead=dead)


def _release_pp(rid, reason):
    pin = _PPPINS.pop(rid, None)
    if pin is None:
        return 0
    STATS[reason] += 1
    return _unref(pin["pool"], pin["blocks"])


def _release_oldest(only_mode=None):
    """Release the least recently touched pin (group or preemption). Returns freed block count or -1."""
    cand = [(p["touch"], "g", k) for k, p in _GPINS.items() if only_mode is None or p.get("mode") == only_mode]
    cand += [(p["touch"], "p", k) for k, p in _PPPINS.items() if only_mode is None or p.get("mode") == only_mode]
    if not cand:
        return -1
    _, kind, key = min(cand)
    pin = _GPINS.pop(key) if kind == "g" else _PPPINS.pop(key)
    n = _unref(pin["pool"], pin["blocks"])
    STATS["soft_releases"] += 1
    STATS["soft_release_blocks"] += n
    return n


def _note_pinned():
    n = _pinned_blocks()
    if n > STATS["max_pinned_blocks"]:
        STATS["max_pinned_blocks"] = n


def _patch_single_type(mod):
    cls = mod.SingleTypeKVCacheManager
    orig_free = cls.free
    if getattr(orig_free, "_canopy_hooked", False):
        return

    def free(self, request_id, *args, **kwargs):
        hint = _CUR_HINT["hint"] if _CUR_HINT["rid"] == request_id else None
        if hint is None or hint.get("canopy_mode") != "lifetime":
            return orig_free(self, request_id, *args, **kwargs)
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        bs = self.block_size
        n_shared = int(hint.get("canopy_shared_len", 0)) // bs
        terminal = bool(hint.get("canopy_terminal", 0))
        group_done = bool(hint.get("canopy_group_done", 0))
        group = hint.get("canopy_group")
        now = time.monotonic()
        if hint.get("canopy_pin") and group is not None and not group_done and n_shared > 0:
            pin = _GPINS.get(group)
            if pin is None:
                pin = {"pool": self.block_pool, "blocks": [], "ids": set(), "touch": now,
                       "ttl": float(hint.get("canopy_pin_ttl", 60.0)),
                       "mode": hint.get("canopy_pin_mode", "evict")}
                _GPINS[group] = pin
                STATS["group_pins"] += 1
            pin["touch"] = now
            for b in req_blocks[:n_shared]:
                if b.is_null or b.block_hash is None or b.block_id in pin["ids"]:
                    continue
                b.ref_cnt += 1
                pin["blocks"].append(b)
                pin["ids"].add(b.block_id)
                STATS["group_pin_blocks"] += 1
            _note_pinned()
        elif group is not None and group in _GPINS:
            _GPINS[group]["touch"] = now
        dead, live = [], []
        for i, b in enumerate(req_blocks):
            b.ref_cnt -= 1
            if b.ref_cnt == 0 and not b.is_null:
                if terminal and (i >= n_shared or group_done):
                    dead.append(b)
                else:
                    live.append(b)
        q = self.block_pool.free_block_queue
        live.reverse()  # suffix first among live blocks (vLLM default order)
        q.append_n(live)
        _prepend_n(q, dead)
        STATS["lifetime_frees"] += 1
        STATS["dead_blocks_head"] += len(dead)
        STATS["live_blocks_tail"] += len(live)
        if group_done and group is not None:
            _release_group(group, "group_pin_released_done", dead=True)

    free._canopy_hooked = True
    cls.free = free


def _patch_kv_manager(mod):
    cls = mod.KVCacheManager
    orig_alloc = cls.allocate_slots
    if getattr(orig_alloc, "_canopy_hooked", False):
        return

    def allocate_slots(self, request, *args, **kwargs):
        out = orig_alloc(self, request, *args, **kwargs)
        if out is not None or not (_GPINS or _PPPINS):
            return out
        only = None if getattr(getattr(request, "status", None), "name", "") == "RUNNING" else "evict"
        while out is None and (_GPINS or _PPPINS):
            if _release_oldest(only) < 0:
                break
            out = orig_alloc(self, request, *args, **kwargs)
        return out

    allocate_slots._canopy_hooked = True
    cls.allocate_slots = allocate_slots


def _patch_scheduler(mod):
    cls = mod.Scheduler
    orig_free_blocks = cls._free_blocks
    if getattr(orig_free_blocks, "_canopy_hooked", False):
        return
    orig_schedule = cls.schedule
    orig_reset = getattr(cls, "reset_prefix_cache", None)
    orig_preempt = cls._preempt_request

    def _free_blocks(self, request, *args, **kwargs):
        try:
            if request.request_id in _PPPINS:
                _release_pp(request.request_id, "pp_released_finished")
            return _free_blocks_inner(self, request, *args, **kwargs)
        finally:
            if not self.running and not self.waiting:
                _dump_stats()

    def _free_blocks_inner(self, request, *args, **kwargs):
        hint = _hint_of(request)
        STATS["finished_frees"] += 1
        if hint is None:
            return orig_free_blocks(self, request, *args, **kwargs)
        mode = hint.get("canopy_mode")
        if mode == "ttl" and not hint.get("canopy_terminal", 0):
            ttl = float(hint.get("canopy_pin_ttl", 0.0))
            if ttl > 0:
                try:
                    mgr = self.kv_cache_manager.coordinator.single_type_managers[0]
                    blocks = list(mgr.req_to_blocks.get(request.request_id, []))
                    for b in blocks:
                        b.ref_cnt += 1
                    traj = hint.get("canopy_traj")
                    old = _PIN_BY_TRAJ.get(traj)
                    if old is not None:
                        _PINS.remove(old)
                        _release_pin(old, "ttl_released_by_next_turn")
                    pin = {"expire": time.monotonic() + ttl, "blocks": blocks,
                           "traj": traj, "pool": self.kv_cache_manager.block_pool}
                    _PINS.append(pin)
                    if traj is not None:
                        _PIN_BY_TRAJ[traj] = pin
                    STATS["ttl_pins"] += 1
                    STATS["ttl_pin_blocks"] += len(blocks)
                except Exception:
                    pass
            return orig_free_blocks(self, request, *args, **kwargs)
        _CUR_HINT["rid"] = request.request_id
        _CUR_HINT["hint"] = hint
        try:
            return orig_free_blocks(self, request, *args, **kwargs)
        finally:
            _CUR_HINT["rid"] = None
            _CUR_HINT["hint"] = None

    def schedule(self, *args, **kwargs):
        now = time.monotonic()
        if _PINS:
            keep = []
            for p in _PINS:
                if p["expire"] <= now:
                    _release_pin(p, "ttl_expired")
                else:
                    keep.append(p)
            _PINS[:] = keep
            # A pinned trajectory whose next turn is waiting releases its pin
            # once that turn has been admitted (its blocks are referenced again).
            if _PIN_BY_TRAJ:
                for req in self.running:
                    h = _hint_of(req)
                    if h is None:
                        continue
                    pin = _PIN_BY_TRAJ.get(h.get("canopy_traj"))
                    if pin is not None and req.num_computed_tokens > 0:
                        _PINS.remove(pin)
                        _release_pin(pin, "ttl_released_by_next_turn")
        if _GPINS:
            for g in [g for g, p in _GPINS.items() if now - p["touch"] > p["ttl"]]:
                _release_group(g, "group_pin_expired")
        if (_GPINS or _PPPINS) and not self.running and self.waiting:
            while _GPINS or _PPPINS:
                _release_oldest()
            STATS["idle_releases"] += 1
        out = orig_schedule(self, *args, **kwargs)
        if _PPPINS:
            for req in self.running:
                if req.request_id in _PPPINS and req.num_computed_tokens > 0:
                    _release_pp(req.request_id, "pp_resumed")
        try:
            STATS["sched_tokens"] += int(out.total_num_scheduled_tokens)
            STATS["sched_steps"] += 1
        except Exception:
            pass
        if now - _LAST_DUMP[0] > 1.0:
            _LAST_DUMP[0] = now
            _dump_stats()
        return out

    def reset_prefix_cache(self, *a, **kw):
        for p in list(_PINS):
            _release_pin(p, "ttl_forced_release")
        _PINS.clear()
        _PIN_BY_TRAJ.clear()
        for g in list(_GPINS):
            _release_group(g, "group_pin_expired")
        for rid in list(_PPPINS):
            _release_pp(rid, "pp_released_finished")
        STATS["resets"] += 1
        _dump_stats()
        for k in STATS:
            if k != "resets":
                STATS[k] = 0
        return orig_reset(self, *a, **kw)

    def _preempt_request(self, request, *args, **kwargs):
        STATS["preempt_calls"] += 1
        hint = _hint_of(request)
        if hint is not None and hint.get("canopy_pp") and request.request_id not in _PPPINS:
            try:
                mgr = self.kv_cache_manager.coordinator.single_type_managers[0]
                bs = mgr.block_size
                blocks = mgr.req_to_blocks.get(request.request_id, [])
                n_keep = min(int(hint.get("canopy_shared_len", 0)) // bs,
                             int(request.num_computed_tokens) // bs, len(blocks))
                kept = []
                for b in blocks[:n_keep]:
                    if b.is_null or b.block_hash is None:
                        break  # only a contiguous cached prefix can be found again on resume
                    b.ref_cnt += 1
                    kept.append(b)
                if kept:
                    _PPPINS[request.request_id] = {"pool": self.kv_cache_manager.block_pool, "blocks": kept,
                                                   "touch": time.monotonic(),
                                                   "mode": hint.get("canopy_pin_mode", "evict")}
                    STATS["pp_preempts"] += 1
                    STATS["pp_kept_blocks"] += len(kept)
                    _note_pinned()
            except Exception:
                pass
        return orig_preempt(self, request, *args, **kwargs)

    _free_blocks._canopy_hooked = True
    cls._free_blocks = _free_blocks
    cls.schedule = schedule
    cls._preempt_request = _preempt_request
    if orig_reset is not None:
        cls.reset_prefix_cache = reset_prefix_cache


_TARGETS = {
    "vllm.v1.core.single_type_kv_cache_manager": _patch_single_type,
    "vllm.v1.core.kv_cache_manager": _patch_kv_manager,
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
}


def _apply(module, fn):
    try:
        fn(module)
        sys.stderr.write(f"[canopy] patched {module.__name__} pid={os.getpid()}\n")
    except Exception as e:  # never break the server
        sys.stderr.write(f"[canopy] patch FAILED {module.__name__}: {e!r}\n")


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fn):
        self.inner = inner
        self.fn = fn

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _apply(module, self.fn)


class _Finder(importlib.abc.MetaPathFinder):
    # Marks the finder across module copies (e.g. ``vllm_hooks`` loaded by
    # sitecustomize and ``canopy.engine.vllm_hooks`` imported later), so the
    # hook is installed once and finders never delegate to each other.
    _canopy_engine_hooks = True

    def find_spec(self, name, path, target=None):
        fn = _TARGETS.get(name)
        if fn is None:
            return None
        for finder in sys.meta_path:
            if getattr(finder, "_canopy_engine_hooks", False):
                continue
            spec = finder.find_spec(name, path, target) if hasattr(finder, "find_spec") else None
            if spec is not None:
                spec.loader = _Loader(spec.loader, fn)
                return spec
        return None


def install(stats_dir=None):
    """Install the hooks in this process. Idempotent.

    Already-imported target modules are patched immediately; the others are
    patched on first import. Returns True if a new import hook was added.
    """
    global _STATS_DIR
    if stats_dir is not None:
        _STATS_DIR = stats_dir
    for name, fn in _TARGETS.items():
        module = sys.modules.get(name)
        if module is not None:
            _apply(module, fn)
    if any(getattr(f, "_canopy_engine_hooks", False) for f in sys.meta_path):
        return False
    sys.meta_path.insert(0, _Finder())
    return True
