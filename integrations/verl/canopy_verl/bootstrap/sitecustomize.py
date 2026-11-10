# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this file re-implement internal logic of vLLM v0.13.0
# (vllm/v1/core/block_pool.py).
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for Canopy.
"""Engine-core bootstrap for the Canopy verl integration (vLLM 0.13).

vLLM V1 runs ``EngineCore`` -- and with it the block pool and the scheduler --
in a subprocess started with ``multiprocessing`` (spawn).  Patches applied by
importing :mod:`canopy_verl` reach the driver and the Ray workers, but not that
subprocess.  Python imports ``sitecustomize`` at every interpreter start, so
with this directory on ``PYTHONPATH`` this file also runs in the subprocess.
It installs an import hook that patches, on first import:

  * ``vllm.v1.core.block_pool.BlockPool`` -- the prefix lock (same logic as
    :mod:`canopy_verl.vllm_patch.prefix_lock`, inlined because the package
    cannot be imported this early);
  * ``vllm.v1.engine.core.EngineCore`` -- utility methods reachable through
    ``call_utility_async``: prefix-lock set / unset / stats, and with
    ``CANOPY_BOOTSTRAP_B2=1`` partial-preemption tag set / unset / stats.

The partial-preemption utilities only set and clear tags on the scheduler's
requests.  The ``Scheduler._preempt_request`` wrap that acts on the tags comes
from :func:`canopy_verl.vllm_patch.apply_partial_preempt_patch`, which runs in
processes that import :mod:`canopy_verl`.

Nothing happens unless ``CANOPY_BOOTSTRAP_B1=1``.  A failure prints a
``[CANOPY-BOOTSTRAP]`` line and leaves the process unpatched.  Patched classes
are marked (``_canopy_bootstrap_done``) and never patched twice by this hook.

Ray runtime environment, for example::

    env_vars:
      CANOPY_BOOTSTRAP_B1: "1"
      CANOPY_BOOTSTRAP_B2: "1"      # admission mode b2 only
      PYTHONPATH: "<repo>/integrations/verl/canopy_verl/bootstrap:<repo>/integrations/verl"

Each patched process prints::

    [CANOPY-BOOTSTRAP pid=<PID>] _CanopyFinder installed on sys.meta_path
    [CANOPY-BOOTSTRAP pid=<PID>] BlockPool PATCHED via sitecustomize meta_path hook
    [CANOPY-BOOTSTRAP pid=<PID>] EngineCore PATCHED (utility methods attached)
"""

from __future__ import annotations

import os
import sys


_MARKER_PREFIX = f"[CANOPY-BOOTSTRAP pid={os.getpid()}]"


def _emit(msg: str) -> None:
    try:
        print(f"{_MARKER_PREFIX} {msg}", flush=True)
    except Exception:
        pass


if os.environ.get("CANOPY_BOOTSTRAP_B1", "0") == "1":
    try:
        import time
        import importlib.abc
        import importlib.util

        # ------------------------------------------------------------
        # vllm.v1.core.block_pool.BlockPool: prefix lock
        # ------------------------------------------------------------
        def _canopy_ensure_state(self):
            if not hasattr(self, "_canopy_locked_block_ids"):
                self._canopy_locked_block_ids = set()
                self._canopy_lock_deadlines = {}
                self._canopy_counters = {
                    "locked_blocks": 0,
                    "lock_events_total": 0,
                    "unlock_events_total": 0,
                    "evictions_skipped_due_to_lock": 0,
                    "locked_eviction_forced": 0,
                    "ttl_expired_events_total": 0,
                    "get_new_blocks_invocations": 0,
                    "get_new_blocks_with_locks": 0,
                }

        def _set_prefix_lock(self, block_ids, ttl_sec=60.0):
            _canopy_ensure_state(self)
            now = time.time()
            added = 0
            for raw in block_ids:
                try:
                    bid = int(raw)
                except Exception:
                    continue
                if bid not in self._canopy_locked_block_ids:
                    added += 1
                self._canopy_locked_block_ids.add(bid)
                self._canopy_lock_deadlines[bid] = now + float(ttl_sec)
            self._canopy_counters["lock_events_total"] += added
            self._canopy_counters["locked_blocks"] = len(
                self._canopy_locked_block_ids
            )
            return added

        def _unset_prefix_lock(self, block_ids):
            _canopy_ensure_state(self)
            removed = 0
            for raw in block_ids:
                try:
                    bid = int(raw)
                except Exception:
                    continue
                if bid in self._canopy_locked_block_ids:
                    self._canopy_locked_block_ids.discard(bid)
                    self._canopy_lock_deadlines.pop(bid, None)
                    removed += 1
            self._canopy_counters["unlock_events_total"] += removed
            self._canopy_counters["locked_blocks"] = len(
                self._canopy_locked_block_ids
            )
            return removed

        def _expire_ttl(self):
            _canopy_ensure_state(self)
            now = time.time()
            expired = [
                bid
                for bid, dl in self._canopy_lock_deadlines.items()
                if dl <= now
            ]
            for bid in expired:
                self._canopy_locked_block_ids.discard(bid)
                self._canopy_lock_deadlines.pop(bid, None)
            if expired:
                self._canopy_counters["ttl_expired_events_total"] += len(
                    expired
                )
                self._canopy_counters["locked_blocks"] = len(
                    self._canopy_locked_block_ids
                )
            return len(expired)

        def _get_state(self):
            _canopy_ensure_state(self)
            return dict(self._canopy_counters)

        _ORIG_GET_NEW_BLOCKS = {}

        def _canopy_get_new_blocks_wrap(self, num_blocks):
            _canopy_ensure_state(self)
            self._canopy_counters["get_new_blocks_invocations"] += 1
            self._canopy_expire_ttl()
            orig = _ORIG_GET_NEW_BLOCKS["fn"]
            locked_ids = self._canopy_locked_block_ids
            if not locked_ids or num_blocks <= 0:
                return orig(self, num_blocks)
            self._canopy_counters["get_new_blocks_with_locks"] += 1
            q = getattr(self, "free_block_queue", None)
            if q is None:
                return orig(self, num_blocks)
            remove_fn = getattr(q, "remove", None)
            append_fn = getattr(q, "append", None)
            # vLLM 0.13's FreeKVCacheBlockQueue has get_all_free_blocks() but no
            # __iter__ / __len__.
            get_all = getattr(q, "get_all_free_blocks", None)
            if (
                not callable(remove_fn)
                or not callable(append_fn)
                or not callable(get_all)
            ):
                return orig(self, num_blocks)
            skipped = []
            try:
                snapshot = get_all()
            except Exception:
                return orig(self, num_blocks)
            blocks_arr = getattr(self, "blocks", None)
            id_to_idx = None
            for block in snapshot:
                matched = False
                bid_attr = getattr(block, "block_id", None)
                if bid_attr is not None:
                    try:
                        if int(bid_attr) in locked_ids:
                            matched = True
                    except Exception:
                        pass
                elif blocks_arr is not None:
                    # No block_id attribute: match by index in BlockPool.blocks.
                    if id_to_idx is None:
                        try:
                            id_to_idx = {id(b): i for i, b in enumerate(blocks_arr)}
                        except Exception:
                            id_to_idx = {}
                    idx = id_to_idx.get(id(block))
                    if idx is not None and idx in locked_ids:
                        matched = True
                if matched:
                    try:
                        remove_fn(block)
                        skipped.append(block)
                    except Exception:
                        continue
            try:
                result = orig(self, num_blocks)
            except Exception:
                # Not enough unlocked blocks: put the locked ones back and let
                # vLLM allocate from the whole queue.
                for block in skipped:
                    try:
                        append_fn(block)
                    except Exception:
                        pass
                self._canopy_counters["locked_eviction_forced"] += 1
                return orig(self, num_blocks)
            for block in skipped:
                try:
                    append_fn(block)
                except Exception:
                    continue
            self._canopy_counters["evictions_skipped_due_to_lock"] += len(
                skipped
            )
            return result

        def _patch_block_pool(module):
            cls = getattr(module, "BlockPool", None)
            if cls is None:
                return False
            if getattr(cls, "_canopy_bootstrap_done", False):
                return True
            _ORIG_GET_NEW_BLOCKS["fn"] = cls.get_new_blocks
            cls.set_prefix_lock = _set_prefix_lock
            cls.unset_prefix_lock = _unset_prefix_lock
            cls._canopy_expire_ttl = _expire_ttl
            cls.get_prefix_lock_state = _get_state
            cls.get_new_blocks = _canopy_get_new_blocks_wrap
            cls._canopy_bootstrap_done = True
            _emit("BlockPool PATCHED via sitecustomize meta_path hook")
            return True

        # ------------------------------------------------------------
        # vllm.v1.engine.core.EngineCore: utility methods for
        # AsyncMPClient.call_utility_async.  They run inside the engine-core
        # process, with direct access to self.scheduler.
        # ------------------------------------------------------------
        def _engine_core_canopy_set_prefix_lock(self, request_id, ttl_sec=60.0):
            try:
                km = self.scheduler.kv_cache_manager
                bid_groups = km.get_block_ids(request_id)
            except Exception as exc:
                return {
                    "locked": 0,
                    "error": f"engine_core_get_block_ids: {type(exc).__name__}: {exc}",
                    "request_id": request_id,
                }
            flat = []
            try:
                for group in bid_groups or []:
                    for bid in group or []:
                        try:
                            flat.append(int(bid))
                        except Exception:
                            pass
            except Exception:
                try:
                    flat = [int(x) for x in (bid_groups or [])]
                except Exception:
                    flat = []
            try:
                added = km.block_pool.set_prefix_lock(flat, ttl_sec=ttl_sec)
            except Exception as exc:
                return {
                    "locked": 0,
                    "error": f"engine_core_set_prefix_lock: {type(exc).__name__}: {exc}",
                    "request_id": request_id,
                }
            return {
                "locked": int(added),
                "total_blocks": len(flat),
                "request_id": request_id,
            }

        def _engine_core_canopy_unset_prefix_lock(self, request_id):
            try:
                km = self.scheduler.kv_cache_manager
                bid_groups = km.get_block_ids(request_id)
            except Exception as exc:
                return {
                    "unlocked": 0,
                    "error": f"engine_core_get_block_ids: {type(exc).__name__}: {exc}",
                    "request_id": request_id,
                }
            flat = []
            try:
                for group in bid_groups or []:
                    for bid in group or []:
                        try:
                            flat.append(int(bid))
                        except Exception:
                            pass
            except Exception:
                try:
                    flat = [int(x) for x in (bid_groups or [])]
                except Exception:
                    flat = []
            try:
                removed = km.block_pool.unset_prefix_lock(flat)
            except Exception as exc:
                return {
                    "unlocked": 0,
                    "error": f"engine_core_unset_prefix_lock: {type(exc).__name__}: {exc}",
                    "request_id": request_id,
                }
            return {
                "unlocked": int(removed),
                "total_blocks": len(flat),
                "request_id": request_id,
            }

        def _engine_core_canopy_get_prefix_lock_stats(self):
            try:
                return self.scheduler.kv_cache_manager.block_pool.get_prefix_lock_state()
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

        # Partial-preemption tags (CANOPY_BOOTSTRAP_B2=1): attributes on
        # scheduler.requests[rid] that the Scheduler._preempt_request wrap in
        # canopy_verl.vllm_patch.partial_preempt reads.

        # Same counter dict as partial_preempt.py; whichever runs first creates it.
        def _ensure_b2_counters(scheduler_cls):
            if not hasattr(scheduler_cls, "_canopy_partial_preempt_counters"):
                scheduler_cls._canopy_partial_preempt_counters = {
                    "tags_set": 0,
                    "tags_unset": 0,
                    "preempt_invocations": 0,
                    "preempt_partial_path_taken": 0,
                    "succeeded": 0,
                    "fallback_to_full": 0,
                    "kept_blocks_total": 0,
                    "released_blocks_total": 0,
                    "skip_due_to_no_common_prefix": 0,
                    "skip_due_to_no_tag": 0,
                    "skip_due_to_ttl_expired": 0,
                    "skip_due_to_zero_keep_after_safety": 0,
                    "ttl_expired_at_preempt": 0,
                    "coordinator_unsync_warnings": 0,
                    "partial_free_exceptions": 0,
                    "block_overlap_with_b1_lock": 0,
                }

        def _engine_core_canopy_set_partial_preempt(
            self, request_id, keep_n_blocks, ttl_sec=60.0
        ):
            """Tag scheduler.requests[request_id] with keep_n and a deadline."""
            try:
                sched = self.scheduler
                req = sched.requests.get(request_id) if hasattr(sched, "requests") else None
                if req is None:
                    return {
                        "tagged": 0,
                        "error": "request_not_in_scheduler",
                        "request_id": request_id,
                    }
                try:
                    req._canopy_partial_preempt_keep_n = int(keep_n_blocks)
                    req._canopy_partial_preempt_deadline = time.time() + float(ttl_sec)
                except Exception:
                    object.__setattr__(
                        req, "_canopy_partial_preempt_keep_n", int(keep_n_blocks)
                    )
                    object.__setattr__(
                        req, "_canopy_partial_preempt_deadline",
                        time.time() + float(ttl_sec),
                    )
                _ensure_b2_counters(type(sched))
                type(sched)._canopy_partial_preempt_counters["tags_set"] += 1
                return {
                    "tagged": 1,
                    "request_id": request_id,
                    "keep_n_blocks": int(keep_n_blocks),
                    "ttl_sec": float(ttl_sec),
                }
            except Exception as exc:
                return {
                    "tagged": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request_id": request_id,
                }

        def _engine_core_canopy_unset_partial_preempt(self, request_id):
            try:
                sched = self.scheduler
                req = sched.requests.get(request_id) if hasattr(sched, "requests") else None
                if req is not None and hasattr(req, "_canopy_partial_preempt_keep_n"):
                    try:
                        del req._canopy_partial_preempt_keep_n
                    except Exception:
                        pass
                    if hasattr(req, "_canopy_partial_preempt_deadline"):
                        try:
                            del req._canopy_partial_preempt_deadline
                        except Exception:
                            pass
                    _ensure_b2_counters(type(sched))
                    type(sched)._canopy_partial_preempt_counters["tags_unset"] += 1
                return {"untagged": 1, "request_id": request_id}
            except Exception as exc:
                return {
                    "untagged": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "request_id": request_id,
                }

        def _engine_core_canopy_get_partial_preempt_stats(self):
            try:
                sched_cls = type(self.scheduler)
                _ensure_b2_counters(sched_cls)
                return dict(sched_cls._canopy_partial_preempt_counters)
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

        _CANOPY_B2_ENABLED = os.environ.get("CANOPY_BOOTSTRAP_B2", "0") == "1"

        def _attach_b2_utilities(cls):
            cls.canopy_set_partial_preempt_for_request = (
                _engine_core_canopy_set_partial_preempt
            )
            cls.canopy_unset_partial_preempt_for_request = (
                _engine_core_canopy_unset_partial_preempt
            )
            cls.canopy_get_partial_preempt_stats = (
                _engine_core_canopy_get_partial_preempt_stats
            )
            cls._canopy_b2_bootstrap_done = True
            _emit("EngineCore PATCHED b2 (partial-preempt utility methods attached)")

        def _patch_engine_core(module):
            cls = getattr(module, "EngineCore", None)
            if cls is None:
                return False
            if getattr(cls, "_canopy_bootstrap_done", False):
                if _CANOPY_B2_ENABLED and not getattr(cls, "_canopy_b2_bootstrap_done", False):
                    _attach_b2_utilities(cls)
                return True
            cls.canopy_set_prefix_lock_for_request = (
                _engine_core_canopy_set_prefix_lock
            )
            cls.canopy_unset_prefix_lock_for_request = (
                _engine_core_canopy_unset_prefix_lock
            )
            cls.canopy_get_prefix_lock_stats = (
                _engine_core_canopy_get_prefix_lock_stats
            )
            cls._canopy_bootstrap_done = True
            _emit("EngineCore PATCHED (utility methods attached)")
            if _CANOPY_B2_ENABLED:
                _attach_b2_utilities(cls)
            return True

        # ------------------------------------------------------------
        # Import hook that patches the target modules on first import
        # ------------------------------------------------------------
        _TARGETS = {
            "vllm.v1.core.block_pool": _patch_block_pool,
            "vllm.v1.engine.core": _patch_engine_core,
        }
        _WRAPPED = set()

        class _PostImportLoader(importlib.abc.Loader):
            def __init__(self, wrapped, callback):
                self._wrapped = wrapped
                self._callback = callback

            def create_module(self, spec):
                try:
                    return self._wrapped.create_module(spec)
                except AttributeError:
                    return None

            def exec_module(self, module):
                self._wrapped.exec_module(module)
                try:
                    self._callback(module)
                except Exception as e:
                    _emit(f"post-import callback FAIL: {type(e).__name__}: {e}")

        class _CanopyFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name not in _TARGETS or name in _WRAPPED:
                    return None
                # Resolve the real spec with the remaining finders (this finder
                # is removed meanwhile to avoid recursion).
                try:
                    idx = sys.meta_path.index(self)
                except ValueError:
                    idx = -1
                if idx >= 0:
                    sys.meta_path.pop(idx)
                try:
                    spec = importlib.util.find_spec(name)
                finally:
                    if idx >= 0:
                        sys.meta_path.insert(idx, self)
                if spec is None or spec.loader is None:
                    return None
                _WRAPPED.add(name)
                spec.loader = _PostImportLoader(spec.loader, _TARGETS[name])
                return spec

        _finder = _CanopyFinder()
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)
            _emit("_CanopyFinder installed on sys.meta_path")

        # Modules imported before this file ran (unusual) are patched now.
        for name, callback in list(_TARGETS.items()):
            if name in sys.modules:
                try:
                    callback(sys.modules[name])
                    _WRAPPED.add(name)
                except Exception as e:
                    _emit(f"retro-apply {name} FAIL: {type(e).__name__}: {e}")

    except Exception as outer_exc:
        _emit(f"bootstrap OUTER FAIL: {type(outer_exc).__name__}: {outer_exc}")
