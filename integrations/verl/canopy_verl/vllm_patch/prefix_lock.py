# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this file re-implement internal logic of vLLM v0.13.0
# (vllm/v1/core/block_pool.py).
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for Canopy.
"""Shared-prefix retention for vLLM 0.13 (V1 engine): a soft lock in ``BlockPool``.

vLLM 0.13.0 members used here:

  * ``BlockPool.get_new_blocks(num_blocks)`` -- allocation entry point (wrapped)
  * ``BlockPool.free_block_queue`` -- ``FreeKVCacheBlockQueue`` with
    ``get_all_free_blocks()``, ``remove(block)`` and ``append(block)``
  * ``KVCacheManager.get_block_ids(request_id) -> tuple[list[int], ...]``

Soft lock, not a hard pin:

  * Locked block ids are kept in ``BlockPool._canopy_locked_block_ids``, each
    with a deadline (TTL).
  * ``get_new_blocks`` temporarily removes the locked blocks from the free
    queue, runs vLLM's allocation, and appends the locked blocks back at the
    tail of the queue (evicted last).  If the allocation cannot be served
    without them, it falls back to vLLM's allocation, which may evict locked
    blocks, and counts ``locked_eviction_forced``.

Each step is wrapped in try/except; on unexpected vLLM internals the pool
behaves like unpatched vLLM instead of failing the run.  The engine-core
subprocess gets the same ``BlockPool`` patch from
``canopy_verl/bootstrap/sitecustomize.py``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Module state, so that apply / revert are exact.
_PATCHED_BLOCK_POOL: bool = False
_PATCHED_ASYNC_LLM: bool = False
_ORIG_GET_NEW_BLOCKS = None

_APPLY_MARKER_PRINTED: bool = False


# ---------------------------------------------------------------------
# BlockPool methods (installed on the class)
# ---------------------------------------------------------------------


def _canopy_ensure_state(self) -> None:
    """Create the per-instance lock state on first use.

    Pools created before the patch was applied get their fields here.
    """
    if not hasattr(self, "_canopy_locked_block_ids"):
        self._canopy_locked_block_ids = set()
    if not hasattr(self, "_canopy_lock_deadlines"):
        self._canopy_lock_deadlines = {}
    if not hasattr(self, "_canopy_counters"):
        self._canopy_counters = {
            "locked_blocks": 0,
            "lock_events_total": 0,
            "unlock_events_total": 0,
            "evictions_skipped_due_to_lock": 0,
            "locked_eviction_forced": 0,
            "ttl_expired_events_total": 0,
            # Every call of the wrapped get_new_blocks, and those that found
            # locks.  Non-zero invocations show the patch is active even when no
            # locked block was ever skipped (no memory pressure).
            "get_new_blocks_invocations": 0,
            "get_new_blocks_with_locks": 0,
        }


def _canopy_set_prefix_lock(self, block_ids: Iterable[int], ttl_sec: float = 60.0) -> int:
    """Lock block ids for ``ttl_sec``.  Returns the number of newly locked ids (re-locking refreshes the deadline)."""
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
    self._canopy_counters["locked_blocks"] = len(self._canopy_locked_block_ids)
    return added


def _canopy_unset_prefix_lock(self, block_ids: Iterable[int]) -> int:
    """Unlock block ids.  Returns the number removed."""
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
    self._canopy_counters["locked_blocks"] = len(self._canopy_locked_block_ids)
    return removed


def _canopy_expire_ttl(self) -> int:
    """Drop locks whose deadline has passed.  Returns the number expired."""
    _canopy_ensure_state(self)
    now = time.time()
    expired = [bid for bid, dl in self._canopy_lock_deadlines.items() if dl <= now]
    for bid in expired:
        self._canopy_locked_block_ids.discard(bid)
        self._canopy_lock_deadlines.pop(bid, None)
    if expired:
        self._canopy_counters["ttl_expired_events_total"] += len(expired)
        self._canopy_counters["locked_blocks"] = len(self._canopy_locked_block_ids)
    return len(expired)


def _canopy_get_prefix_lock_state(self) -> dict[str, int]:
    """Copy of the lock counters."""
    _canopy_ensure_state(self)
    return dict(self._canopy_counters)


# ---------------------------------------------------------------------
# get_new_blocks wrapper: skip locked blocks
# ---------------------------------------------------------------------


def _canopy_get_new_blocks(self, num_blocks: int):
    """``BlockPool.get_new_blocks`` that does not hand out locked blocks.

      1. No locks: call vLLM's method.
      2. Otherwise remove the locked blocks from the free queue (found through
         ``get_all_free_blocks()``; the queue has no ``__iter__``), call vLLM's
         method, and append the removed blocks back.
    """
    _canopy_ensure_state(self)
    self._canopy_counters["get_new_blocks_invocations"] += 1
    self._canopy_expire_ttl()
    locked_ids = self._canopy_locked_block_ids
    if not locked_ids or num_blocks <= 0:
        return _ORIG_GET_NEW_BLOCKS(self, num_blocks)
    self._canopy_counters["get_new_blocks_with_locks"] += 1

    q = getattr(self, "free_block_queue", None)
    if q is None:
        return _ORIG_GET_NEW_BLOCKS(self, num_blocks)

    skipped: list[Any] = []
    try:
        remove_fn = getattr(q, "remove", None)
        append_fn = getattr(q, "append", None)
        get_all = getattr(q, "get_all_free_blocks", None)
        if (
            not callable(remove_fn)
            or not callable(append_fn)
            or not callable(get_all)
        ):
            return _ORIG_GET_NEW_BLOCKS(self, num_blocks)

        try:
            snapshot = get_all()
        except Exception:
            return _ORIG_GET_NEW_BLOCKS(self, num_blocks)

        # Blocks without a ``block_id`` attribute are matched by their index in
        # ``self.blocks``.
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
        result = _ORIG_GET_NEW_BLOCKS(self, num_blocks)
        for block in skipped:
            try:
                append_fn(block)
            except Exception:
                continue
        self._canopy_counters["evictions_skipped_due_to_lock"] += len(skipped)
        return result
    except Exception as exc:
        # Not enough unlocked blocks (or an unexpected API): put the locked
        # blocks back and let vLLM allocate from the whole queue.
        self._canopy_counters["locked_eviction_forced"] += 1
        logger.warning(
            "[CANOPY-VLLM-PATCH] get_new_blocks fallback: %s: %s",
            type(exc).__name__,
            exc,
        )
        for block in skipped:
            try:
                q.append(block)
            except Exception:
                continue
        return _ORIG_GET_NEW_BLOCKS(self, num_blocks)


# ---------------------------------------------------------------------
# AsyncLLM RPC methods
# ---------------------------------------------------------------------


async def _canopy_async_llm_set_prefix_lock_for_request(
    self, request_id: str, ttl_sec: float = 60.0
) -> dict[str, Any]:
    # In-process engine core (unit tests, synchronous engines): direct access.
    km = getattr(self.engine_core, "kv_cache_manager", None)
    if km is not None:
        try:
            bid_groups = km.get_block_ids(request_id)
        except Exception as exc:
            return {
                "locked": 0,
                "error": f"get_block_ids: {type(exc).__name__}: {exc}",
                "request_id": request_id,
                "path": "direct",
            }
        flat: list[int] = []
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
                "error": f"set_prefix_lock: {type(exc).__name__}: {exc}",
                "request_id": request_id,
                "path": "direct",
            }
        return {
            "locked": int(added),
            "total_blocks": len(flat),
            "request_id": request_id,
            "path": "direct",
        }

    # Multiprocess engine (vLLM's default): ``engine_core`` is a client of the
    # engine-core subprocess.  The bootstrap sitecustomize adds the utility
    # method ``canopy_set_prefix_lock_for_request`` to ``EngineCore`` there,
    # which is reachable through ``call_utility_async``.
    call_fn = getattr(self.engine_core, "call_utility_async", None)
    if call_fn is None:
        return {
            "locked": 0,
            "error": "engine_core_proxy_no_call_utility_async",
            "engine_core_type": type(self.engine_core).__name__,
            "request_id": request_id,
        }
    try:
        result = await call_fn(
            "canopy_set_prefix_lock_for_request",
            request_id,
            float(ttl_sec),
        )
    except Exception as exc:
        return {
            "locked": 0,
            "error": f"call_utility_async: {type(exc).__name__}: {exc}",
            "engine_core_type": type(self.engine_core).__name__,
            "request_id": request_id,
            "path": "rpc",
        }
    if isinstance(result, dict):
        result.setdefault("path", "rpc")
        return result
    return {
        "locked": 0,
        "error": f"rpc_unexpected_result: {type(result).__name__}",
        "request_id": request_id,
        "path": "rpc",
    }


async def _canopy_async_llm_unset_prefix_lock_for_request(
    self, request_id: str
) -> dict[str, Any]:
    km = getattr(self.engine_core, "kv_cache_manager", None)
    if km is None:
        call_fn = getattr(self.engine_core, "call_utility_async", None)
        if call_fn is None:
            return {
                "unlocked": 0,
                "error": "engine_core_proxy_no_call_utility_async",
                "engine_core_type": type(self.engine_core).__name__,
                "request_id": request_id,
            }
        try:
            result = await call_fn(
                "canopy_unset_prefix_lock_for_request", request_id
            )
        except Exception as exc:
            return {
                "unlocked": 0,
                "error": f"call_utility_async: {type(exc).__name__}: {exc}",
                "request_id": request_id,
                "path": "rpc",
            }
        if isinstance(result, dict):
            result.setdefault("path", "rpc")
            return result
        return {
            "unlocked": 0,
            "error": f"rpc_unexpected_result: {type(result).__name__}",
            "request_id": request_id,
            "path": "rpc",
        }
    try:
        bid_groups = km.get_block_ids(request_id)
    except Exception as exc:
        return {
            "unlocked": 0,
            "error": f"get_block_ids: {type(exc).__name__}: {exc}",
            "request_id": request_id,
        }
    flat: list[int] = []
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
            "error": f"unset_prefix_lock: {type(exc).__name__}: {exc}",
            "request_id": request_id,
        }
    return {
        "unlocked": int(removed),
        "total_blocks": len(flat),
        "request_id": request_id,
    }


async def _canopy_async_llm_get_prefix_lock_stats(self) -> dict[str, Any]:
    try:
        km = getattr(self.engine_core, "kv_cache_manager", None)
        if km is not None:
            result = km.block_pool.get_prefix_lock_state()
            result["path"] = "direct"
            return result
    except Exception as exc:
        return {"error": f"direct: {type(exc).__name__}: {exc}"}
    call_fn = getattr(self.engine_core, "call_utility_async", None)
    if call_fn is None:
        return {"error": "engine_core_proxy_no_call_utility_async"}
    try:
        result = await call_fn("canopy_get_prefix_lock_stats")
    except Exception as exc:
        return {"error": f"call_utility_async: {type(exc).__name__}: {exc}"}
    if isinstance(result, dict):
        result.setdefault("path", "rpc")
        return result
    return {"error": f"rpc_unexpected_result: {type(result).__name__}"}


# ---------------------------------------------------------------------
# Apply / revert
# ---------------------------------------------------------------------


def apply_prefix_lock_patch() -> bool:
    """Install the BlockPool and AsyncLLM patches.  Idempotent.

    Returns ``True`` on success (or if already applied), ``False`` when vLLM
    cannot be imported.
    """
    global _PATCHED_BLOCK_POOL, _PATCHED_ASYNC_LLM, _ORIG_GET_NEW_BLOCKS
    global _APPLY_MARKER_PRINTED

    if _PATCHED_BLOCK_POOL and _PATCHED_ASYNC_LLM:
        return True

    try:
        from vllm.v1.core.block_pool import BlockPool
    except Exception as exc:
        logger.warning(
            "[CANOPY-VLLM-PATCH] BlockPool import failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        return False

    if not _PATCHED_BLOCK_POOL:
        if _ORIG_GET_NEW_BLOCKS is None:
            _ORIG_GET_NEW_BLOCKS = BlockPool.get_new_blocks
        BlockPool.set_prefix_lock = _canopy_set_prefix_lock
        BlockPool.unset_prefix_lock = _canopy_unset_prefix_lock
        BlockPool._canopy_expire_ttl = _canopy_expire_ttl
        BlockPool.get_prefix_lock_state = _canopy_get_prefix_lock_state
        BlockPool.get_new_blocks = _canopy_get_new_blocks
        _PATCHED_BLOCK_POOL = True

    if not _PATCHED_ASYNC_LLM:
        try:
            from vllm.v1.engine.async_llm import AsyncLLM
        except Exception as exc:
            logger.warning(
                "[CANOPY-VLLM-PATCH] AsyncLLM import failed (BlockPool patched, AsyncLLM not): %s: %s",
                type(exc).__name__,
                exc,
            )
            return _PATCHED_BLOCK_POOL

        AsyncLLM.set_prefix_lock_for_request = (
            _canopy_async_llm_set_prefix_lock_for_request
        )
        AsyncLLM.unset_prefix_lock_for_request = (
            _canopy_async_llm_unset_prefix_lock_for_request
        )
        AsyncLLM.get_prefix_lock_stats = _canopy_async_llm_get_prefix_lock_stats
        _PATCHED_ASYNC_LLM = True

    if not _APPLY_MARKER_PRINTED:
        _APPLY_MARKER_PRINTED = True
        print(
            "[CANOPY-VLLM-PATCH] applied BlockPool + AsyncLLM patches",
            flush=True,
        )

    return _PATCHED_BLOCK_POOL and _PATCHED_ASYNC_LLM


def applied_prefix_lock_patch() -> bool:
    return _PATCHED_BLOCK_POOL and _PATCHED_ASYNC_LLM


def revert_prefix_lock_patch() -> None:
    """Undo the patch (used by tests)."""
    global _PATCHED_BLOCK_POOL, _PATCHED_ASYNC_LLM, _ORIG_GET_NEW_BLOCKS
    global _APPLY_MARKER_PRINTED

    try:
        from vllm.v1.core.block_pool import BlockPool
    except Exception:
        BlockPool = None  # type: ignore

    if BlockPool is not None and _PATCHED_BLOCK_POOL:
        for name in (
            "set_prefix_lock",
            "unset_prefix_lock",
            "_canopy_expire_ttl",
            "get_prefix_lock_state",
        ):
            if hasattr(BlockPool, name):
                try:
                    delattr(BlockPool, name)
                except Exception:
                    pass
        if _ORIG_GET_NEW_BLOCKS is not None:
            BlockPool.get_new_blocks = _ORIG_GET_NEW_BLOCKS
        _ORIG_GET_NEW_BLOCKS = None
    _PATCHED_BLOCK_POOL = False

    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception:
        AsyncLLM = None  # type: ignore

    if AsyncLLM is not None and _PATCHED_ASYNC_LLM:
        for name in (
            "set_prefix_lock_for_request",
            "unset_prefix_lock_for_request",
            "get_prefix_lock_stats",
        ):
            if hasattr(AsyncLLM, name):
                try:
                    delattr(AsyncLLM, name)
                except Exception:
                    pass
    _PATCHED_ASYNC_LLM = False
    _APPLY_MARKER_PRINTED = False


__all__ = [
    "apply_prefix_lock_patch",
    "applied_prefix_lock_patch",
    "revert_prefix_lock_patch",
]
