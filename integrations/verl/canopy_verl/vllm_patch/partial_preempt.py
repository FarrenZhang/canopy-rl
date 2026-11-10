# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this file re-implement internal logic of vLLM v0.13.0
# (vllm/v1/core/block_pool.py, vllm/v1/core/sched/scheduler.py).
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for Canopy.
"""Prefix-preserving partial preemption for vLLM 0.13 (V1 engine).

When vLLM preempts a running request it frees all of the request's KV blocks
and recomputes the whole context when the request resumes.  With this patch, a
request that carries a partial-preemption tag keeps its leading blocks
(normally the tree's shared prefix) and frees only its private suffix, so only
the suffix is recomputed on resume.

vLLM 0.13.0 members used here:

  * ``Scheduler._preempt_request(request, timestamp)`` -- frees the request
    (``kv_cache_manager.free``), marks it PREEMPTED, resets
    ``num_computed_tokens`` to 0 and puts it back at the head of the waiting
    queue.  Wrapped by :func:`_canopy_preempt_request_partial`.
  * ``KVCacheManager.get_block_ids(request_id) -> tuple[list[int], ...]``.
  * ``BlockPool.free_blocks(ordered_blocks)`` -- decrements ``ref_cnt`` and
    appends blocks that reach 0 to the free queue;
    :func:`_canopy_block_pool_free_blocks_partial` does the same for a
    caller-chosen subset.

A tag is two attributes on the vLLM ``Request``:
``_canopy_partial_preempt_keep_n`` (blocks to keep) and
``_canopy_partial_preempt_deadline`` (expiry time).  The admission coordinator
sets them with ``AsyncLLM.set_partial_preempt_for_request``, which reaches the
``EngineCore`` utility method installed by ``bootstrap/sitecustomize.py``.

The wrap falls back to vLLM's original ``_preempt_request`` whenever a check
fails or anything raises, and counts the reason (``skip_*``,
``fallback_to_full``).  ``partial_free`` releases blocks directly instead of
going through ``KVCacheCoordinator.free`` (counted in
``coordinator_unsync_warnings``).

``apply_partial_preempt_patch`` returns ``False`` when vLLM cannot be imported;
the coordinator then stays below mode b2.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Module state, so that apply / revert are exact.
_PATCHED_SCHEDULER: bool = False
_PATCHED_KV_CACHE_MANAGER: bool = False
_PATCHED_BLOCK_POOL: bool = False
_PATCHED_ASYNC_LLM_B2: bool = False

_ORIG_PREEMPT_REQUEST = None
_APPLY_MARKER_PRINTED: bool = False

# Blocks subtracted from the request's block count before choosing how many to
# keep, so that a partially filled boundary block is never kept.  Overridden
# per request by ``_canopy_partial_preempt_safety_margin``.
DEFAULT_SAFETY_MARGIN = 1


# ---------------------------------------------------------------------
# Counters, attached to the Scheduler class so that the EngineCore utility
# ``canopy_get_partial_preempt_stats`` can read them from
# ``type(scheduler)._canopy_partial_preempt_counters``.
# ---------------------------------------------------------------------


def _ensure_scheduler_counters(scheduler_cls) -> None:
    """Attach the counter dict to the Scheduler class if missing."""
    if not hasattr(scheduler_cls, "_canopy_partial_preempt_counters"):
        scheduler_cls._canopy_partial_preempt_counters = {
            "tags_set": 0,
            "tags_unset": 0,
            "preempt_invocations": 0,           # every _preempt_request call
            "preempt_partial_path_taken": 0,    # passed all checks
            "succeeded": 0,                     # partially freed
            "fallback_to_full": 0,              # failure after the checks
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


# ---------------------------------------------------------------------
# Scheduler._preempt_request wrap
# ---------------------------------------------------------------------


def _canopy_preempt_request_partial(self, request, timestamp):
    """``Scheduler._preempt_request`` that keeps a tagged request's leading blocks.

      1. Count the call (``preempt_invocations``).
      2. No tag (``_canopy_partial_preempt_keep_n`` missing or <= 0): original.
      3. Tag expired (``_canopy_partial_preempt_deadline`` passed): original.
      4. The request's own block count (minimum over KV cache groups) bounds
         what can be kept.  vLLM's runtime common-prefix count is not used: for
         the last running request of a tree it is 0, although its prefix is
         still worth keeping.
      5. ``effective_keep = min(keep_n, n_blocks - 1 - safety_margin)``, so at
         least one block is released; if it is <= 0: original.
      6. ``KVCacheManager.partial_free(request, effective_keep)``.
      7. As the original: free the encoder cache, mark the request PREEMPTED,
         count the preemption, record the PREEMPTED event when stats are on,
         and put the request back at the head of the waiting queue; but set
         ``num_computed_tokens = effective_keep * block_size`` instead of 0.

    Any exception falls back to the original.
    """
    try:
        # Counters live on the scheduler class (vLLM's Scheduler, or a stub in
        # tests), so the wrap does not need to import vLLM.
        _sched_cls = type(self)
        _ensure_scheduler_counters(_sched_cls)
        counters = _sched_cls._canopy_partial_preempt_counters
        counters["preempt_invocations"] += 1

        keep_n = getattr(request, "_canopy_partial_preempt_keep_n", 0)
        try:
            keep_n = int(keep_n)
        except Exception:
            keep_n = 0
        if keep_n <= 0:
            counters["skip_due_to_no_tag"] += 1
            return _ORIG_PREEMPT_REQUEST(self, request, timestamp)

        keep_dl = getattr(request, "_canopy_partial_preempt_deadline", 0.0)
        try:
            keep_dl = float(keep_dl)
        except Exception:
            keep_dl = 0.0
        if keep_dl < time.time():
            counters["skip_due_to_ttl_expired"] += 1
            counters["ttl_expired_at_preempt"] += 1
            return _ORIG_PREEMPT_REQUEST(self, request, timestamp)

        try:
            block_groups = self.kv_cache_manager.get_block_ids(request.request_id)
        except Exception as exc:
            counters["partial_free_exceptions"] += 1
            logger.warning(
                "[CANOPY-VLLM-B2] get_block_ids exc: %s: %s",
                type(exc).__name__, exc,
            )
            return _ORIG_PREEMPT_REQUEST(self, request, timestamp)

        if not block_groups:
            counters["skip_due_to_no_common_prefix"] += 1
            return _ORIG_PREEMPT_REQUEST(self, request, timestamp)

        # Keep only blocks present in every KV cache group (conservative for
        # hybrid attention).
        try:
            max_request_blocks = min(len(g) for g in block_groups if g)
        except Exception:
            max_request_blocks = 0
        if max_request_blocks <= 0:
            counters["skip_due_to_no_common_prefix"] += 1
            return _ORIG_PREEMPT_REQUEST(self, request, timestamp)

        safety = getattr(
            request,
            "_canopy_partial_preempt_safety_margin",
            DEFAULT_SAFETY_MARGIN,
        )
        try:
            safety = max(0, int(safety))
        except Exception:
            safety = DEFAULT_SAFETY_MARGIN
        upper = max_request_blocks - 1 - safety
        effective_keep = max(0, min(keep_n, upper))
        if effective_keep <= 0:
            counters["skip_due_to_zero_keep_after_safety"] += 1
            return _ORIG_PREEMPT_REQUEST(self, request, timestamp)

        counters["preempt_partial_path_taken"] += 1

        try:
            result = self.kv_cache_manager.partial_free(
                request, keep_n_blocks=effective_keep
            )
        except Exception as exc:
            counters["partial_free_exceptions"] += 1
            counters["fallback_to_full"] += 1
            logger.warning(
                "[CANOPY-VLLM-B2] partial_free exc, fallback: %s: %s",
                type(exc).__name__, exc,
            )
            return _ORIG_PREEMPT_REQUEST(self, request, timestamp)

        # The encoder cache is not shared across siblings: always free it.
        try:
            self.encoder_cache_manager.free(request)
        except Exception:
            pass

        try:
            from vllm.v1.request import RequestStatus
            request.status = RequestStatus.PREEMPTED
        except Exception:
            pass

        block_size = getattr(self.kv_cache_manager, "block_size", 16) or 16
        request.num_computed_tokens = int(effective_keep) * int(block_size)
        request.num_preemptions = int(getattr(request, "num_preemptions", 0)) + 1

        counters["succeeded"] += 1
        counters["kept_blocks_total"] += int(effective_keep)
        try:
            counters["released_blocks_total"] += int(result.get("released", 0))
        except Exception:
            pass

        if getattr(self, "log_stats", False):
            try:
                from vllm.v1.engine import EngineCoreEventType
                request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
            except Exception:
                pass

        try:
            self.waiting.prepend_request(request)
        except Exception as exc:
            # The original also re-queues the request; let it do so.
            logger.warning(
                "[CANOPY-VLLM-B2] waiting.prepend_request exc, fallback: %s: %s",
                type(exc).__name__, exc,
            )
            counters["fallback_to_full"] += 1
            return _ORIG_PREEMPT_REQUEST(self, request, timestamp)

        return None
    except Exception as exc:
        logger.warning(
            "[CANOPY-VLLM-B2] _preempt_request wrap outer exc: %s: %s",
            type(exc).__name__, exc,
        )
        try:
            _sched_cls = type(self)
            _ensure_scheduler_counters(_sched_cls)
            _sched_cls._canopy_partial_preempt_counters["fallback_to_full"] += 1
        except Exception:
            pass
        if _ORIG_PREEMPT_REQUEST is not None:
            return _ORIG_PREEMPT_REQUEST(self, request, timestamp)
        raise


# ---------------------------------------------------------------------
# KVCacheManager.partial_free (added method)
# ---------------------------------------------------------------------


def _canopy_kv_cache_manager_partial_free(self, request, keep_n_blocks: int) -> dict:
    """Release a request's blocks after the first ``keep_n_blocks`` of each KV cache group.

    The kept blocks keep their ``ref_cnt`` and stay in the prefix cache, so
    the resumed request finds them again.  The released suffix goes through
    :func:`_canopy_block_pool_free_blocks_partial`.

    Returns ``{"released": int, "kept": int, "groups": int}``.  Exceptions are
    re-raised so the scheduler wrap falls back to a full preemption.
    """
    try:
        rid = request.request_id
        block_groups = self.get_block_ids(rid)
        released_total = 0
        kept_total = 0
        groups_count = 0
        bp = self.block_pool
        blocks_arr = getattr(bp, "blocks", None)
        for group in block_groups:
            groups_count += 1
            if not group:
                continue
            n = len(group)
            keep = min(int(keep_n_blocks), n)
            if keep >= n:
                kept_total += n
                continue
            kept_total += keep
            release_ids = group[keep:]
            # Block ids index BlockPool.blocks.
            release_blocks = []
            if blocks_arr is not None:
                for bid in release_ids:
                    try:
                        idx = int(bid)
                        if 0 <= idx < len(blocks_arr):
                            release_blocks.append(blocks_arr[idx])
                    except Exception:
                        continue
            else:
                continue
            try:
                bp.free_blocks_partial(release_blocks)
                released_total += len(release_blocks)
            except Exception as exc:
                # Skip this group; the others are still released.
                logger.warning(
                    "[CANOPY-VLLM-B2] free_blocks_partial group err: %s: %s",
                    type(exc).__name__, exc,
                )

        # Count the bypass of KVCacheCoordinator.free on the Scheduler class
        # (in tests: the first loaded class named *Scheduler with counters).
        try:
            try:
                from vllm.v1.core.sched.scheduler import Scheduler as _Sched
            except Exception:
                _Sched = None
                import gc
                for obj in gc.get_objects():
                    if (
                        isinstance(obj, type)
                        and obj.__name__.endswith("Scheduler")
                        and hasattr(obj, "_canopy_partial_preempt_counters")
                    ):
                        _Sched = obj
                        break
            if _Sched is not None:
                _ensure_scheduler_counters(_Sched)
                _Sched._canopy_partial_preempt_counters[
                    "coordinator_unsync_warnings"
                ] += 1
        except Exception:
            pass

        return {
            "released": released_total,
            "kept": kept_total,
            "groups": groups_count,
        }
    except Exception as exc:
        logger.warning(
            "[CANOPY-VLLM-B2] partial_free outer exc: %s: %s",
            type(exc).__name__, exc,
        )
        raise


# ---------------------------------------------------------------------
# BlockPool.free_blocks_partial (added method)
# ---------------------------------------------------------------------


def _canopy_block_pool_free_blocks_partial(self, blocks_to_release: Iterable) -> None:
    """``BlockPool.free_blocks`` for a subset of a request's blocks.

    Same logic as vLLM 0.13's ``free_blocks``: decrement ``ref_cnt`` and append
    blocks that reach 0 (and are not the null block) to the free queue.  Blocks
    the caller keeps are simply not passed in.
    """
    blocks_list = list(blocks_to_release)
    if not blocks_list:
        return
    for block in blocks_list:
        try:
            block.ref_cnt -= 1
        except Exception:
            continue
    try:
        self.free_block_queue.append_n(
            [
                block
                for block in blocks_list
                if getattr(block, "ref_cnt", 1) == 0
                and not getattr(block, "is_null", False)
            ]
        )
    except Exception as exc:
        logger.warning(
            "[CANOPY-VLLM-B2] free_block_queue.append_n exc: %s: %s",
            type(exc).__name__, exc,
        )


def _canopy_get_partial_preempt_state(scheduler_cls) -> dict[str, int]:
    """Copy of the counter dict."""
    _ensure_scheduler_counters(scheduler_cls)
    return dict(scheduler_cls._canopy_partial_preempt_counters)


# ---------------------------------------------------------------------
# AsyncLLM RPC methods: direct call for an in-process engine core,
# ``call_utility_async`` for the engine-core subprocess.
# ---------------------------------------------------------------------


async def _canopy_async_llm_set_partial_preempt_for_request(
    self,
    request_id: str,
    keep_n_blocks: int,
    ttl_sec: float = 60.0,
) -> dict[str, Any]:
    """Tag a request for partial preemption (mirror of ``set_prefix_lock_for_request``)."""
    direct_fn = getattr(self.engine_core, "canopy_set_partial_preempt_for_request", None)
    if callable(direct_fn):
        try:
            return direct_fn(request_id, int(keep_n_blocks), float(ttl_sec))
        except Exception as exc:
            return {
                "tagged": 0,
                "error": f"direct: {type(exc).__name__}: {exc}",
                "request_id": request_id,
                "path": "direct",
            }
    call_fn = getattr(self.engine_core, "call_utility_async", None)
    if call_fn is None:
        return {
            "tagged": 0,
            "error": "engine_core_proxy_no_call_utility_async",
            "engine_core_type": type(self.engine_core).__name__,
            "request_id": request_id,
        }
    try:
        result = await call_fn(
            "canopy_set_partial_preempt_for_request",
            request_id,
            int(keep_n_blocks),
            float(ttl_sec),
        )
    except Exception as exc:
        return {
            "tagged": 0,
            "error": f"call_utility_async: {type(exc).__name__}: {exc}",
            "engine_core_type": type(self.engine_core).__name__,
            "request_id": request_id,
            "path": "rpc",
        }
    if isinstance(result, dict):
        result.setdefault("path", "rpc")
        return result
    return {
        "tagged": 0,
        "error": f"rpc_unexpected_result: {type(result).__name__}",
        "request_id": request_id,
        "path": "rpc",
    }


async def _canopy_async_llm_unset_partial_preempt_for_request(
    self, request_id: str
) -> dict[str, Any]:
    """Remove a partial-preemption tag."""
    direct_fn = getattr(self.engine_core, "canopy_unset_partial_preempt_for_request", None)
    if callable(direct_fn):
        try:
            return direct_fn(request_id)
        except Exception as exc:
            return {
                "untagged": 0,
                "error": f"direct: {type(exc).__name__}: {exc}",
                "request_id": request_id,
                "path": "direct",
            }
    call_fn = getattr(self.engine_core, "call_utility_async", None)
    if call_fn is None:
        return {
            "untagged": 0,
            "error": "engine_core_proxy_no_call_utility_async",
            "request_id": request_id,
        }
    try:
        result = await call_fn(
            "canopy_unset_partial_preempt_for_request", request_id
        )
    except Exception as exc:
        return {
            "untagged": 0,
            "error": f"call_utility_async: {type(exc).__name__}: {exc}",
            "request_id": request_id,
            "path": "rpc",
        }
    if isinstance(result, dict):
        result.setdefault("path", "rpc")
        return result
    return {
        "untagged": 0,
        "error": f"rpc_unexpected_result: {type(result).__name__}",
        "request_id": request_id,
        "path": "rpc",
    }


async def _canopy_async_llm_get_partial_preempt_stats(self) -> dict[str, Any]:
    """Counter dict of the engine core."""
    direct_fn = getattr(self.engine_core, "canopy_get_partial_preempt_stats", None)
    if callable(direct_fn):
        try:
            return direct_fn()
        except Exception as exc:
            return {"error": f"direct: {type(exc).__name__}: {exc}"}
    call_fn = getattr(self.engine_core, "call_utility_async", None)
    if call_fn is None:
        return {"error": "engine_core_proxy_no_call_utility_async"}
    try:
        result = await call_fn("canopy_get_partial_preempt_stats")
    except Exception as exc:
        return {"error": f"call_utility_async: {type(exc).__name__}: {exc}"}
    if isinstance(result, dict):
        result.setdefault("path", "rpc")
        return result
    return {"error": f"rpc_unexpected_result: {type(result).__name__}"}


# ---------------------------------------------------------------------
# Apply / revert
# ---------------------------------------------------------------------


def apply_partial_preempt_patch() -> bool:
    """Install the Scheduler, KVCacheManager, BlockPool and AsyncLLM patches.  Idempotent.

    Returns ``True`` on success (or if already applied), ``False`` when vLLM's
    scheduler cannot be imported.
    """
    global _PATCHED_SCHEDULER, _PATCHED_KV_CACHE_MANAGER, _PATCHED_BLOCK_POOL
    global _PATCHED_ASYNC_LLM_B2, _ORIG_PREEMPT_REQUEST, _APPLY_MARKER_PRINTED

    if (
        _PATCHED_SCHEDULER
        and _PATCHED_KV_CACHE_MANAGER
        and _PATCHED_BLOCK_POOL
        and _PATCHED_ASYNC_LLM_B2
    ):
        return True

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as exc:
        logger.warning(
            "[CANOPY-VLLM-B2] Scheduler import failed: %s: %s",
            type(exc).__name__, exc,
        )
        return False

    if not _PATCHED_SCHEDULER:
        if _ORIG_PREEMPT_REQUEST is None:
            _ORIG_PREEMPT_REQUEST = Scheduler._preempt_request
        Scheduler._preempt_request = _canopy_preempt_request_partial
        _ensure_scheduler_counters(Scheduler)
        _PATCHED_SCHEDULER = True

    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception as exc:
        logger.warning(
            "[CANOPY-VLLM-B2] KVCacheManager import failed: %s: %s",
            type(exc).__name__, exc,
        )
        return _PATCHED_SCHEDULER

    if not _PATCHED_KV_CACHE_MANAGER:
        KVCacheManager.partial_free = _canopy_kv_cache_manager_partial_free
        _PATCHED_KV_CACHE_MANAGER = True

    try:
        from vllm.v1.core.block_pool import BlockPool
    except Exception as exc:
        logger.warning(
            "[CANOPY-VLLM-B2] BlockPool import failed: %s: %s",
            type(exc).__name__, exc,
        )
        return _PATCHED_SCHEDULER and _PATCHED_KV_CACHE_MANAGER

    if not _PATCHED_BLOCK_POOL:
        BlockPool.free_blocks_partial = _canopy_block_pool_free_blocks_partial
        _PATCHED_BLOCK_POOL = True

    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception as exc:
        logger.warning(
            "[CANOPY-VLLM-B2] AsyncLLM import failed (others patched): %s: %s",
            type(exc).__name__, exc,
        )
        return _PATCHED_SCHEDULER and _PATCHED_KV_CACHE_MANAGER and _PATCHED_BLOCK_POOL

    if not _PATCHED_ASYNC_LLM_B2:
        AsyncLLM.set_partial_preempt_for_request = (
            _canopy_async_llm_set_partial_preempt_for_request
        )
        AsyncLLM.unset_partial_preempt_for_request = (
            _canopy_async_llm_unset_partial_preempt_for_request
        )
        AsyncLLM.get_partial_preempt_stats = (
            _canopy_async_llm_get_partial_preempt_stats
        )
        _PATCHED_ASYNC_LLM_B2 = True

    if not _APPLY_MARKER_PRINTED:
        _APPLY_MARKER_PRINTED = True
        print(
            "[CANOPY-VLLM-B2] applied Scheduler + KVCacheManager + BlockPool "
            "+ AsyncLLM partial-preemption patches",
            flush=True,
        )

    return (
        _PATCHED_SCHEDULER
        and _PATCHED_KV_CACHE_MANAGER
        and _PATCHED_BLOCK_POOL
        and _PATCHED_ASYNC_LLM_B2
    )


def applied_partial_preempt_patch() -> bool:
    return (
        _PATCHED_SCHEDULER
        and _PATCHED_KV_CACHE_MANAGER
        and _PATCHED_BLOCK_POOL
        and _PATCHED_ASYNC_LLM_B2
    )


def revert_partial_preempt_patch() -> None:
    """Undo all partial-preemption patches (used by tests)."""
    global _PATCHED_SCHEDULER, _PATCHED_KV_CACHE_MANAGER, _PATCHED_BLOCK_POOL
    global _PATCHED_ASYNC_LLM_B2, _ORIG_PREEMPT_REQUEST, _APPLY_MARKER_PRINTED

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        Scheduler = None  # type: ignore

    if Scheduler is not None and _PATCHED_SCHEDULER:
        if _ORIG_PREEMPT_REQUEST is not None:
            Scheduler._preempt_request = _ORIG_PREEMPT_REQUEST
        if hasattr(Scheduler, "_canopy_partial_preempt_counters"):
            try:
                delattr(Scheduler, "_canopy_partial_preempt_counters")
            except Exception:
                pass
        _ORIG_PREEMPT_REQUEST = None
    _PATCHED_SCHEDULER = False

    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception:
        KVCacheManager = None  # type: ignore

    if KVCacheManager is not None and _PATCHED_KV_CACHE_MANAGER:
        if hasattr(KVCacheManager, "partial_free"):
            try:
                delattr(KVCacheManager, "partial_free")
            except Exception:
                pass
    _PATCHED_KV_CACHE_MANAGER = False

    try:
        from vllm.v1.core.block_pool import BlockPool
    except Exception:
        BlockPool = None  # type: ignore

    if BlockPool is not None and _PATCHED_BLOCK_POOL:
        if hasattr(BlockPool, "free_blocks_partial"):
            try:
                delattr(BlockPool, "free_blocks_partial")
            except Exception:
                pass
    _PATCHED_BLOCK_POOL = False

    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception:
        AsyncLLM = None  # type: ignore

    if AsyncLLM is not None and _PATCHED_ASYNC_LLM_B2:
        for name in (
            "set_partial_preempt_for_request",
            "unset_partial_preempt_for_request",
            "get_partial_preempt_stats",
        ):
            if hasattr(AsyncLLM, name):
                try:
                    delattr(AsyncLLM, name)
                except Exception:
                    pass
    _PATCHED_ASYNC_LLM_B2 = False

    _APPLY_MARKER_PRINTED = False


__all__ = [
    "apply_partial_preempt_patch",
    "applied_partial_preempt_patch",
    "revert_partial_preempt_patch",
    "_canopy_preempt_request_partial",
    "_canopy_kv_cache_manager_partial_free",
    "_canopy_block_pool_free_blocks_partial",
    "DEFAULT_SAFETY_MARGIN",
]
