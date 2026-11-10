# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Partial-preemption RPC methods on verl's ``vLLMHttpServer`` (admission mode b2).

Counterpart of :mod:`canopy_verl.vllm_patch.verl_server_extensions`: three
methods attached to the class (before the Ray actors are created), each
forwarding to the ``AsyncLLM`` method added by
:mod:`canopy_verl.vllm_patch.partial_preempt`.
"""

from __future__ import annotations

import logging
import os
from typing import Any


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_APPLIED: bool = False
_APPLY_MARKER_PRINTED: bool = False


async def _canopy_set_partial_preempt_for_request(
    self,
    request_id: str,
    keep_n_blocks: int,
    ttl_sec: float = 60.0,
) -> dict[str, Any]:
    """Forward to ``AsyncLLM.set_partial_preempt_for_request``."""
    try:
        engine = getattr(self, "engine", None)
        if engine is None:
            return {"tagged": 0, "error": "no_engine"}
        fn = getattr(engine, "set_partial_preempt_for_request", None)
        if fn is None:
            return {"tagged": 0, "error": "patch_not_applied"}
        return await fn(request_id, int(keep_n_blocks), ttl_sec=float(ttl_sec))
    except Exception as exc:
        return {"tagged": 0, "error": f"{type(exc).__name__}: {exc}"}


async def _canopy_unset_partial_preempt_for_request(
    self, request_id: str
) -> dict[str, Any]:
    try:
        engine = getattr(self, "engine", None)
        if engine is None:
            return {"untagged": 0, "error": "no_engine"}
        fn = getattr(engine, "unset_partial_preempt_for_request", None)
        if fn is None:
            return {"untagged": 0, "error": "patch_not_applied"}
        return await fn(request_id)
    except Exception as exc:
        return {"untagged": 0, "error": f"{type(exc).__name__}: {exc}"}


async def _canopy_get_partial_preempt_stats(self) -> dict[str, Any]:
    try:
        engine = getattr(self, "engine", None)
        if engine is None:
            return {"error": "no_engine"}
        fn = getattr(engine, "get_partial_preempt_stats", None)
        if fn is None:
            return {"error": "patch_not_applied"}
        return await fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def apply_verl_server_extensions_b2() -> bool:
    """Attach the three partial-preemption methods to ``vLLMHttpServer``.  Idempotent."""
    global _APPLIED, _APPLY_MARKER_PRINTED
    if _APPLIED:
        return True

    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import (
            vLLMHttpServer,
        )
    except Exception as exc:
        logger.warning(
            "[CANOPY-VERL-SERVER-B2] vLLMHttpServer import failed: %s: %s",
            type(exc).__name__, exc,
        )
        return False

    vLLMHttpServer.set_partial_preempt_for_request = (
        _canopy_set_partial_preempt_for_request
    )
    vLLMHttpServer.unset_partial_preempt_for_request = (
        _canopy_unset_partial_preempt_for_request
    )
    vLLMHttpServer.get_partial_preempt_stats = _canopy_get_partial_preempt_stats

    _APPLIED = True
    if not _APPLY_MARKER_PRINTED:
        _APPLY_MARKER_PRINTED = True
        print(
            "[CANOPY-VERL-SERVER-B2] applied 3 async partial-preempt "
            "methods on vLLMHttpServer",
            flush=True,
        )
    return True


def applied_verl_server_extensions_b2() -> bool:
    return _APPLIED


__all__ = [
    "apply_verl_server_extensions_b2",
    "applied_verl_server_extensions_b2",
]
