# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Prefix-lock RPC methods on verl's ``vLLMHttpServer`` (admission mode b1).

The methods are attached to the class, so they must be applied before verl
creates the ``vLLMHttpServer`` Ray actors.  Each one forwards to the
``AsyncLLM`` method added by :mod:`canopy_verl.vllm_patch.prefix_lock` and
returns a dict with an ``error`` key instead of raising.
"""

from __future__ import annotations

import logging
import os
from typing import Any


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_APPLIED: bool = False
_APPLY_MARKER_PRINTED: bool = False


async def _canopy_set_prefix_lock_for_request(
    self, request_id: str, ttl_sec: float = 60.0
) -> dict[str, Any]:
    try:
        engine = getattr(self, "engine", None)
        if engine is None:
            return {"locked": 0, "error": "no_engine"}
        fn = getattr(engine, "set_prefix_lock_for_request", None)
        if fn is None:
            return {"locked": 0, "error": "patch_not_applied"}
        return await fn(request_id, ttl_sec=ttl_sec)
    except Exception as exc:
        return {"locked": 0, "error": f"{type(exc).__name__}: {exc}"}


async def _canopy_unset_prefix_lock_for_request(
    self, request_id: str
) -> dict[str, Any]:
    try:
        engine = getattr(self, "engine", None)
        if engine is None:
            return {"unlocked": 0, "error": "no_engine"}
        fn = getattr(engine, "unset_prefix_lock_for_request", None)
        if fn is None:
            return {"unlocked": 0, "error": "patch_not_applied"}
        return await fn(request_id)
    except Exception as exc:
        return {"unlocked": 0, "error": f"{type(exc).__name__}: {exc}"}


async def _canopy_get_prefix_lock_stats(self) -> dict[str, Any]:
    try:
        engine = getattr(self, "engine", None)
        if engine is None:
            return {"error": "no_engine"}
        fn = getattr(engine, "get_prefix_lock_stats", None)
        if fn is None:
            return {"error": "patch_not_applied"}
        return await fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def apply_verl_server_extensions() -> bool:
    """Attach the three prefix-lock methods to ``vLLMHttpServer``.  Idempotent."""
    global _APPLIED, _APPLY_MARKER_PRINTED
    if _APPLIED:
        return True

    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import (
            vLLMHttpServer,
        )
    except Exception as exc:
        logger.warning(
            "[CANOPY-VERL-SERVER-PATCH] vLLMHttpServer import failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        return False

    vLLMHttpServer.set_prefix_lock_for_request = _canopy_set_prefix_lock_for_request
    vLLMHttpServer.unset_prefix_lock_for_request = (
        _canopy_unset_prefix_lock_for_request
    )
    vLLMHttpServer.get_prefix_lock_stats = _canopy_get_prefix_lock_stats

    _APPLIED = True
    if not _APPLY_MARKER_PRINTED:
        _APPLY_MARKER_PRINTED = True
        print(
            "[CANOPY-VERL-SERVER-PATCH] applied 3 async methods on vLLMHttpServer",
            flush=True,
        )
    return True


def applied_verl_server_extensions() -> bool:
    return _APPLIED


__all__ = [
    "apply_verl_server_extensions",
    "applied_verl_server_extensions",
]
