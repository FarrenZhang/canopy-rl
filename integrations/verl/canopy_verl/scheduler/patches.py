# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch that carries ``tree_uid`` through verl's ``AsyncLLMServerManager``.

:func:`apply_patches` (run when :mod:`canopy_verl` is imported) replaces
``AsyncLLMServerManager.generate``:

  * ``tree_uid`` comes from a ``tree_uid=`` keyword argument or, if absent,
    from a per-task ``ContextVar`` set with :func:`set_tree_uid_for_routing`.
    Agent loops that delegate generation to ``ToolAgentLoop`` (and cannot add
    a keyword argument) use the ContextVar.
  * Without a ``tree_uid`` the original method runs unchanged.
  * With a ``tree_uid`` the server is acquired with
    ``acquire_server(request_id, tree_uid)``, the vLLM request id is
    registered as one of the tree's in-flight requests (the admission
    coordinator addresses prefix locks and preemption tags by it), and the
    release reports the request id and ``tree_uid`` back to the balancer.

With verl's own ``GlobalRequestLoadBalancer`` (no ``tree_uid`` parameter) the
calls fall back to the original signatures.

The patch is applied at method level because ``AsyncLLMServerManager`` is
created inside verl's ``AgentLoopWorker`` Ray actor.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import os
from uuid import uuid4


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_PATCHED_FLAG = "_canopy_placement_patched"

# asyncio gives every task its own copy of the context, so the value never leaks
# between agent-loop coroutines.
_canopy_current_tree_uid: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "canopy_current_tree_uid", default=None
)


def set_tree_uid_for_routing(tree_uid: str | None):
    """Set the ``tree_uid`` used by generate calls of the current asyncio task.

    Returns a token for :func:`reset_tree_uid_for_routing`::

        token = set_tree_uid_for_routing(tree_state.tree_uid)
        try:
            await super()._handle_generating_state(...)
        finally:
            reset_tree_uid_for_routing(token)
    """
    return _canopy_current_tree_uid.set(tree_uid)


def reset_tree_uid_for_routing(token) -> None:
    """Undo a :func:`set_tree_uid_for_routing` call."""
    try:
        _canopy_current_tree_uid.reset(token)
    except (LookupError, ValueError):
        # Token created in another context; never fail the agent loop over it.
        pass


def get_current_tree_uid() -> str | None:
    """Return the ``tree_uid`` of the current asyncio task, or ``None``."""
    return _canopy_current_tree_uid.get()


def apply_patches() -> None:
    """Patch ``AsyncLLMServerManager.generate``.  Idempotent; no-op without verl."""
    try:
        from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
    except ImportError:
        return

    if getattr(AsyncLLMServerManager, _PATCHED_FLAG, False):
        return

    _patch_generate(AsyncLLMServerManager)

    setattr(AsyncLLMServerManager, _PATCHED_FLAG, True)
    logger.warning(
        "[canopy-placement] AsyncLLMServerManager patched for tree_uid-aware routing"
    )
    print(
        "[canopy-placement] AsyncLLMServerManager patched for tree_uid-aware routing",
        flush=True,
    )


def _patch_generate(cls) -> None:
    """Route generate calls that carry a ``tree_uid`` through the tree-aware balancer.

    The original method calls ``self._acquire_server(request_id)``, whose
    signature has no ``tree_uid``.  The tree-aware path therefore acquires the
    server itself and then issues the same ``server.generate`` call as the
    original method.
    """
    original_generate = cls.generate

    @functools.wraps(original_generate)
    async def canopy_generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
        **kwargs,
    ):
        # An explicit keyword argument wins over the ContextVar; neither is
        # forwarded to the vLLM server.
        tree_uid = kwargs.pop("tree_uid", None)
        if tree_uid is None:
            tree_uid = _canopy_current_tree_uid.get()

        if tree_uid is None:
            return await original_generate(
                self,
                request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )

        server_id, server = await _canopy_acquire_server(
            self, request_id=request_id, tree_uid=tree_uid
        )
        vllm_request_id = uuid4().hex
        # Register the request as in flight before it starts, so the admission
        # coordinator can address it while its blocks are allocated.
        # Fire-and-forget; verl's own balancer has no such method.
        try:
            self._load_balancer.register_active_request.remote(
                server_id=server_id,
                request_id=vllm_request_id,
                tree_uid=tree_uid,
            )
        except (AttributeError, TypeError):
            pass
        try:
            return await server.generate.remote(
                request_id=vllm_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )
        finally:
            # Fire-and-forget release with the request id vLLM saw, so the
            # balancer can retire it from the tree's in-flight list.
            try:
                self._load_balancer.release_server.remote(
                    server_id=server_id,
                    request_id=vllm_request_id,
                    tree_uid=tree_uid,
                )
            except TypeError:
                # verl's balancer: release_server(server_id) only.
                self._release_server(server_id)

    cls.generate = canopy_generate


async def _canopy_acquire_server(
    server_manager,
    *,
    request_id: str,
    tree_uid: str,
):
    """``acquire_server(request_id, tree_uid)``, falling back to verl's one-argument form."""
    lb = server_manager._load_balancer
    try:
        server_id = await lb.acquire_server.remote(
            request_id=request_id, tree_uid=tree_uid
        )
    except TypeError:
        server_id = await lb.acquire_server.remote(request_id=request_id)

    handle = server_manager._server_id_to_handle.get(server_id)
    if handle is None:
        raise RuntimeError(f"Unknown server_id returned by load balancer: {server_id}")
    return server_id, handle
