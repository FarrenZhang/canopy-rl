# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Runtime patches of vLLM 0.13 (V1 engine) and verl's ``vLLMHttpServer``.

Applied in-process; vLLM is neither forked nor rebuilt.

  * :mod:`.prefix_lock` (shared-prefix retention, admission mode b1)
      - ``vllm.v1.core.block_pool.BlockPool``: ``set_prefix_lock``,
        ``unset_prefix_lock``, ``get_prefix_lock_state``, ``_canopy_expire_ttl``;
        ``get_new_blocks`` is wrapped to skip locked blocks.
      - ``vllm.v1.engine.async_llm.AsyncLLM``: ``set_prefix_lock_for_request``,
        ``unset_prefix_lock_for_request``, ``get_prefix_lock_stats``.
  * :mod:`.verl_server_extensions` -- the same three methods on verl's
    ``vLLMHttpServer``, forwarding to ``AsyncLLM``.
  * :mod:`.partial_preempt` (prefix-preserving partial preemption, mode b2)
      - ``vllm.v1.core.sched.scheduler.Scheduler._preempt_request`` is wrapped;
        ``KVCacheManager.partial_free`` and ``BlockPool.free_blocks_partial``
        are added.
      - ``AsyncLLM``: ``set_partial_preempt_for_request``,
        ``unset_partial_preempt_for_request``, ``get_partial_preempt_stats``.
  * :mod:`.verl_server_extensions_b2` -- the same three methods on
    ``vLLMHttpServer``.

With vLLM's default multiprocess engine, the block pool and scheduler live in
the engine-core subprocess; ``canopy_verl/bootstrap/sitecustomize.py`` installs
the parts that must run there.

Every ``apply_*`` is idempotent and returns ``True`` on success (also when
already applied).  If the upstream class cannot be imported it returns
``False`` and logs a warning; callers then fall back to a lower mode.
"""

from __future__ import annotations

from .prefix_lock import (
    apply_prefix_lock_patch,
    revert_prefix_lock_patch,
    applied_prefix_lock_patch,
)
from .verl_server_extensions import (
    apply_verl_server_extensions,
    applied_verl_server_extensions,
)
from .partial_preempt import (
    apply_partial_preempt_patch,
    applied_partial_preempt_patch,
    revert_partial_preempt_patch,
)
from .verl_server_extensions_b2 import (
    apply_verl_server_extensions_b2,
    applied_verl_server_extensions_b2,
)


def apply_all_b1_patches() -> dict[str, bool]:
    """Apply the patches needed by admission mode b1 (shared-prefix retention).

    Returns ``{patch name: success}``; safe to call repeatedly.
    """
    return {
        "prefix_lock": apply_prefix_lock_patch(),
        "verl_server_extensions": apply_verl_server_extensions(),
    }


def apply_all_b2_patches() -> dict[str, bool]:
    """Apply the patches needed by admission mode b2 (b1 + partial preemption).

    The prefix lock protects shared prefixes on the eviction path and partial
    preemption protects them on the preemption path; the two act on separate
    vLLM decision points, and b2 always includes b1.

    Returns ``{patch name: success}``; safe to call repeatedly.
    """
    return {
        "prefix_lock": apply_prefix_lock_patch(),
        "verl_server_extensions": apply_verl_server_extensions(),
        "partial_preempt": apply_partial_preempt_patch(),
        "verl_server_extensions_b2": apply_verl_server_extensions_b2(),
    }


__all__ = [
    "apply_prefix_lock_patch",
    "apply_verl_server_extensions",
    "apply_all_b1_patches",
    "applied_prefix_lock_patch",
    "applied_verl_server_extensions",
    "revert_prefix_lock_patch",
    "apply_partial_preempt_patch",
    "applied_partial_preempt_patch",
    "revert_partial_preempt_patch",
    "apply_verl_server_extensions_b2",
    "applied_verl_server_extensions_b2",
    "apply_all_b2_patches",
]
