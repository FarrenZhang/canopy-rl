# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Tree-aware rollout scheduling for verl.

  * :mod:`placement` -- ``TreeAwareLoadBalancer``: sends all requests of a tree
    (one ``tree_uid``) to the same vLLM server (group-level placement), with a
    soft spill to the least-loaded server under overload.
  * :mod:`patches` -- carries ``tree_uid`` from agent loops to the balancer
    through ``AsyncLLMServerManager.generate``.
  * :mod:`canopy_agent_loop_manager` -- ``CanopyAgentLoopManager``: installs the
    balancer and, if enabled, the admission coordinator.
  * :mod:`admission` -- ``AdmissionCoordinator``: issues shared-prefix
    retention (mode ``b1``) and prefix-preserving preemption (mode ``b2``)
    requests to the vLLM servers.

Configuration (top-level hydra key, read by ``CanopyAgentLoopManager``)::

    canopy_scheduler:
      placement:
        enabled: true              # false: same routing as verl's GlobalRequestLoadBalancer
        sticky_threshold: 0.75     # soft-sticky spill threshold
        max_cache_size: 10000
      admission:
        enabled: true
        mode: b2                   # "a" | "b1" | "b2", see admission.py

The integration also needs a few changes to verl itself; see
``integrations/verl/patches/README.md``.
"""

__all__ = [
    "admission",
    "canopy_agent_loop_manager",
    "patches",
    "placement",
]
