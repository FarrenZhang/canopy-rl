# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Tree-aware placement: a per-``tree_uid`` sticky load balancer for verl.

Drop-in replacement for verl's
:class:`verl.experimental.agent_loop.agent_loop.GlobalRequestLoadBalancer`.
When ``tree_uid`` is ``None`` (or ``enable_tree_sticky=False``) the routing is
exactly verl's: per-``request_id`` sticky with least-inflight selection.

Agent loops supply ``tree_uid`` through the patched
``AsyncLLMServerManager.generate`` (see :mod:`canopy_verl.scheduler.patches`).

Soft-sticky semantics:

  * First request of a ``tree_uid``: pick the least-inflight server and
    register ``tree_uid -> server``.
  * Later requests of the tree go to the registered server unless it is
    overloaded (:meth:`TreeAwareLoadBalancer._is_preferred_overloaded`); then
    the request spills to the least-inflight server.  The registration stays,
    so later siblings try the registered server again.
  * :meth:`TreeAwareLoadBalancer.on_tree_complete` drops the registration and
    asks the admission coordinator (if attached) to remove the tree's prefix
    locks.  Nothing in this package calls it; without a caller, registrations
    persist and prefix locks expire after their TTL.

The balancer also keeps the in-flight vLLM request ids of each tree; the
admission coordinator reads them with :meth:`get_active_trees_snapshot` to
address prefix-retention and partial-preemption requests.

Metrics (:meth:`get_metrics`):

  * ``placement/siblings_co_located_ratio`` -- ``hits / (hits + spills +
    registrations)``.  The first request of each tree counts as a miss, so
    with ``k`` requests per tree the ratio is at most ``1 - 1/k``.
  * ``placement/siblings_co_located_ratio_strict`` -- ``hits / (hits +
    spills)``: of the requests that could join their tree's server, the
    fraction that did (1.0 when nothing spilled).
  * ``placement/spill_events`` -- soft-sticky spills (server overload).
  * ``placement/tree_registrations`` -- new trees.
  * ``placement/active_trees`` / ``placement/active_trees_per_server`` --
    registered trees, in total and per server.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import ray
from cachetools import LRUCache

from canopy_verl.scheduler.admission import TreeValueSnapshot


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Same request-id cache size as verl's GlobalRequestLoadBalancer.
DEFAULT_ROUTING_CACHE_SIZE = 10000
DEFAULT_STICKY_THRESHOLD = 0.75


@dataclass
class TreeEntry:
    """Per-tree state held inside the balancer actor."""

    tree_uid: str
    primary_server: str
    active_count: int = 0
    completed_count: int = 0
    spill_count: int = 0
    # The last k in-flight vLLM request ids of the tree.  Only requests that are
    # still allocated can be resolved to KV blocks by vLLM, so these are the
    # targets of prefix-lock and partial-preemption requests.  k is
    # ``completed_request_ids_cap``.
    active_request_ids: list[str] = field(default_factory=list)
    # The last k released request ids (their blocks are already freed; kept
    # for inspection only).
    completed_request_ids: list[str] = field(default_factory=list)

    def total_acquires(self) -> int:
        return self.active_count + self.completed_count + self.spill_count


@ray.remote
class TreeAwareLoadBalancer:
    """Per-``tree_uid`` sticky load balancer with least-inflight fallback.

    The API is a superset of verl's ``GlobalRequestLoadBalancer``:

      * ``acquire_server(request_id, tree_uid=None)`` -- same as verl's when
        ``tree_uid`` is ``None``.
      * ``release_server(server_id, request_id=None, tree_uid=None)`` -- same
        as verl's when only ``server_id`` is given.

    Additional methods:

      * ``register_active_request`` -- record an in-flight vLLM request id.
      * ``on_tree_complete(tree_uid)`` -- drop a tree's registration.
      * ``get_active_trees_snapshot()`` -- input of the admission coordinator.
      * ``set_admission_coordinator`` / ``get_admission_coordinator``.
      * ``get_metrics()`` / ``reset_metrics()`` -- ``placement/*`` counters.
    """

    def __init__(
        self,
        server_actor_ids: list[str],
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        sticky_threshold: float = DEFAULT_STICKY_THRESHOLD,
        enable_tree_sticky: bool = True,
        abs_spill_gap: int = 10,
        completed_request_ids_cap: int = 4,
    ):
        if not server_actor_ids:
            raise ValueError("server_actor_ids must be non-empty")
        if not 0.0 < sticky_threshold <= 1.0:
            raise ValueError(
                f"sticky_threshold must be in (0, 1], got {sticky_threshold}"
            )
        if abs_spill_gap < 0:
            raise ValueError(f"abs_spill_gap must be >= 0, got {abs_spill_gap}")
        if completed_request_ids_cap < 0:
            raise ValueError(
                f"completed_request_ids_cap must be >= 0, got {completed_request_ids_cap}"
            )

        self._server_ids: list[str] = list(server_actor_ids)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in server_actor_ids}
        # Per-request_id sticky cache, as in verl's GlobalRequestLoadBalancer.
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)

        self._trees: dict[str, TreeEntry] = {}
        self._sticky_threshold: float = sticky_threshold
        self._abs_spill_gap: int = abs_spill_gap
        self._enable_tree_sticky: bool = enable_tree_sticky
        self._completed_request_ids_cap: int = completed_request_ids_cap

        # Monotonic counters; ratios are derived in get_metrics.
        self._m_tree_sticky_hits: int = 0
        self._m_tree_sticky_spills: int = 0
        self._m_tree_registrations: int = 0
        self._m_request_sticky_hits: int = 0
        self._m_new_request_routes: int = 0
        self._m_releases: int = 0
        self._m_tree_completes: int = 0

        # AdmissionCoordinator handle, attached by CanopyAgentLoopManager.  It
        # lives here because agent-loop workers already hold the balancer handle,
        # so prom_hook can reach the coordinator through it.
        self._admission_coordinator: Any | None = None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def acquire_server(
        self, request_id: str, tree_uid: str | None = None
    ) -> str:
        """Pick a vLLM server for a request.

        Args:
            request_id: request identifier (as in verl).
            tree_uid: if given and tree-sticky routing is enabled, prefer the
                server registered for this tree (soft sticky).  Otherwise the
                choice is identical to verl's ``GlobalRequestLoadBalancer``.

        Returns:
            One of ``server_actor_ids``.
        """
        if self._enable_tree_sticky and tree_uid is not None:
            server_id = self._acquire_tree_aware(request_id, tree_uid)
        else:
            server_id = self._acquire_request_sticky(request_id)

        self._inflight_requests[server_id] += 1
        return server_id

    def register_active_request(
        self,
        server_id: str,
        request_id: str,
        tree_uid: str,
    ) -> None:
        """Record that the in-flight vLLM request ``request_id`` belongs to ``tree_uid``.

        Called by :mod:`canopy_verl.scheduler.patches` right before the request
        is sent to vLLM.  No-op for unregistered trees; the per-tree list keeps
        the last ``completed_request_ids_cap`` ids.
        """
        if self._completed_request_ids_cap <= 0:
            return
        entry = self._trees.get(tree_uid)
        if entry is None:
            return
        entry.active_request_ids.append(str(request_id))
        if len(entry.active_request_ids) > self._completed_request_ids_cap:
            extra = (
                len(entry.active_request_ids)
                - self._completed_request_ids_cap
            )
            del entry.active_request_ids[0:extra]

    def release_server(
        self,
        server_id: str,
        request_id: str | None = None,
        tree_uid: str | None = None,
    ) -> None:
        """Release a server after a request completes.

        With only ``server_id`` this is verl's ``release_server``.  With
        ``request_id`` and ``tree_uid`` the request also moves from the tree's
        in-flight list to its completed list (vLLM has freed its blocks, so it
        is no longer a lock target), and the tree's active / completed counts
        are updated.
        """
        if server_id not in self._inflight_requests:
            raise ValueError(f"Invalid server_id for release: {server_id}")
        if self._inflight_requests[server_id] <= 0:
            raise ValueError(
                f"Release called with no inflight requests on server {server_id}"
            )
        self._inflight_requests[server_id] -= 1
        self._m_releases += 1

        if (
            request_id is not None
            and tree_uid is not None
            and self._completed_request_ids_cap > 0
        ):
            entry = self._trees.get(tree_uid)
            if entry is not None:
                rid_s = str(request_id)
                try:
                    entry.active_request_ids.remove(rid_s)
                except ValueError:
                    pass
                entry.completed_request_ids.append(rid_s)
                if len(entry.completed_request_ids) > self._completed_request_ids_cap:
                    extra = (
                        len(entry.completed_request_ids)
                        - self._completed_request_ids_cap
                    )
                    del entry.completed_request_ids[0:extra]
                # Never let active_count go negative.
                if entry.active_count > 0:
                    entry.active_count -= 1
                    entry.completed_count += 1

    def on_tree_complete(self, tree_uid: str) -> None:
        """Drop the registration of a finished tree.  Idempotent.

        If an admission coordinator is attached, it is asked (fire-and-forget)
        to remove the tree's prefix locks; the lock TTL covers failures.
        """
        entry = self._trees.pop(tree_uid, None)
        if entry is None:
            return
        entry.completed_count += entry.active_count
        entry.active_count = 0
        self._m_tree_completes += 1

        coord = getattr(self, "_admission_coordinator", None)
        if coord is not None:
            try:
                coord.dispatch_prefix_unlock_on_complete.remote(tree_uid)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Admission support
    # ------------------------------------------------------------------

    def set_admission_coordinator(self, handle: Any) -> None:
        """Attach an AdmissionCoordinator handle (``None`` detaches)."""
        self._admission_coordinator = handle

    def get_admission_coordinator(self) -> Any | None:
        """Return the attached AdmissionCoordinator handle, or ``None``."""
        return self._admission_coordinator

    def get_server_ids(self) -> list[str]:
        """Return the known vLLM server ids, in order."""
        return list(self._server_ids)

    def get_active_trees_snapshot(self) -> list["TreeValueSnapshot"]:
        """One :class:`TreeValueSnapshot` per registered tree.

        ``kv_blocks_approx`` uses the number of active requests as a proxy for
        the tree's KV footprint.
        """
        out: list["TreeValueSnapshot"] = []
        for entry in self._trees.values():
            out.append(
                TreeValueSnapshot(
                    tree_uid=entry.tree_uid,
                    complete_siblings=entry.completed_count,
                    active_siblings=entry.active_count,
                    kv_blocks_approx=max(1, entry.active_count),
                    private_kv_blocks_approx=0,
                    primary_server=entry.primary_server,
                    active_request_ids=list(entry.active_request_ids),
                    completed_request_ids=list(entry.completed_request_ids),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Counters and derived ratios (``placement/*``)."""
        tree_acquires_total = (
            self._m_tree_sticky_hits
            + self._m_tree_sticky_spills
            + self._m_tree_registrations
        )
        siblings_co_located_ratio = (
            self._m_tree_sticky_hits / tree_acquires_total
            if tree_acquires_total > 0
            else 0.0
        )
        sibling_opportunities = (
            self._m_tree_sticky_hits + self._m_tree_sticky_spills
        )
        siblings_co_located_ratio_strict = (
            self._m_tree_sticky_hits / sibling_opportunities
            if sibling_opportunities > 0
            else 0.0
        )

        per_server_active: dict[str, int] = {sid: 0 for sid in self._server_ids}
        for entry in self._trees.values():
            per_server_active[entry.primary_server] = (
                per_server_active.get(entry.primary_server, 0) + 1
            )

        return {
            "placement/tree_acquires_total": tree_acquires_total,
            "placement/siblings_co_located_ratio": siblings_co_located_ratio,
            "placement/siblings_co_located_ratio_strict": siblings_co_located_ratio_strict,
            "placement/tree_sticky_hits": self._m_tree_sticky_hits,
            "placement/spill_events": self._m_tree_sticky_spills,
            "placement/tree_registrations": self._m_tree_registrations,
            "placement/request_sticky_hits": self._m_request_sticky_hits,
            "placement/new_request_routes": self._m_new_request_routes,
            "placement/active_trees": len(self._trees),
            "placement/active_trees_per_server": per_server_active,
            "placement/releases": self._m_releases,
            "placement/tree_completes": self._m_tree_completes,
            "placement/inflight_per_server": dict(self._inflight_requests),
        }

    def reset_metrics(self) -> None:
        """Zero the counters; routing state is kept."""
        self._m_tree_sticky_hits = 0
        self._m_tree_sticky_spills = 0
        self._m_tree_registrations = 0
        self._m_request_sticky_hits = 0
        self._m_new_request_routes = 0
        self._m_releases = 0
        self._m_tree_completes = 0

    # ------------------------------------------------------------------
    # Internal routing paths
    # ------------------------------------------------------------------

    def _acquire_tree_aware(self, request_id: str, tree_uid: str) -> str:
        """Tree-level soft-sticky routing."""
        entry = self._trees.get(tree_uid)

        if entry is None:
            server_id = self._least_inflight_server()
            self._trees[tree_uid] = TreeEntry(
                tree_uid=tree_uid, primary_server=server_id, active_count=1
            )
            self._m_tree_registrations += 1
            self._request_id_to_server[request_id] = server_id
            return server_id

        preferred = entry.primary_server
        if self._is_preferred_overloaded(preferred):
            spill_server = self._least_inflight_server()
            entry.spill_count += 1
            self._m_tree_sticky_spills += 1
            logger.warning(
                "[Placement] soft-sticky spill tree_uid=%s primary=%s "
                "primary_load=%d spill_to=%s spill_load=%d",
                tree_uid[:8],
                preferred,
                self._inflight_requests[preferred],
                spill_server,
                self._inflight_requests[spill_server],
            )
            self._request_id_to_server[request_id] = spill_server
            return spill_server

        entry.active_count += 1
        self._m_tree_sticky_hits += 1
        self._request_id_to_server[request_id] = preferred
        return preferred

    def _acquire_request_sticky(self, request_id: str) -> str:
        """Per-request_id sticky routing, identical to verl's balancer."""
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            self._m_request_sticky_hits += 1
            return server_id

        server_id = self._least_inflight_server()
        self._request_id_to_server[request_id] = server_id
        self._m_new_request_routes += 1
        return server_id

    def _least_inflight_server(self) -> str:
        """Least-inflight server; ties go to the first in ``server_actor_ids``, like verl's ``min()``."""
        return min(self._inflight_requests, key=self._inflight_requests.get)

    def _is_preferred_overloaded(self, preferred_server: str) -> bool:
        """Spill only when both guards hold.

          1. Ratio guard: the preferred server carries more than
             ``sticky_threshold`` of the total in-flight requests.
          2. Absolute-gap guard: its load exceeds the least-loaded other server
             by more than ``abs_spill_gap``.

        The ratio alone would spill in a lightly loaded cluster (1 vs 0 requests
        is a ratio of 100%) and split even two-request trees; the gap guard keeps
        typical trees together while still spilling under real overload.
        """
        if len(self._inflight_requests) <= 1:
            return False

        total_load = sum(self._inflight_requests.values())
        if total_load == 0:
            return False
        preferred_load = self._inflight_requests[preferred_server]
        preferred_ratio = preferred_load / total_load

        other_loads = [
            load
            for sid, load in self._inflight_requests.items()
            if sid != preferred_server
        ]
        if not other_loads:
            return False
        min_other = min(other_loads)
        abs_gap = preferred_load - min_other

        return (
            preferred_ratio > self._sticky_threshold
            and abs_gap > self._abs_spill_gap
        )
