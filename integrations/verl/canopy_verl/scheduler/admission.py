# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Admission coordinator: shared-prefix retention and prefix-preserving preemption.

The coordinator protects the KV cache that sibling trajectories of a tree
still need.  It runs as a Ray actor, is ticked by
:func:`canopy_verl.observability.prom_hook.maybe_scrape` with a snapshot of
vLLM's KV usage, reads the registered trees from the
:class:`~canopy_verl.scheduler.placement.TreeAwareLoadBalancer`, and sends
requests to the vLLM servers.  ``canopy_scheduler.admission.mode`` selects one
of three tiers:

  a   Pressure monitor only.  Under KV pressure (usage above
      ``kv_pressure_threshold`` or new preemptions) it sends a ``max_tokens=1``
      dummy request to each server (at most once per
      ``keepalive_interval_sec``) and ranks trees for release without acting
      on the ranking.  With ``abort_mode="all_on_cascade"`` it calls
      ``abort_all_requests(reset_prefix_cache=False)`` after sustained
      preemption pressure.  Tier a is not part of the evaluated system.
  b1  Shared-prefix retention.  On every tick, for each tree with in-flight
      requests, the KV blocks of its ``prefix_lock_top_k`` most recent
      in-flight requests (which include the tree's shared prefix) are locked
      in vLLM's block pool for ``prefix_lock_ttl_sec`` (at most once per
      ``prefix_lock_refresh_sec`` per tree).  Eviction skips locked blocks; see
      :mod:`canopy_verl.vllm_patch.prefix_lock`.
  b2  b1 plus prefix-preserving partial preemption.  Once KV usage stays above
      ``partial_preempt_pressure_threshold`` for
      ``partial_preempt_pressure_consecutive`` pressure ticks, the
      ``partial_preempt_top_k`` most recent in-flight requests of each tree
      are tagged; if vLLM preempts a tagged request, only its private suffix
      is freed and up to ``partial_preempt_keep_n_blocks`` leading blocks are
      kept for the resumed request.  See
      :mod:`canopy_verl.vllm_patch.partial_preempt`.

If a vLLM server lacks the patched RPC methods, b1 / b2 switch off for the rest
of the run (``admission/b1_degraded_to_a`` / ``admission/b2_degraded_to_a``);
in mode b2 this also stops the prefix locks, which run inside the b2 path.

Configuration (``canopy_scheduler.admission``)::

    enabled:                 false   # master switch
    mode:                    "a"     # "a" | "b1" | "b2"
    kv_pressure_threshold:   0.85    # KV usage above which a tick counts as pressure
    keepalive_interval_sec:  30      # tier a dummy-request cadence per server
    abort_low_value:         true    # tier a: count release decisions
    abort_mode:              "observe"   # "observe" | "all_on_cascade"
    cascade_threshold:       3
    cascade_consecutive:     3       # consecutive pressure ticks before a cascade abort
    keepalive_max_tokens:    1
    probe_interval_sec:      5
    top_k_abort:             3
    prefix_lock_ttl_sec:     60.0    # b1 / b2
    prefix_lock_refresh_sec: 15.0
    prefix_lock_top_k:       1
    partial_preempt_keep_n_blocks:        8     # b2
    partial_preempt_ttl_sec:              60.0
    partial_preempt_refresh_sec:          5.0
    partial_preempt_top_k:                3
    partial_preempt_pressure_threshold:   0.80
    partial_preempt_pressure_consecutive: 2
    partial_preempt_safety_margin_blocks: 1
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class AdmissionConfig:
    """Configuration of the admission coordinator (``canopy_scheduler.admission``)."""

    enabled: bool = False
    mode: str = "a"  # "a" | "b1" | "b2"
    # A tick is under pressure when the KV usage exceeds this (strictly) or new
    # preemptions occurred since the previous tick.
    kv_pressure_threshold: float = 0.85
    keepalive_interval_sec: int = 30
    abort_low_value: bool = True
    probe_interval_sec: int = 5
    # -- tier a actions --
    # "observe": count release decisions only, never call vLLM.
    # "all_on_cascade": call server.abort_all_requests(reset_prefix_cache=False)
    #   after `cascade_consecutive` consecutive pressure ticks with new preemptions.
    abort_mode: str = "observe"
    cascade_threshold: int = 3
    cascade_consecutive: int = 3
    # When True, a cascade abort no longer requires new preemptions: sustained
    # high KV usage alone triggers it.  Used to exercise the abort path on
    # setups where vLLM never preempts.
    cascade_allow_usage_only: bool = False
    keepalive_max_tokens: int = 1
    top_k_abort: int = 3
    # -- b1: shared-prefix retention --
    # Lock lifetime.  It should outlast the gap between two rollout requests of
    # a tree (e.g. a tool call), otherwise the lock expires before reuse.
    prefix_lock_ttl_sec: float = 60.0
    # Minimum interval between lock requests for the same tree.
    prefix_lock_refresh_sec: float = 15.0
    # Number of most recent in-flight requests of a tree whose blocks are locked.
    prefix_lock_top_k: int = 1
    # -- b2: prefix-preserving partial preemption --
    partial_preempt_keep_n_blocks: int = 8
    partial_preempt_ttl_sec: float = 60.0
    partial_preempt_refresh_sec: float = 5.0
    partial_preempt_top_k: int = 3
    partial_preempt_pressure_threshold: float = 0.80
    partial_preempt_pressure_consecutive: int = 2
    partial_preempt_safety_margin_blocks: int = 1

    @classmethod
    def from_hydra(cls, cfg) -> "AdmissionConfig":
        """Build from the ``canopy_scheduler`` node (hydra / OmegaConf or any ``.get`` mapping)."""
        if cfg is None:
            return cls()
        admission = getattr(cfg, "admission", None)
        if admission is None:
            return cls()

        def _get(key, default):
            try:
                v = admission.get(key, default)
            except Exception:
                v = getattr(admission, key, default)
            return default if v is None else v

        return cls(
            enabled=bool(_get("enabled", False)),
            mode=str(_get("mode", "a")).lower(),
            kv_pressure_threshold=float(_get("kv_pressure_threshold", 0.85)),
            keepalive_interval_sec=int(_get("keepalive_interval_sec", 30)),
            abort_low_value=bool(_get("abort_low_value", True)),
            probe_interval_sec=int(_get("probe_interval_sec", 5)),
            abort_mode=str(_get("abort_mode", "observe")).lower(),
            cascade_threshold=int(_get("cascade_threshold", 3)),
            cascade_consecutive=int(_get("cascade_consecutive", 3)),
            cascade_allow_usage_only=bool(_get("cascade_allow_usage_only", False)),
            keepalive_max_tokens=int(_get("keepalive_max_tokens", 1)),
            top_k_abort=int(_get("top_k_abort", 3)),
            prefix_lock_ttl_sec=float(_get("prefix_lock_ttl_sec", 60.0)),
            prefix_lock_refresh_sec=float(_get("prefix_lock_refresh_sec", 15.0)),
            prefix_lock_top_k=int(_get("prefix_lock_top_k", 1)),
            partial_preempt_keep_n_blocks=int(
                _get("partial_preempt_keep_n_blocks", 8)
            ),
            partial_preempt_ttl_sec=float(_get("partial_preempt_ttl_sec", 60.0)),
            partial_preempt_refresh_sec=float(
                _get("partial_preempt_refresh_sec", 5.0)
            ),
            partial_preempt_top_k=int(_get("partial_preempt_top_k", 3)),
            partial_preempt_pressure_threshold=float(
                _get("partial_preempt_pressure_threshold", 0.80)
            ),
            partial_preempt_pressure_consecutive=int(
                _get("partial_preempt_pressure_consecutive", 2)
            ),
            partial_preempt_safety_margin_blocks=int(
                _get("partial_preempt_safety_margin_blocks", 1)
            ),
        )

    def validate(self) -> list[str]:
        """Return a list of human-readable errors; empty if valid."""
        errs = []
        if self.mode not in ("a", "b1", "b2"):
            errs.append(f"mode must be a/b1/b2, got {self.mode!r}")
        if not 0.0 < self.kv_pressure_threshold <= 1.0:
            errs.append(
                f"kv_pressure_threshold must be in (0,1], got {self.kv_pressure_threshold}"
            )
        if self.keepalive_interval_sec < 1:
            errs.append("keepalive_interval_sec must be >= 1")
        if self.probe_interval_sec < 1:
            errs.append("probe_interval_sec must be >= 1")
        if self.abort_mode not in ("observe", "all_on_cascade"):
            errs.append(
                f"abort_mode must be observe|all_on_cascade, got {self.abort_mode!r}"
            )
        if self.cascade_threshold < 1:
            errs.append("cascade_threshold must be >= 1")
        if self.cascade_consecutive < 1:
            errs.append("cascade_consecutive must be >= 1")
        if self.keepalive_max_tokens < 1:
            errs.append("keepalive_max_tokens must be >= 1")
        if self.top_k_abort < 0:
            errs.append("top_k_abort must be >= 0")
        if self.partial_preempt_keep_n_blocks < 0:
            errs.append("partial_preempt_keep_n_blocks must be >= 0")
        if self.partial_preempt_ttl_sec <= 0:
            errs.append("partial_preempt_ttl_sec must be > 0")
        if self.partial_preempt_refresh_sec <= 0:
            errs.append("partial_preempt_refresh_sec must be > 0")
        if self.partial_preempt_top_k < 0:
            errs.append("partial_preempt_top_k must be >= 0")
        if not 0.0 < self.partial_preempt_pressure_threshold <= 1.0:
            errs.append(
                f"partial_preempt_pressure_threshold must be in (0,1], got "
                f"{self.partial_preempt_pressure_threshold}"
            )
        if self.partial_preempt_pressure_consecutive < 1:
            errs.append("partial_preempt_pressure_consecutive must be >= 1")
        if self.partial_preempt_safety_margin_blocks < 0:
            errs.append("partial_preempt_safety_margin_blocks must be >= 0")
        return errs


# ---------------------------------------------------------------------
# Pressure detection (pure functions)
# ---------------------------------------------------------------------


@dataclass
class KVPressureSnapshot:
    """One scrape of a vLLM server's metrics; input of a tick."""

    gpu_cache_usage_perc: float = 0.0
    num_preempted_total: int = 0
    num_running_seqs: int = 0
    num_pending_seqs: int = 0
    timestamp_s: float = 0.0


@dataclass
class AdmissionDecision:
    """What the pressure monitor recommends for one tick."""

    under_pressure: bool = False
    should_keepalive: bool = False
    should_abort_low_value: bool = False
    reason: str = ""
    snapshot: Optional[KVPressureSnapshot] = None


def detect_kv_pressure(
    snapshot: KVPressureSnapshot,
    cfg: AdmissionConfig,
    preempt_delta: int = 0,
) -> AdmissionDecision:
    """Decide whether the KV cache is under pressure.

    ``under_pressure`` holds iff ``gpu_cache_usage_perc > kv_pressure_threshold``
    or ``preempt_delta > 0`` (new preemptions since the previous tick).  Under
    pressure, ``should_keepalive`` is set and ``should_abort_low_value`` follows
    ``cfg.abort_low_value``.

    Args:
        snapshot: latest metrics scrape.
        cfg: thresholds.
        preempt_delta: increase of ``num_preempted_total`` since the previous
            tick (tracked by the caller).
    """
    if not cfg.enabled:
        return AdmissionDecision(reason="admission_disabled", snapshot=snapshot)

    usage_high = snapshot.gpu_cache_usage_perc > cfg.kv_pressure_threshold
    preempt_seen = preempt_delta > 0

    under_pressure = usage_high or preempt_seen

    if not under_pressure:
        return AdmissionDecision(reason="no_pressure", snapshot=snapshot)

    if usage_high and preempt_seen:
        reason = "usage_high+preempt"
    elif usage_high:
        reason = "usage_above_threshold"
    else:
        reason = "new_preemptions"

    return AdmissionDecision(
        under_pressure=True,
        should_keepalive=True,
        should_abort_low_value=cfg.abort_low_value,
        reason=reason,
        snapshot=snapshot,
    )


# ---------------------------------------------------------------------
# Tree ranking (pure function)
# ---------------------------------------------------------------------


@dataclass
class TreeValueSnapshot:
    """Per-tree state the admission coordinator works on (from the load balancer)."""

    tree_uid: str
    # Siblings that already produced their final response.
    complete_siblings: int = 0
    # Siblings still generating.
    active_siblings: int = 0
    # Approximate KV footprint of the tree.
    kv_blocks_approx: int = 0
    private_kv_blocks_approx: int = 0
    # Server the tree is placed on; empty when unknown (the tree is skipped).
    primary_server: str = ""
    # Last k in-flight vLLM request ids: the only ids whose blocks vLLM can still
    # resolve, hence the targets of lock and preemption-tag requests.
    active_request_ids: list[str] = field(default_factory=list)
    # Last k released request ids (blocks already freed; informational).
    completed_request_ids: list[str] = field(default_factory=list)


def rank_trees_for_abort(
    trees: list[TreeValueSnapshot], top_k: int = 3
) -> list[TreeValueSnapshot]:
    """Top-k trees whose KV is safest to release: most complete first.

    Ties on the completion ratio go to the larger ``kv_blocks_approx`` (more KV
    reclaimed), then to the smaller ``tree_uid`` (deterministic).
    """
    if top_k <= 0 or not trees:
        return []

    def _score(t: TreeValueSnapshot) -> tuple[float, int]:
        total = t.complete_siblings + t.active_siblings
        complete_ratio = t.complete_siblings / total if total > 0 else 0.0
        return (complete_ratio, t.kv_blocks_approx)

    scored = [(- _score(t)[0], -_score(t)[1], t.tree_uid, t) for t in trees]
    scored.sort(key=lambda q: (q[0], q[1], q[2]))
    return [t for *_, t in scored][:top_k]


def build_keepalive_prompt(shared_prefix_tokens: list[int]) -> list[int]:
    """Prompt that re-reads a shared prefix: the prefix itself, as a new list.

    The caller sends it with ``max_tokens=1`` so that prefix caching records a
    hit on the prefix and refreshes its LRU position.
    """
    return list(shared_prefix_tokens)


# ---------------------------------------------------------------------
# Pressure monitor (pure state machine)
# ---------------------------------------------------------------------


class AdmissionMonitor:
    """Tracks ``num_preempted_total`` across ticks and evaluates each tick.

    :meth:`tick` turns a :class:`KVPressureSnapshot` into an
    :class:`AdmissionDecision`; the coordinator executes the decision.
    """

    def __init__(self, cfg: AdmissionConfig):
        errs = cfg.validate()
        if errs:
            raise ValueError(f"invalid AdmissionConfig: {errs}")
        self.cfg = cfg
        self._last_preempted_total: Optional[int] = None
        self._m_pressure_ticks: int = 0
        self._m_keepalive_dispatched: int = 0
        self._m_aborts_dispatched: int = 0
        self._m_total_ticks: int = 0

    def tick(self, snapshot: KVPressureSnapshot) -> AdmissionDecision:
        """Evaluate one scrape and return the decision."""
        self._m_total_ticks += 1

        if self._last_preempted_total is None:
            preempt_delta = 0  # first tick: no baseline yet
        else:
            preempt_delta = max(
                0, snapshot.num_preempted_total - self._last_preempted_total
            )
        self._last_preempted_total = snapshot.num_preempted_total

        decision = detect_kv_pressure(snapshot, self.cfg, preempt_delta)

        if decision.under_pressure:
            self._m_pressure_ticks += 1
        if decision.should_keepalive:
            self._m_keepalive_dispatched += 1
        if decision.should_abort_low_value:
            self._m_aborts_dispatched += 1

        return decision

    def reset(self) -> None:
        """Reset counters and the preemption baseline."""
        self._last_preempted_total = None
        self._m_pressure_ticks = 0
        self._m_keepalive_dispatched = 0
        self._m_aborts_dispatched = 0
        self._m_total_ticks = 0

    def get_metrics(self) -> dict[str, Any]:
        """``admission/*`` counters of the monitor."""
        pressure_ratio = (
            self._m_pressure_ticks / self._m_total_ticks
            if self._m_total_ticks > 0
            else 0.0
        )
        return {
            "admission/total_ticks": self._m_total_ticks,
            "admission/pressure_ticks": self._m_pressure_ticks,
            "admission/pressure_ratio": pressure_ratio,
            "admission/keepalive_dispatched": self._m_keepalive_dispatched,
            "admission/aborts_dispatched": self._m_aborts_dispatched,
        }


# ---------------------------------------------------------------------
# AdmissionCoordinator
# ---------------------------------------------------------------------


class AdmissionCoordinatorImpl:
    """Admission coordinator logic; used as the Ray actor :data:`AdmissionCoordinator`.

    Combines :class:`AdmissionMonitor` (pressure state), the
    ``TreeAwareLoadBalancer`` (which trees exist, where, and their in-flight
    requests) and the vLLM server handles (where requests are sent).  Created
    by ``CanopyAgentLoopManager`` after the balancer and attached to it with
    ``set_admission_coordinator``; it lives as long as the driver.

    One :meth:`tick` (driven by ``prom_hook.maybe_scrape``):

      1. ``monitor.tick(snapshot)`` gives an :class:`AdmissionDecision`.
      2. Tier a: under pressure, send keepalive requests and rank trees
         (``abort_mode`` decides whether to abort).
      3. Mode b1 / b2: lock the blocks of each tree's latest in-flight
         requests (independent of pressure) and poll the lock counters.
      4. Mode b2: once the partial-preemption gate opens, tag each tree's
         latest in-flight requests; always poll the preemption counters.
      5. Return :meth:`get_metrics` (a plain dict).

    All server and balancer calls go through Ray handles (``x.method.remote``);
    tests can pass any objects with the same interface.
    """

    def __init__(
        self,
        cfg: "AdmissionConfig",
        lb_handle: Any | None = None,
        server_handles: dict[str, Any] | None = None,
        vllm_dummy_prompt_ids: list[int] | None = None,
    ) -> None:
        errs = cfg.validate()
        if errs:
            raise ValueError(f"invalid AdmissionConfig: {errs}")
        self._cfg = cfg
        self._monitor = AdmissionMonitor(cfg)
        self._lb_handle = lb_handle
        self._server_handles: dict[str, Any] = dict(server_handles or {})
        self._dummy_prompt_ids: list[int] = list(vllm_dummy_prompt_ids or [1, 2, 3])

        # Tier a state.
        self._last_keepalive_per_server: dict[str, float] = {}
        self._consecutive_pressure_ticks: int = 0
        self._last_cascade_abort_ts: float = 0.0
        self._m_keepalive_calls_attempted: int = 0
        self._m_keepalive_calls_succeeded: int = 0
        self._m_keepalive_errors: int = 0
        self._m_cascade_aborts_attempted: int = 0
        self._m_cascade_aborts_succeeded: int = 0
        self._m_abort_errors: int = 0
        self._m_tick_errors: int = 0
        self._m_lb_query_errors: int = 0
        self._last_reason: str = ""
        self._m_abort_decisions_observe_only: int = 0
        self._startup_marker_printed: bool = False

        # b1 state: last lock time per tree (rate limit) and, for unlocking on
        # tree completion, the tree's server and the request ids last seen.
        self._last_prefix_lock_ts_per_tree: dict[str, float] = {}
        self._tree_primary_server: dict[str, str] = {}
        self._tree_completed_reqs: dict[str, list[str]] = {}
        self._m_prefix_lock_attempts: int = 0
        self._m_prefix_lock_succeeded: int = 0
        self._m_prefix_lock_errors: int = 0
        self._m_prefix_lock_blocks_total: int = 0
        self._m_prefix_unlock_attempts: int = 0
        self._m_prefix_unlock_succeeded: int = 0
        self._m_prefix_unlock_errors: int = 0
        # Mirrors of the patched BlockPool counters, polled from one server per
        # tick; unchanged when the poll fails.
        self._m_prefix_lock_active_blocks: int = 0
        self._m_prefix_lock_evictions_skipped: int = 0
        self._m_prefix_lock_eviction_forced: int = 0
        self._m_prefix_lock_ttl_expired_events: int = 0
        # Number of (lock-aware) get_new_blocks calls in the engine.  Non-zero
        # invocations with zero skipped evictions mean the patch is active but
        # there was no memory pressure on locked blocks.
        self._m_get_new_blocks_invocations: int = 0
        self._m_get_new_blocks_with_locks: int = 0
        # Set when a server lacks the patched RPCs; b1 then stays off.
        self._b1_degraded_to_a: bool = False

        # b2 state: last tag time per tree (rate limit) and counters.
        self._last_partial_preempt_ts_per_tree: dict[str, float] = {}
        self._m_partial_preempt_attempts: int = 0
        self._m_partial_preempt_succeeded: int = 0
        self._m_partial_preempt_errors: int = 0
        self._m_partial_preempt_kept_blocks_total: int = 0
        self._m_partial_preempt_released_blocks_total: int = 0
        self._m_partial_preempt_fallback_to_full: int = 0
        self._m_partial_preempt_ttl_expired_at_preempt: int = 0
        self._m_partial_preempt_skip_due_to_no_common_prefix: int = 0
        self._m_partial_preempt_coordinator_unsync_warnings: int = 0
        # Engine-side counters mirrored by _poll_partial_preempt_stats.
        self._m_partial_preempt_preempt_invocations: int = 0
        self._m_partial_preempt_preempt_partial_path_taken: int = 0
        self._m_partial_preempt_tags_set: int = 0
        self._m_partial_preempt_tags_unset: int = 0
        # Set when a server lacks the patched RPCs; b2 then stays off.
        self._b2_degraded_to_a: bool = False

    # ------- public API ---------------------------------------------

    async def tick(self, snapshot: "KVPressureSnapshot") -> dict[str, Any]:
        """Run one cycle and return the metrics dict."""
        if not self._startup_marker_printed:
            self._startup_marker_printed = True
            print(
                f"[ADMISSION-COORD] startup mode={self._cfg.mode} "
                f"abort_mode={self._cfg.abort_mode} "
                f"threshold={self._cfg.kv_pressure_threshold} "
                f"keepalive_interval_sec={self._cfg.keepalive_interval_sec}",
                flush=True,
            )

        try:
            decision = self._monitor.tick(snapshot)
            self._last_reason = decision.reason or ""
        except Exception as exc:  # never fail the caller
            self._m_tick_errors += 1
            logger.warning("[ADMISSION-COORD] tick err: %s", exc)
            return self.get_metrics()

        if not decision.under_pressure:
            self._consecutive_pressure_ticks = 0
        else:
            self._consecutive_pressure_ticks += 1

        # Modes b1 / b2 need the trees on every tick: locks are issued while
        # siblings are in flight, not only under pressure.
        active_trees: list[TreeValueSnapshot] = []
        need_trees = (
            decision.should_keepalive
            or decision.should_abort_low_value
            or (self._cfg.mode == "b1" and not self._b1_degraded_to_a)
            or (self._cfg.mode == "b2" and not self._b2_degraded_to_a)
        )
        if need_trees:
            active_trees = await self._query_active_trees()

        if decision.should_keepalive and active_trees:
            await self._dispatch_keepalive(active_trees)

        if decision.should_abort_low_value:
            ranked = rank_trees_for_abort(active_trees, top_k=self._cfg.top_k_abort)
            if self._cfg.abort_mode == "observe":
                self._m_abort_decisions_observe_only += 1
            elif self._cfg.abort_mode == "all_on_cascade":
                if self._cascade_gate(decision):
                    await self._dispatch_cascade_abort(ranked)

        # b1: lock independently of pressure.  A failed RPC may switch b1 off.
        if (
            self._cfg.mode == "b1"
            and not self._b1_degraded_to_a
            and active_trees
        ):
            await self._dispatch_prefix_lock(active_trees)
            await self._poll_block_pool_stats()

        # b2 = b1 + partial-preemption tags (b2 applies the b1 patches too).
        if (
            self._cfg.mode == "b2"
            and not self._b2_degraded_to_a
        ):
            if active_trees and not self._b1_degraded_to_a:
                await self._dispatch_prefix_lock(active_trees)
                await self._poll_block_pool_stats()
            if active_trees and self._partial_preempt_gate(decision):
                await self._dispatch_partial_preempt(active_trees)
            await self._poll_partial_preempt_stats()

        return self.get_metrics()

    def get_metrics(self) -> dict[str, Any]:
        """Serializable ``admission/*`` metrics."""
        base = self._monitor.get_metrics()
        base.update(
            {
                "admission/keepalive_calls_attempted": self._m_keepalive_calls_attempted,
                "admission/keepalive_calls_succeeded": self._m_keepalive_calls_succeeded,
                "admission/keepalive_errors": self._m_keepalive_errors,
                "admission/cascade_aborts_attempted": self._m_cascade_aborts_attempted,
                "admission/cascade_aborts_succeeded": self._m_cascade_aborts_succeeded,
                "admission/abort_errors": self._m_abort_errors,
                "admission/abort_decisions_observe_only": self._m_abort_decisions_observe_only,
                "admission/tick_errors": self._m_tick_errors,
                "admission/lb_query_errors": self._m_lb_query_errors,
                "admission/consecutive_pressure_ticks": self._consecutive_pressure_ticks,
                # b1
                "admission/prefix_lock_attempts": self._m_prefix_lock_attempts,
                "admission/prefix_lock_succeeded": self._m_prefix_lock_succeeded,
                "admission/prefix_lock_errors": self._m_prefix_lock_errors,
                "admission/prefix_lock_blocks_total": self._m_prefix_lock_blocks_total,
                "admission/prefix_unlock_attempts": self._m_prefix_unlock_attempts,
                "admission/prefix_unlock_succeeded": self._m_prefix_unlock_succeeded,
                "admission/prefix_unlock_errors": self._m_prefix_unlock_errors,
                "admission/prefix_lock_active_blocks": self._m_prefix_lock_active_blocks,
                "admission/prefix_lock_evictions_skipped": self._m_prefix_lock_evictions_skipped,
                "admission/prefix_lock_eviction_forced": self._m_prefix_lock_eviction_forced,
                "admission/prefix_lock_ttl_expired_events": self._m_prefix_lock_ttl_expired_events,
                "admission/b1_degraded_to_a": int(self._b1_degraded_to_a),
                "admission/get_new_blocks_invocations": self._m_get_new_blocks_invocations,
                "admission/get_new_blocks_with_locks": self._m_get_new_blocks_with_locks,
                # b2 (``partial_preempt_succeeded`` counts accepted tag requests;
                # ``partial_preempt_preempt_partial_path_taken`` counts preemptions
                # that took the partial path in the engine)
                "admission/partial_preempt_attempts": self._m_partial_preempt_attempts,
                "admission/partial_preempt_succeeded": self._m_partial_preempt_succeeded,
                "admission/partial_preempt_errors": self._m_partial_preempt_errors,
                "admission/partial_preempt_kept_blocks_total": self._m_partial_preempt_kept_blocks_total,
                "admission/partial_preempt_released_blocks_total": self._m_partial_preempt_released_blocks_total,
                "admission/partial_preempt_fallback_to_full": self._m_partial_preempt_fallback_to_full,
                "admission/partial_preempt_ttl_expired_at_preempt": self._m_partial_preempt_ttl_expired_at_preempt,
                "admission/partial_preempt_skip_due_to_no_common_prefix": self._m_partial_preempt_skip_due_to_no_common_prefix,
                "admission/partial_preempt_coordinator_unsync_warnings": self._m_partial_preempt_coordinator_unsync_warnings,
                "admission/partial_preempt_preempt_invocations": self._m_partial_preempt_preempt_invocations,
                "admission/partial_preempt_preempt_partial_path_taken": self._m_partial_preempt_preempt_partial_path_taken,
                "admission/partial_preempt_tags_set": self._m_partial_preempt_tags_set,
                "admission/partial_preempt_tags_unset": self._m_partial_preempt_tags_unset,
                "admission/b2_degraded_to_a": int(self._b2_degraded_to_a),
            }
        )
        return base

    def reset_metrics(self) -> None:
        self._monitor.reset()
        self._m_keepalive_calls_attempted = 0
        self._m_keepalive_calls_succeeded = 0
        self._m_keepalive_errors = 0
        self._m_cascade_aborts_attempted = 0
        self._m_cascade_aborts_succeeded = 0
        self._m_abort_errors = 0
        self._m_abort_decisions_observe_only = 0
        self._m_tick_errors = 0
        self._m_lb_query_errors = 0
        self._consecutive_pressure_ticks = 0

    def ping(self) -> str:
        """Liveness check."""
        return "ok"

    def set_server_handles(self, handles: dict[str, Any]) -> None:
        """Replace the ``{server_id: actor handle}`` map."""
        self._server_handles = dict(handles or {})

    def set_dummy_prompt_ids(self, prompt_ids: list[int]) -> None:
        """Override the prompt of tier-a keepalive requests."""
        self._dummy_prompt_ids = list(prompt_ids or [1, 2, 3])

    # ------- internal helpers --------------------------------------

    async def _query_active_trees(self) -> list[TreeValueSnapshot]:
        if self._lb_handle is None:
            return []
        try:
            snaps = await self._lb_handle.get_active_trees_snapshot.remote()
            if isinstance(snaps, list):
                return [s for s in snaps if isinstance(s, TreeValueSnapshot)]
            return []
        except Exception as exc:
            self._m_lb_query_errors += 1
            logger.warning("[ADMISSION-COORD] lb query err: %s", exc)
            return []

    def _cascade_gate(self, decision: "AdmissionDecision") -> bool:
        """Cascade abort needs ``cascade_consecutive`` pressure ticks and, unless
        ``cascade_allow_usage_only``, new preemptions in this tick."""
        if decision.snapshot is None:
            return False
        if self._consecutive_pressure_ticks < self._cfg.cascade_consecutive:
            return False
        if not self._cfg.cascade_allow_usage_only:
            if "preempt" not in (decision.reason or ""):
                return False
        return True

    async def _dispatch_keepalive(
        self, active_trees: list[TreeValueSnapshot]
    ) -> None:
        """Send one dummy request per server, at most once per ``keepalive_interval_sec``."""
        if not self._server_handles:
            return
        now = time.time()
        interval = float(max(1, self._cfg.keepalive_interval_sec))
        for server_id in list(self._server_handles):
            last = self._last_keepalive_per_server.get(server_id, 0.0)
            if now - last < interval:
                continue
            self._last_keepalive_per_server[server_id] = now
            self._m_keepalive_calls_attempted += 1
            try:
                await self._send_dummy_generate(server_id)
                self._m_keepalive_calls_succeeded += 1
            except Exception as exc:
                self._m_keepalive_errors += 1
                logger.warning(
                    "[ADMISSION-COORD] keepalive err server=%s: %s",
                    server_id,
                    exc,
                )

    async def _send_dummy_generate(self, server_id: str) -> None:
        """``max_tokens``-limited generate on one server, shaped like verl's calls."""
        handle = self._server_handles.get(server_id)
        if handle is None:
            raise RuntimeError(f"no handle for server {server_id}")
        from uuid import uuid4

        sampling_params = {
            "max_tokens": int(self._cfg.keepalive_max_tokens),
            "temperature": 0.0,
        }
        await handle.generate.remote(
            request_id=uuid4().hex,
            prompt_ids=list(self._dummy_prompt_ids),
            sampling_params=sampling_params,
        )

    async def _dispatch_prefix_lock(
        self, active_trees: list[TreeValueSnapshot]
    ) -> None:
        """b1: lock the blocks of each tree's latest in-flight requests.

        For every tree with in-flight requests and a known server, call
        ``server.set_prefix_lock_for_request(request_id, ttl_sec)`` for the last
        ``prefix_lock_top_k`` ids of ``active_request_ids`` (only in-flight ids
        can be resolved to blocks by vLLM).  Rate-limited per tree by
        ``prefix_lock_refresh_sec``.
        """
        if not self._server_handles:
            return
        now = time.time()
        refresh = float(max(1.0, self._cfg.prefix_lock_refresh_sec))
        top_k = max(1, int(self._cfg.prefix_lock_top_k))
        for snap in active_trees:
            if not snap.active_request_ids:
                continue
            if not snap.primary_server:
                continue
            handle = self._server_handles.get(snap.primary_server)
            if handle is None:
                continue
            last = self._last_prefix_lock_ts_per_tree.get(snap.tree_uid, 0.0)
            if now - last < refresh:
                continue
            self._last_prefix_lock_ts_per_tree[snap.tree_uid] = now

            # Remember the server and all in-flight ids for the unlock on tree
            # completion (ids locked on earlier ticks may have left the top-k
            # window; unlocking an unlocked id is a no-op).
            self._tree_primary_server[snap.tree_uid] = snap.primary_server
            self._tree_completed_reqs[snap.tree_uid] = list(
                snap.active_request_ids
            )

            target_req_ids = list(snap.active_request_ids[-top_k:])
            for rid in target_req_ids:
                self._m_prefix_lock_attempts += 1
                try:
                    result = await handle.set_prefix_lock_for_request.remote(
                        rid,
                        float(self._cfg.prefix_lock_ttl_sec),
                    )
                except AttributeError:
                    self._maybe_degrade_b1(
                        reason="vLLMHttpServer missing set_prefix_lock_for_request"
                    )
                    return
                except Exception as exc:
                    self._m_prefix_lock_errors += 1
                    logger.warning(
                        "[ADMISSION-COORD] prefix_lock RPC err tree=%s rid=%s: %s",
                        snap.tree_uid[:8],
                        rid,
                        exc,
                    )
                    continue
                if isinstance(result, dict):
                    if result.get("error"):
                        self._m_prefix_lock_errors += 1
                        logger.warning(
                            "[ADMISSION-COORD] prefix_lock remote error tree=%s rid=%s: %s",
                            snap.tree_uid[:8],
                            rid,
                            result.get("error"),
                        )
                    else:
                        self._m_prefix_lock_succeeded += 1
                        self._m_prefix_lock_blocks_total += int(
                            result.get("locked", 0)
                        )

    async def _poll_block_pool_stats(self) -> None:
        """Mirror the patched BlockPool counters from the first server that answers.

        With several replicas this reports one replica's counters.
        """
        if not self._server_handles:
            return
        for server_id, handle in self._server_handles.items():
            try:
                stats = await handle.get_prefix_lock_stats.remote()
            except AttributeError:
                self._maybe_degrade_b1(
                    reason="vLLMHttpServer missing get_prefix_lock_stats"
                )
                return
            except Exception:
                continue
            if isinstance(stats, dict) and "error" not in stats:
                self._m_prefix_lock_active_blocks = int(
                    stats.get("locked_blocks", self._m_prefix_lock_active_blocks)
                )
                self._m_prefix_lock_evictions_skipped = int(
                    stats.get(
                        "evictions_skipped_due_to_lock",
                        self._m_prefix_lock_evictions_skipped,
                    )
                )
                self._m_prefix_lock_eviction_forced = int(
                    stats.get(
                        "locked_eviction_forced",
                        self._m_prefix_lock_eviction_forced,
                    )
                )
                self._m_prefix_lock_ttl_expired_events = int(
                    stats.get(
                        "ttl_expired_events_total",
                        self._m_prefix_lock_ttl_expired_events,
                    )
                )
                self._m_get_new_blocks_invocations = int(
                    stats.get(
                        "get_new_blocks_invocations",
                        self._m_get_new_blocks_invocations,
                    )
                )
                self._m_get_new_blocks_with_locks = int(
                    stats.get(
                        "get_new_blocks_with_locks",
                        self._m_get_new_blocks_with_locks,
                    )
                )
                return

    async def dispatch_prefix_unlock_on_complete(self, tree_uid: str) -> None:
        """Unlock a finished tree's requests (called by the balancer's ``on_tree_complete``).

        Uses the server and request ids remembered by
        :meth:`_dispatch_prefix_lock`; the lock TTL covers a missing or failed
        call.
        """
        if self._b1_degraded_to_a:
            return
        server_id = self._tree_primary_server.pop(tree_uid, "")
        req_ids = self._tree_completed_reqs.pop(tree_uid, [])
        self._last_prefix_lock_ts_per_tree.pop(tree_uid, None)
        if not req_ids or not server_id:
            return
        handle = self._server_handles.get(server_id)
        if handle is None:
            return
        for rid in req_ids:
            self._m_prefix_unlock_attempts += 1
            try:
                result = await handle.unset_prefix_lock_for_request.remote(rid)
            except AttributeError:
                self._maybe_degrade_b1(
                    reason="vLLMHttpServer missing unset_prefix_lock_for_request"
                )
                return
            except Exception:
                self._m_prefix_unlock_errors += 1
                continue
            if isinstance(result, dict) and result.get("error") is None:
                self._m_prefix_unlock_succeeded += 1
            else:
                self._m_prefix_unlock_errors += 1

    def _maybe_degrade_b1(self, reason: str) -> None:
        """Switch b1 off for the rest of the run."""
        if self._b1_degraded_to_a:
            return
        self._b1_degraded_to_a = True
        print(
            f"[ADMISSION-COORD] B1 degraded to A due to patch failure: {reason}",
            flush=True,
        )

    # ------- b2: prefix-preserving partial preemption -----------------

    def _partial_preempt_gate(self, decision: "AdmissionDecision") -> bool:
        """Open when KV usage exceeds ``partial_preempt_pressure_threshold`` and the
        last ``partial_preempt_pressure_consecutive`` ticks were under pressure.

        The threshold sits below vLLM's preemption point, so tags are in place
        before the scheduler preempts.
        """
        if decision.snapshot is None:
            return False
        usage = float(getattr(decision.snapshot, "gpu_cache_usage_perc", 0.0) or 0.0)
        if usage <= float(self._cfg.partial_preempt_pressure_threshold):
            return False
        if (
            self._consecutive_pressure_ticks
            < int(self._cfg.partial_preempt_pressure_consecutive)
        ):
            return False
        return True

    async def _dispatch_partial_preempt(
        self, active_trees: list[TreeValueSnapshot]
    ) -> None:
        """b2: tag each tree's latest in-flight requests for partial preemption.

        Calls ``server.set_partial_preempt_for_request(request_id, keep_n, ttl)``
        for the last ``partial_preempt_top_k`` in-flight ids of each tree, so
        that a later preemption of those requests keeps their first ``keep_n``
        blocks.  Rate-limited per tree by ``partial_preempt_refresh_sec``.
        """
        if not self._server_handles:
            return
        now = time.time()
        refresh = float(max(1.0, self._cfg.partial_preempt_refresh_sec))
        top_k = max(1, int(self._cfg.partial_preempt_top_k))
        keep_n = max(0, int(self._cfg.partial_preempt_keep_n_blocks))
        ttl_sec = float(self._cfg.partial_preempt_ttl_sec)
        for snap in active_trees:
            if not snap.active_request_ids:
                continue
            if not snap.primary_server:
                continue
            handle = self._server_handles.get(snap.primary_server)
            if handle is None:
                continue
            last = self._last_partial_preempt_ts_per_tree.get(snap.tree_uid, 0.0)
            if now - last < refresh:
                continue
            self._last_partial_preempt_ts_per_tree[snap.tree_uid] = now

            target_req_ids = list(snap.active_request_ids[-top_k:])
            for rid in target_req_ids:
                self._m_partial_preempt_attempts += 1
                try:
                    result = await handle.set_partial_preempt_for_request.remote(
                        rid,
                        keep_n,
                        float(ttl_sec),
                    )
                except AttributeError:
                    self._maybe_degrade_b2(
                        reason="vLLMHttpServer missing set_partial_preempt_for_request"
                    )
                    return
                except Exception as exc:
                    self._m_partial_preempt_errors += 1
                    logger.warning(
                        "[ADMISSION-COORD] partial_preempt RPC err tree=%s rid=%s: %s",
                        snap.tree_uid[:8] if snap.tree_uid else "?",
                        rid,
                        exc,
                    )
                    continue
                if isinstance(result, dict):
                    if result.get("error"):
                        self._m_partial_preempt_errors += 1
                        logger.warning(
                            "[ADMISSION-COORD] partial_preempt remote error "
                            "tree=%s rid=%s: %s",
                            snap.tree_uid[:8] if snap.tree_uid else "?",
                            rid,
                            result.get("error"),
                        )
                    else:
                        self._m_partial_preempt_succeeded += 1

    async def _poll_partial_preempt_stats(self) -> None:
        """Mirror the engine-side partial-preemption counters from the first server that answers."""
        if not self._server_handles:
            return
        for server_id, handle in self._server_handles.items():
            try:
                stats = await handle.get_partial_preempt_stats.remote()
            except AttributeError:
                self._maybe_degrade_b2(
                    reason="vLLMHttpServer missing get_partial_preempt_stats"
                )
                return
            except Exception:
                continue
            if isinstance(stats, dict) and "error" not in stats:
                self._m_partial_preempt_tags_set = int(
                    stats.get("tags_set", self._m_partial_preempt_tags_set)
                )
                self._m_partial_preempt_tags_unset = int(
                    stats.get("tags_unset", self._m_partial_preempt_tags_unset)
                )
                self._m_partial_preempt_preempt_invocations = int(
                    stats.get(
                        "preempt_invocations",
                        self._m_partial_preempt_preempt_invocations,
                    )
                )
                self._m_partial_preempt_preempt_partial_path_taken = int(
                    stats.get(
                        "preempt_partial_path_taken",
                        self._m_partial_preempt_preempt_partial_path_taken,
                    )
                )
                self._m_partial_preempt_kept_blocks_total = int(
                    stats.get(
                        "kept_blocks_total",
                        self._m_partial_preempt_kept_blocks_total,
                    )
                )
                self._m_partial_preempt_released_blocks_total = int(
                    stats.get(
                        "released_blocks_total",
                        self._m_partial_preempt_released_blocks_total,
                    )
                )
                self._m_partial_preempt_fallback_to_full = int(
                    stats.get(
                        "fallback_to_full",
                        self._m_partial_preempt_fallback_to_full,
                    )
                )
                self._m_partial_preempt_ttl_expired_at_preempt = int(
                    stats.get(
                        "ttl_expired_at_preempt",
                        self._m_partial_preempt_ttl_expired_at_preempt,
                    )
                )
                self._m_partial_preempt_skip_due_to_no_common_prefix = int(
                    stats.get(
                        "skip_due_to_no_common_prefix",
                        self._m_partial_preempt_skip_due_to_no_common_prefix,
                    )
                )
                self._m_partial_preempt_coordinator_unsync_warnings = int(
                    stats.get(
                        "coordinator_unsync_warnings",
                        self._m_partial_preempt_coordinator_unsync_warnings,
                    )
                )
                return

    def _maybe_degrade_b2(self, reason: str) -> None:
        """Switch b2 off for the rest of the run."""
        if self._b2_degraded_to_a:
            return
        self._b2_degraded_to_a = True
        print(
            f"[ADMISSION-COORD] B2 degraded due to patch/RPC failure: {reason}",
            flush=True,
        )

    async def _dispatch_cascade_abort(
        self, ranked_trees: list[TreeValueSnapshot]
    ) -> None:
        """Tier a cascade abort: ``abort_all_requests(reset_prefix_cache=False)`` on one server.

        At most one cascade abort per 60 s.  The first server in the handle map
        is used.
        """
        if not self._server_handles:
            return
        now = time.time()
        if now - self._last_cascade_abort_ts < 60.0:
            return
        self._last_cascade_abort_ts = now

        server_id = next(iter(self._server_handles.keys()))
        handle = self._server_handles[server_id]
        self._m_cascade_aborts_attempted += 1
        try:
            result = await handle.abort_all_requests.remote(
                reset_prefix_cache=False
            )
            if isinstance(result, dict) and result.get("error") is None:
                self._m_cascade_aborts_succeeded += 1
                print(
                    f"[ADMISSION-COORD] cascade abort server={server_id} "
                    f"aborted_count={result.get('aborted_count', 0)}",
                    flush=True,
                )
        except Exception as exc:
            self._m_abort_errors += 1
            logger.warning(
                "[ADMISSION-COORD] cascade abort err server=%s: %s",
                server_id,
                exc,
            )


try:
    import ray as _ray
except Exception:  # pragma: no cover
    _ray = None

if _ray is not None:
    AdmissionCoordinator = _ray.remote(AdmissionCoordinatorImpl)
else:  # pragma: no cover

    class AdmissionCoordinator:  # type: ignore[no-redef]
        """Placeholder used when Ray is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("AdmissionCoordinator requires Ray.")


__all__ = [
    "AdmissionConfig",
    "KVPressureSnapshot",
    "AdmissionDecision",
    "TreeValueSnapshot",
    "detect_kv_pressure",
    "rank_trees_for_abort",
    "build_keepalive_prompt",
    "AdmissionMonitor",
    "AdmissionCoordinatorImpl",
    "AdmissionCoordinator",
]
