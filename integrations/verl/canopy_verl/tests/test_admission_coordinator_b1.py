# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Admission coordinator, mode b1 (shared-prefix retention via prefix locks).

Drives the coordinator logic in-process with stub balancer and vLLM servers.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from canopy_verl.scheduler.admission import (
    AdmissionConfig,
    AdmissionCoordinatorImpl,
    KVPressureSnapshot,
    TreeValueSnapshot,
)
from canopy_verl.tests._fakes import LocalHandle


class StubLB:
    """Stand-in for TreeAwareLoadBalancer that returns a fixed tree list."""

    def __init__(self, trees=()):
        self.trees = list(trees)

    def get_active_trees_snapshot(self):
        return list(self.trees)


class StubVLLMServer:
    """vLLMHttpServer stand-in exposing the prefix-lock RPCs; records calls."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.set_calls: list[tuple[str, float]] = []
        self.unset_calls: list[str] = []
        self.stats_polls = 0

    async def set_prefix_lock_for_request(self, request_id, ttl_sec):
        self.set_calls.append((request_id, ttl_sec))
        if self.fail:
            return {"locked": 0, "error": "simulated"}
        return {"locked": 3, "total_blocks": 3, "request_id": request_id}

    async def unset_prefix_lock_for_request(self, request_id):
        self.unset_calls.append(request_id)
        if self.fail:
            return {"unlocked": 0, "error": "simulated"}
        return {"unlocked": 3, "total_blocks": 3, "request_id": request_id}

    async def get_prefix_lock_stats(self):
        self.stats_polls += 1
        return {
            "locked_blocks": 3 * len(self.set_calls) - 3 * len(self.unset_calls),
            "lock_events_total": 3 * len(self.set_calls),
            "unlock_events_total": 3 * len(self.unset_calls),
            "evictions_skipped_due_to_lock": 7,
            "locked_eviction_forced": 0,
            "ttl_expired_events_total": 1,
        }


class StubVLLMServerMissing:
    """Server without the prefix-lock methods (patch not applied)."""

    async def ping(self):
        return "ok"


def _cfg(mode: str = "b1", **kw) -> AdmissionConfig:
    defaults = dict(
        enabled=True,
        mode=mode,
        kv_pressure_threshold=0.85,
        keepalive_interval_sec=60,
        probe_interval_sec=5,
        abort_mode="observe",
        prefix_lock_ttl_sec=10.0,
        prefix_lock_refresh_sec=0.1,
        prefix_lock_top_k=1,
    )
    defaults.update(kw)
    return AdmissionConfig(**defaults)


def _tree(uid, server="s0", active_rids=None, completed=None, active=None) -> TreeValueSnapshot:
    active_rids = list(active_rids or [])
    active_cnt = active if active is not None else len(active_rids)
    return TreeValueSnapshot(
        tree_uid=uid,
        complete_siblings=len(completed or []),
        active_siblings=active_cnt,
        kv_blocks_approx=max(1, active_cnt),
        primary_server=server,
        active_request_ids=active_rids,
        completed_request_ids=list(completed or []),
    )


def _coordinator(cfg, trees=None, servers=None):
    lb = LocalHandle(StubLB(trees)) if trees is not None else None
    handles = {sid: LocalHandle(s) for sid, s in (servers or {}).items()}
    return AdmissionCoordinatorImpl(cfg=cfg, lb_handle=lb, server_handles=handles)


def _tick(coord, usage=0.01):
    return asyncio.run(coord.tick(KVPressureSnapshot(
        gpu_cache_usage_perc=usage, num_preempted_total=0, num_running_seqs=1,
        num_pending_seqs=0, timestamp_s=time.time())))


class TestB1Gating:
    def test_mode_a_never_dispatches_prefix_lock(self):
        srv = StubVLLMServer()
        coord = _coordinator(_cfg(mode="a"), [_tree("t1", active_rids=["r0", "r1"])], {"s0": srv})
        _tick(coord)
        assert srv.set_calls == []

    def test_mode_b1_dispatches_when_active_rids_exist(self):
        cfg = _cfg()
        srv = StubVLLMServer()
        coord = _coordinator(cfg, [_tree("t1", active_rids=["r0", "r1"])], {"s0": srv})
        _tick(coord)
        assert len(srv.set_calls) == 1
        # top_k=1: the most recent in-flight request
        assert srv.set_calls[0][0] == "r1"
        assert srv.set_calls[0][1] == pytest.approx(cfg.prefix_lock_ttl_sec)

    def test_tree_without_active_rids_skipped(self):
        srv = StubVLLMServer()
        coord = _coordinator(_cfg(), [_tree("t1", active_rids=[], completed=["rX", "rY"], active=0)],
                             {"s0": srv})
        _tick(coord)
        assert srv.set_calls == []

    def test_primary_server_missing_skipped(self):
        srv = StubVLLMServer()
        coord = _coordinator(_cfg(), [_tree("t1", server="", active_rids=["r0"])], {"s0": srv})
        _tick(coord)
        assert srv.set_calls == []


class TestB1RateLimit:
    def test_rate_limit_per_tree(self):
        srv = StubVLLMServer()
        coord = _coordinator(_cfg(prefix_lock_refresh_sec=5.0), [_tree("t1", active_rids=["r0"])], {"s0": srv})
        _tick(coord)
        _tick(coord)
        assert len(srv.set_calls) == 1


class TestB1CountersAndErrors:
    def test_lock_error_counter_increments(self):
        srv = StubVLLMServer(fail=True)
        coord = _coordinator(_cfg(), [_tree("t1", active_rids=["r0"])], {"s0": srv})
        metrics = _tick(coord)
        assert metrics["admission/prefix_lock_attempts"] >= 1
        assert metrics["admission/prefix_lock_errors"] >= 1
        assert metrics["admission/prefix_lock_succeeded"] == 0

    def test_lock_stats_are_mirrored(self):
        srv = StubVLLMServer()
        coord = _coordinator(_cfg(), [_tree("t1", active_rids=["r0"])], {"s0": srv})
        metrics = _tick(coord)
        assert metrics["admission/prefix_lock_succeeded"] == 1
        assert metrics["admission/prefix_lock_blocks_total"] == 3
        assert metrics["admission/prefix_lock_evictions_skipped"] == 7
        assert srv.stats_polls == 1

    def test_counter_keys_in_get_metrics(self):
        metrics = _coordinator(_cfg()).get_metrics()
        for key in (
            "admission/prefix_lock_attempts",
            "admission/prefix_lock_succeeded",
            "admission/prefix_lock_errors",
            "admission/prefix_lock_blocks_total",
            "admission/prefix_unlock_attempts",
            "admission/prefix_unlock_succeeded",
            "admission/prefix_unlock_errors",
            "admission/prefix_lock_active_blocks",
            "admission/prefix_lock_evictions_skipped",
            "admission/prefix_lock_eviction_forced",
            "admission/prefix_lock_ttl_expired_events",
            "admission/b1_degraded_to_a",
        ):
            assert key in metrics, f"missing {key}"


class TestB1UnlockOnComplete:
    def test_dispatch_prefix_unlock_after_lock(self):
        srv = StubVLLMServer()
        coord = _coordinator(_cfg(), [_tree("t1", active_rids=["r0", "r1"])], {"s0": srv})
        _tick(coord)  # locks r1 (top_k=1) and remembers all in-flight ids
        asyncio.run(coord.dispatch_prefix_unlock_on_complete("t1"))
        # every remembered id is unlocked (unlocking an unlocked id is a no-op in vLLM)
        assert srv.unset_calls == ["r0", "r1"]

    def test_unknown_tree_no_raise(self):
        coord = _coordinator(_cfg())
        asyncio.run(coord.dispatch_prefix_unlock_on_complete("never-seen"))


class TestB1DegradeOnPatchFailure:
    def test_degrade_when_server_missing_method(self):
        coord = _coordinator(_cfg(), [_tree("t1", active_rids=["r0"])], {"s0": StubVLLMServerMissing()})
        metrics = _tick(coord)
        assert metrics["admission/b1_degraded_to_a"] == 1


class TestB1TopK:
    def test_top_k_locks_last_k_request_ids(self):
        srv = StubVLLMServer()
        coord = _coordinator(_cfg(prefix_lock_top_k=3), [_tree("t1", active_rids=["r0", "r1", "r2", "r3"])],
                             {"s0": srv})
        _tick(coord)
        assert [c[0] for c in srv.set_calls] == ["r1", "r2", "r3"]
