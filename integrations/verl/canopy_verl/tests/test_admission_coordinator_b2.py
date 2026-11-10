# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Admission coordinator, mode b2 (prefix-preserving partial preemption).

Config parsing and validation, plus the coordinator logic driven in-process
with stub balancer and vLLM servers.
"""

from __future__ import annotations

import asyncio
import time

from canopy_verl.scheduler.admission import (
    AdmissionConfig,
    AdmissionCoordinatorImpl,
    KVPressureSnapshot,
    TreeValueSnapshot,
)
from canopy_verl.tests._fakes import LocalHandle


B2_METRIC_KEYS = [
    "admission/partial_preempt_attempts",
    "admission/partial_preempt_succeeded",
    "admission/partial_preempt_errors",
    "admission/partial_preempt_kept_blocks_total",
    "admission/partial_preempt_released_blocks_total",
    "admission/partial_preempt_fallback_to_full",
    "admission/partial_preempt_ttl_expired_at_preempt",
    "admission/partial_preempt_skip_due_to_no_common_prefix",
    "admission/partial_preempt_coordinator_unsync_warnings",
    "admission/partial_preempt_preempt_invocations",
    "admission/partial_preempt_preempt_partial_path_taken",
    "admission/partial_preempt_tags_set",
    "admission/partial_preempt_tags_unset",
    "admission/b2_degraded_to_a",
]


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


def test_admission_config_b2_validates():
    assert AdmissionConfig(enabled=True, mode="b2").validate() == []


def test_admission_config_b2_invalid_threshold():
    cfg = AdmissionConfig(enabled=True, mode="b2", partial_preempt_pressure_threshold=1.5)
    assert any("partial_preempt_pressure_threshold" in e for e in cfg.validate())


def test_admission_config_b2_invalid_consecutive():
    cfg = AdmissionConfig(enabled=True, mode="b2", partial_preempt_pressure_consecutive=0)
    assert any("partial_preempt_pressure_consecutive" in e for e in cfg.validate())


def test_admission_config_b2_safety_margin_negative():
    cfg = AdmissionConfig(enabled=True, mode="b2", partial_preempt_safety_margin_blocks=-1)
    assert any("partial_preempt_safety_margin_blocks" in e for e in cfg.validate())


def test_admission_config_b2_from_hydra_round_trip():
    class _Section:
        def __init__(self, **kw):
            self._d = kw

        def get(self, k, d):
            return self._d.get(k, d)

    class _Root:
        def __init__(self, admission):
            self.admission = admission

    cfg = AdmissionConfig.from_hydra(_Root(_Section(
        enabled=True,
        mode="b2",
        partial_preempt_keep_n_blocks=12,
        partial_preempt_ttl_sec=120.0,
        partial_preempt_refresh_sec=2.0,
        partial_preempt_top_k=5,
        partial_preempt_pressure_threshold=0.7,
        partial_preempt_pressure_consecutive=4,
        partial_preempt_safety_margin_blocks=2,
    )))
    assert cfg.mode == "b2"
    assert cfg.partial_preempt_keep_n_blocks == 12
    assert cfg.partial_preempt_ttl_sec == 120.0
    assert cfg.partial_preempt_refresh_sec == 2.0
    assert cfg.partial_preempt_top_k == 5
    assert cfg.partial_preempt_pressure_threshold == 0.7
    assert cfg.partial_preempt_pressure_consecutive == 4
    assert cfg.partial_preempt_safety_margin_blocks == 2


# ---------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------


class StubLB:
    def __init__(self, trees=()):
        self.trees = list(trees)

    def get_active_trees_snapshot(self):
        return list(self.trees)


class StubB1Server:
    """vLLMHttpServer stand-in with keepalive and prefix-lock RPCs."""

    def __init__(self):
        self.generate_calls = 0
        self.lock_calls: list[str] = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **_):
        self.generate_calls += 1
        return {"request_id": request_id, "token_ids": [0]}

    async def set_prefix_lock_for_request(self, request_id, ttl_sec):
        self.lock_calls.append(request_id)
        return {"locked": 2, "total_blocks": 2, "request_id": request_id}

    async def get_prefix_lock_stats(self):
        return {"locked_blocks": 2 * len(self.lock_calls)}


class StubB2Server(StubB1Server):
    """Adds the partial-preemption RPCs; records tag requests."""

    def __init__(self, fail=None, stats=None):
        super().__init__()
        self.fail = fail  # None | "raise" | "error"
        self.tag_calls: list[tuple[str, int, float]] = []
        self.stats = dict(stats or {})

    async def set_partial_preempt_for_request(self, request_id, keep_n_blocks, ttl_sec=60.0):
        self.tag_calls.append((request_id, keep_n_blocks, ttl_sec))
        if self.fail == "raise":
            raise RuntimeError("simulated RPC failure")
        if self.fail == "error":
            return {"tagged": 0, "error": "simulated"}
        return {"tagged": 1, "request_id": request_id}

    async def get_partial_preempt_stats(self):
        return dict(self.stats)


def _cfg(**kw) -> AdmissionConfig:
    defaults = dict(
        enabled=True,
        mode="b2",
        kv_pressure_threshold=0.85,
        keepalive_interval_sec=60,
        partial_preempt_keep_n_blocks=8,
        partial_preempt_ttl_sec=30.0,
        partial_preempt_refresh_sec=5.0,
        partial_preempt_top_k=3,
        partial_preempt_pressure_threshold=0.80,
        partial_preempt_pressure_consecutive=2,
    )
    defaults.update(kw)
    return AdmissionConfig(**defaults)


def _tree(uid="t1", rids=("r0", "r1"), server="s0") -> TreeValueSnapshot:
    return TreeValueSnapshot(tree_uid=uid, active_siblings=len(rids), primary_server=server,
                             active_request_ids=list(rids))


def _coordinator(cfg, trees, server):
    return AdmissionCoordinatorImpl(cfg=cfg, lb_handle=LocalHandle(StubLB(trees)),
                                    server_handles={"s0": LocalHandle(server)})


def _tick(coord, usage):
    return asyncio.run(coord.tick(KVPressureSnapshot(gpu_cache_usage_perc=usage, timestamp_s=time.time())))


def test_admission_coordinator_init_b2():
    coord = AdmissionCoordinatorImpl(cfg=_cfg())
    assert coord.ping() == "ok"
    metrics = coord.get_metrics()
    assert all(metrics[k] == 0 for k in B2_METRIC_KEYS)


def test_tick_dispatch_partial_preempt_under_pressure():
    srv = StubB2Server()
    coord = _coordinator(_cfg(), [_tree()], srv)
    _tick(coord, 0.95)
    m = _tick(coord, 0.95)
    assert srv.tag_calls == [("r0", 8, 30.0), ("r1", 8, 30.0)]
    assert m["admission/partial_preempt_attempts"] == 2
    assert m["admission/partial_preempt_succeeded"] == 2
    # b2 includes the b1 prefix locks
    assert srv.lock_calls == ["r1"]


def test_tick_no_dispatch_below_threshold():
    srv = StubB2Server()
    coord = _coordinator(_cfg(), [_tree()], srv)
    for _ in range(3):
        _tick(coord, 0.5)
    assert srv.tag_calls == []


def test_consecutive_pressure_required():
    srv = StubB2Server()
    coord = _coordinator(_cfg(partial_preempt_pressure_consecutive=2), [_tree()], srv)
    _tick(coord, 0.95)
    assert srv.tag_calls == []
    _tick(coord, 0.95)
    assert len(srv.tag_calls) == 2


def test_dispatch_per_tree_rate_limit():
    srv = StubB2Server()
    coord = _coordinator(_cfg(partial_preempt_pressure_consecutive=1, partial_preempt_refresh_sec=5.0),
                         [_tree()], srv)
    _tick(coord, 0.95)
    _tick(coord, 0.95)
    assert len(srv.tag_calls) == 2  # second tick is within refresh_sec of the first


def test_dispatch_top_k_takes_last_k():
    srv = StubB2Server()
    coord = _coordinator(_cfg(partial_preempt_pressure_consecutive=1, partial_preempt_top_k=3),
                         [_tree(rids=("r0", "r1", "r2", "r3", "r4"))], srv)
    _tick(coord, 0.95)
    assert [c[0] for c in srv.tag_calls] == ["r2", "r3", "r4"]


def test_rpc_error_increments_b2_errors():
    for fail in ("raise", "error"):
        srv = StubB2Server(fail=fail)
        coord = _coordinator(_cfg(partial_preempt_pressure_consecutive=1), [_tree()], srv)
        m = _tick(coord, 0.95)
        assert m["admission/partial_preempt_errors"] == 2, fail
        assert m["admission/partial_preempt_succeeded"] == 0, fail
        assert m["admission/b2_degraded_to_a"] == 0, fail


def test_attribute_error_degrades_b2_to_a():
    srv = StubB1Server()  # no partial-preemption RPCs
    coord = _coordinator(_cfg(partial_preempt_pressure_consecutive=1), [_tree()], srv)
    m = _tick(coord, 0.95)
    assert m["admission/b2_degraded_to_a"] == 1
    attempts, locks = m["admission/partial_preempt_attempts"], len(srv.lock_calls)
    m = _tick(coord, 0.95)
    assert m["admission/partial_preempt_attempts"] == attempts
    assert len(srv.lock_calls) == locks


def test_poll_partial_preempt_stats_mirrors_counters():
    stats = {"tags_set": 5, "tags_unset": 1, "preempt_invocations": 4, "preempt_partial_path_taken": 3,
             "kept_blocks_total": 24, "released_blocks_total": 10, "fallback_to_full": 1,
             "ttl_expired_at_preempt": 0, "skip_due_to_no_common_prefix": 1, "coordinator_unsync_warnings": 3}
    coord = _coordinator(_cfg(), [_tree()], StubB2Server(stats=stats))
    m = _tick(coord, 0.1)
    assert m["admission/partial_preempt_tags_set"] == 5
    assert m["admission/partial_preempt_tags_unset"] == 1
    assert m["admission/partial_preempt_preempt_invocations"] == 4
    assert m["admission/partial_preempt_preempt_partial_path_taken"] == 3
    assert m["admission/partial_preempt_kept_blocks_total"] == 24
    assert m["admission/partial_preempt_released_blocks_total"] == 10
    assert m["admission/partial_preempt_fallback_to_full"] == 1
    assert m["admission/partial_preempt_skip_due_to_no_common_prefix"] == 1
    assert m["admission/partial_preempt_coordinator_unsync_warnings"] == 3


def test_get_metrics_includes_b2_counters():
    metrics = AdmissionCoordinatorImpl(cfg=_cfg()).get_metrics()
    missing = [k for k in B2_METRIC_KEYS if k not in metrics]
    assert not missing, missing
