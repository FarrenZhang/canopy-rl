# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""AdmissionCoordinator as a Ray actor with the real TreeAwareLoadBalancer (tier a, CPU only).

The vLLM server is a small Ray actor that records calls.  Covers: disabled
ticks, metric keys, no dispatch without pressure, keepalive under pressure,
pressure from new preemptions, observe-only release decisions, cascade abort,
failure handling, keepalive rate limiting, and the balancer's tree snapshot.
"""

from __future__ import annotations

from typing import Any

import pytest

ray = pytest.importorskip("ray")

from canopy_verl.scheduler.admission import (  # noqa: E402
    AdmissionConfig,
    AdmissionCoordinator,
    KVPressureSnapshot,
    TreeValueSnapshot,
)
from canopy_verl.scheduler.placement import TreeAwareLoadBalancer  # noqa: E402


@ray.remote
class _FakeVLLMServer:
    """Records keepalive generate calls and abort_all_requests calls."""

    def __init__(self, fail_mode: str = "none") -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.abort_all_calls: list[dict[str, Any]] = []
        self.fail_mode = fail_mode  # "none" | "generate" | "abort"

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        **_ignored: Any,
    ) -> dict[str, Any]:
        self.generate_calls.append(
            dict(request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params)
        )
        if self.fail_mode == "generate":
            raise RuntimeError("injected keepalive failure")
        return {"request_id": request_id, "token_ids": [0]}

    async def abort_all_requests(
        self, reset_prefix_cache: bool = True
    ) -> dict[str, Any]:
        self.abort_all_calls.append(dict(reset_prefix_cache=reset_prefix_cache))
        if self.fail_mode == "abort":
            return {"aborted_count": 0, "request_ids": [], "error": "injected abort failure"}
        return {"aborted_count": 1, "request_ids": ["req_abc"]}

    def stats(self) -> dict[str, Any]:
        return {
            "generate_calls": len(self.generate_calls),
            "abort_all_calls": len(self.abort_all_calls),
            "abort_reset_flags": [c["reset_prefix_cache"] for c in self.abort_all_calls],
        }


@pytest.fixture(scope="module")
def ray_for_admission():
    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)
    yield
    ray.shutdown()


def _make_lb(server_ids: list[str]) -> Any:
    return TreeAwareLoadBalancer.remote(
        server_actor_ids=server_ids, enable_tree_sticky=True
    )


def _make_coordinator(cfg: AdmissionConfig, lb_handle: Any, server_handles: dict[str, Any]) -> Any:
    return AdmissionCoordinator.remote(
        cfg=cfg,
        lb_handle=lb_handle,
        server_handles=server_handles,
        vllm_dummy_prompt_ids=[1, 2, 3],
    )


class TestCoordinatorDisabled:
    def test_disabled_tick_returns_metrics_without_dispatch(self, ray_for_admission):
        cfg = AdmissionConfig(enabled=False)
        lb = _make_lb(["s0"])
        fake = _FakeVLLMServer.remote()
        coord = _make_coordinator(cfg, lb, {"s0": fake})

        metrics = ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.99)))

        assert metrics["admission/total_ticks"] == 1
        assert metrics["admission/pressure_ticks"] == 0
        assert metrics["admission/keepalive_dispatched"] == 0
        assert metrics["admission/aborts_dispatched"] == 0
        stats = ray.get(fake.stats.remote())
        assert stats["generate_calls"] == 0
        assert stats["abort_all_calls"] == 0


class TestCoordinatorMetricsShape:
    def test_metrics_keys_present(self, ray_for_admission):
        cfg = AdmissionConfig(enabled=True, kv_pressure_threshold=0.5)
        coord = _make_coordinator(cfg, _make_lb(["s0"]), {"s0": _FakeVLLMServer.remote()})

        metrics = ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.1)))

        expected = {
            "admission/total_ticks",
            "admission/pressure_ticks",
            "admission/pressure_ratio",
            "admission/keepalive_dispatched",
            "admission/aborts_dispatched",
            "admission/keepalive_calls_attempted",
            "admission/keepalive_calls_succeeded",
            "admission/keepalive_errors",
            "admission/cascade_aborts_attempted",
            "admission/cascade_aborts_succeeded",
            "admission/abort_errors",
            "admission/abort_decisions_observe_only",
            "admission/tick_errors",
            "admission/lb_query_errors",
            "admission/consecutive_pressure_ticks",
        }
        assert expected.issubset(metrics.keys()), (
            f"missing keys: {expected - metrics.keys()}"
        )


class TestNoPressureNoDispatch:
    def test_no_pressure_no_dispatch(self, ray_for_admission):
        cfg = AdmissionConfig(enabled=True, kv_pressure_threshold=0.85)
        fake = _FakeVLLMServer.remote()
        coord = _make_coordinator(cfg, _make_lb(["s0"]), {"s0": fake})

        ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.3)))
        metrics = ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.4)))

        assert metrics["admission/pressure_ticks"] == 0
        stats = ray.get(fake.stats.remote())
        assert stats["generate_calls"] == 0
        assert stats["abort_all_calls"] == 0


class TestUnderPressureDispatchesKeepalive:
    def test_pressure_dispatches_keepalive_when_active_tree_exists(self, ray_for_admission):
        cfg = AdmissionConfig(
            enabled=True, kv_pressure_threshold=0.5, keepalive_interval_sec=1
        )
        lb = _make_lb(["s0"])
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="tree-A"))
        fake = _FakeVLLMServer.remote()
        coord = _make_coordinator(cfg, lb, {"s0": fake})

        metrics = ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.9)))

        assert metrics["admission/pressure_ticks"] == 1
        assert metrics["admission/keepalive_dispatched"] == 1
        assert metrics["admission/keepalive_calls_attempted"] >= 1
        assert metrics["admission/keepalive_calls_succeeded"] >= 1
        stats = ray.get(fake.stats.remote())
        assert stats["generate_calls"] >= 1
        # the default abort_mode "observe" never calls vLLM
        assert stats["abort_all_calls"] == 0
        assert metrics["admission/abort_decisions_observe_only"] == 1


class TestPreemptDeltaTriggersPressure:
    def test_preempt_delta_only_path(self, ray_for_admission):
        cfg = AdmissionConfig(
            enabled=True,
            kv_pressure_threshold=0.99,
            keepalive_interval_sec=1,
        )
        lb = _make_lb(["s0"])
        ray.get(lb.acquire_server.remote(request_id="rp", tree_uid="tree-P"))
        coord = _make_coordinator(cfg, lb, {"s0": _FakeVLLMServer.remote()})

        ray.get(coord.tick.remote(KVPressureSnapshot(num_preempted_total=10)))
        m = ray.get(coord.tick.remote(KVPressureSnapshot(num_preempted_total=15)))

        assert m["admission/pressure_ticks"] == 1
        assert m["admission/keepalive_dispatched"] == 1


class TestAbortObserveMode:
    def test_observe_mode_does_not_call_vllm_abort(self, ray_for_admission):
        cfg = AdmissionConfig(
            enabled=True,
            kv_pressure_threshold=0.5,
            keepalive_interval_sec=1,
            abort_mode="observe",
        )
        lb = _make_lb(["s0"])
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="tree-A"))
        ray.get(lb.release_server.remote(server_id="s0"))
        ray.get(lb.acquire_server.remote(request_id="r1", tree_uid="tree-A"))
        fake = _FakeVLLMServer.remote()
        coord = _make_coordinator(cfg, lb, {"s0": fake})

        metrics = ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.9)))
        stats = ray.get(fake.stats.remote())

        assert metrics["admission/aborts_dispatched"] == 1
        assert metrics["admission/abort_decisions_observe_only"] == 1
        assert stats["abort_all_calls"] == 0
        assert metrics["admission/cascade_aborts_attempted"] == 0


class TestAbortAllOnCascade:
    def test_cascade_dispatches_after_n_pressure_ticks(self, ray_for_admission):
        cfg = AdmissionConfig(
            enabled=True,
            kv_pressure_threshold=0.5,
            keepalive_interval_sec=999999,
            abort_mode="all_on_cascade",
            cascade_threshold=1,
            cascade_consecutive=2,
        )
        lb = _make_lb(["s0"])
        ray.get(lb.acquire_server.remote(request_id="rc", tree_uid="tree-C"))
        fake = _FakeVLLMServer.remote()
        coord = _make_coordinator(cfg, lb, {"s0": fake})

        # Tick 1 sets the preemption baseline; ticks 2 and 3 see new preemptions,
        # so the third tick has 2 consecutive pressure ticks with "preempt" in the reason.
        ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.9, num_preempted_total=5)))
        ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.9, num_preempted_total=8)))
        m = ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.9, num_preempted_total=9)))

        assert m["admission/cascade_aborts_attempted"] >= 1
        assert m["admission/cascade_aborts_succeeded"] >= 1
        stats = ray.get(fake.stats.remote())
        assert stats["abort_all_calls"] >= 1
        assert stats["abort_reset_flags"] == [False] * stats["abort_all_calls"]


class TestCascadeAllowUsageOnly:
    """``cascade_allow_usage_only`` decides whether a cascade abort needs new preemptions."""

    def test_usage_only_pressure_without_flag_does_not_cascade(self, ray_for_admission):
        cfg = AdmissionConfig(
            enabled=True,
            kv_pressure_threshold=0.5,
            keepalive_interval_sec=999999,
            abort_mode="all_on_cascade",
            cascade_threshold=1,
            cascade_consecutive=1,
            cascade_allow_usage_only=False,
        )
        lb = _make_lb(["s0"])
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="tree-UO"))
        fake = _FakeVLLMServer.remote()
        coord = _make_coordinator(cfg, lb, {"s0": fake})

        for _ in range(3):
            ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.95, num_preempted_total=0)))
        metrics = ray.get(coord.get_metrics.remote())
        assert metrics["admission/pressure_ticks"] == 3
        assert metrics["admission/cascade_aborts_attempted"] == 0
        assert metrics["admission/cascade_aborts_succeeded"] == 0
        assert ray.get(fake.stats.remote())["abort_all_calls"] == 0

    def test_usage_only_pressure_with_flag_does_cascade(self, ray_for_admission):
        cfg = AdmissionConfig(
            enabled=True,
            kv_pressure_threshold=0.5,
            keepalive_interval_sec=999999,
            abort_mode="all_on_cascade",
            cascade_threshold=1,
            cascade_consecutive=1,
            cascade_allow_usage_only=True,
        )
        lb = _make_lb(["s0"])
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="tree-UY"))
        fake = _FakeVLLMServer.remote()
        coord = _make_coordinator(cfg, lb, {"s0": fake})

        ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.95, num_preempted_total=0)))
        metrics = ray.get(coord.get_metrics.remote())
        assert metrics["admission/cascade_aborts_attempted"] >= 1
        assert metrics["admission/cascade_aborts_succeeded"] >= 1
        assert ray.get(fake.stats.remote())["abort_all_calls"] >= 1

    def test_cascade_cooldown_prevents_repeat_abort_within_60s(self, ray_for_admission):
        cfg = AdmissionConfig(
            enabled=True,
            kv_pressure_threshold=0.5,
            keepalive_interval_sec=999999,
            abort_mode="all_on_cascade",
            cascade_threshold=1,
            cascade_consecutive=1,
            cascade_allow_usage_only=True,
        )
        lb = _make_lb(["s0"])
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="tree-CD"))
        fake = _FakeVLLMServer.remote()
        coord = _make_coordinator(cfg, lb, {"s0": fake})

        ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.95)))
        ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.95)))
        metrics = ray.get(coord.get_metrics.remote())
        assert metrics["admission/cascade_aborts_attempted"] == 1
        assert ray.get(fake.stats.remote())["abort_all_calls"] == 1


class TestHttpFailureDoesNotKill:
    def test_keepalive_failure_counted(self, ray_for_admission):
        cfg = AdmissionConfig(
            enabled=True, kv_pressure_threshold=0.5, keepalive_interval_sec=1
        )
        lb = _make_lb(["s0"])
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="tree-F"))
        coord = _make_coordinator(cfg, lb, {"s0": _FakeVLLMServer.remote(fail_mode="generate")})

        m = ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.9)))
        assert m["admission/keepalive_errors"] >= 1
        m2 = ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.2)))
        assert m2["admission/total_ticks"] == 2


class TestRateLimitKeepalive:
    def test_rate_limit_within_interval(self, ray_for_admission):
        cfg = AdmissionConfig(
            enabled=True,
            kv_pressure_threshold=0.5,
            keepalive_interval_sec=999999,
        )
        lb = _make_lb(["s0"])
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="tree-R"))
        coord = _make_coordinator(cfg, lb, {"s0": _FakeVLLMServer.remote()})

        for _ in range(2):
            ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.9)))
        m = ray.get(coord.tick.remote(KVPressureSnapshot(gpu_cache_usage_perc=0.9)))

        assert m["admission/keepalive_calls_attempted"] == 1
        assert m["admission/keepalive_calls_succeeded"] == 1


class TestGetActiveTreesSnapshot:
    def test_snapshot_returns_list_of_tree_value_snapshots(self, ray_for_admission):
        lb = _make_lb(["s0", "s1"])
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="tree-X"))
        ray.get(lb.acquire_server.remote(request_id="r1", tree_uid="tree-X"))
        ray.get(lb.acquire_server.remote(request_id="r2", tree_uid="tree-Y"))

        snaps = ray.get(lb.get_active_trees_snapshot.remote())
        assert isinstance(snaps, list)
        assert all(isinstance(s, TreeValueSnapshot) for s in snaps)
        assert {s.tree_uid for s in snaps} == {"tree-X", "tree-Y"}
        for s in snaps:
            assert s.active_siblings >= 0
            assert s.complete_siblings >= 0
            assert s.kv_blocks_approx >= 0
