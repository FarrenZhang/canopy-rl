# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""TreeAwareLoadBalancer routing and metrics (Ray actor, CPU only).

Covers: a new tree goes to the least-loaded server; siblings follow the tree's
server; soft spill on overload; unregistering a finished tree; concurrent
registration; and verl-identical routing when tree-sticky routing is off.
"""

from __future__ import annotations

import time

import pytest

ray = pytest.importorskip("ray")

from canopy_verl.scheduler.placement import TreeAwareLoadBalancer  # noqa: E402


@pytest.fixture(scope="module")
def ray_for_lb():
    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False)
    yield
    ray.shutdown()


class TestNewTreeGoesToLeastInflight:
    def test_new_tree_picks_least_inflight(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0", "s1", "s2"], enable_tree_sticky=True
        )
        # Two in-flight requests on s0 through the request-id path.
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid=None))
        ray.get(lb.acquire_server.remote(request_id="r0", tree_uid=None))
        server = ray.get(
            lb.acquire_server.remote(request_id="rT0", tree_uid="tree-A")
        )
        assert server in ("s1", "s2"), f"new tree went to overloaded s0 instead of {server}"

        metrics = ray.get(lb.get_metrics.remote())
        assert metrics["placement/tree_registrations"] == 1
        assert metrics["placement/tree_sticky_hits"] == 0


class TestSiblingFollowsParent:
    def test_siblings_co_locate(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0", "s1", "s2"], enable_tree_sticky=True
        )
        parent = ray.get(
            lb.acquire_server.remote(request_id="r_parent", tree_uid="tree-B")
        )
        sibs = [
            ray.get(
                lb.acquire_server.remote(request_id=f"r_sib_{i}", tree_uid="tree-B")
            )
            for i in range(5)
        ]
        assert all(s == parent for s in sibs), (
            f"not all siblings co-located: parent={parent} siblings={sibs}"
        )

        metrics = ray.get(lb.get_metrics.remote())
        assert metrics["placement/tree_sticky_hits"] == 5
        assert metrics["placement/tree_registrations"] == 1
        # 5 / (5 hits + 0 spills + 1 registration)
        assert metrics["placement/siblings_co_located_ratio"] > 0.8


class TestSpillOnOverload:
    def test_spill_when_preferred_overloaded(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0", "s1"],
            enable_tree_sticky=True,
            sticky_threshold=0.75,
            abs_spill_gap=3,
        )
        for i in range(8):
            ray.get(
                lb.acquire_server.remote(
                    request_id=f"rc_sib_{i}", tree_uid="tree-C"
                )
            )
        metrics = ray.get(lb.get_metrics.remote())
        # Siblings stay on the primary until its lead over s1 exceeds abs_spill_gap=3.
        assert metrics["placement/spill_events"] >= 1, (
            f"expected spill after 8 siblings with abs_spill_gap=3, "
            f"got metrics={metrics}"
        )
        assert metrics["placement/tree_registrations"] == 1
        assert (
            metrics["placement/tree_sticky_hits"]
            + metrics["placement/spill_events"]
            == 7
        )


class TestUnregisterAfterComplete:
    def test_unregister_releases_mapping(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0", "s1", "s2"], enable_tree_sticky=True
        )
        ray.get(lb.acquire_server.remote(request_id="rd0", tree_uid="tree-D"))
        ray.get(lb.release_server.remote(server_id="s0"))
        metrics_before = ray.get(lb.get_metrics.remote())
        assert metrics_before["placement/active_trees"] == 1

        ray.get(lb.on_tree_complete.remote(tree_uid="tree-D"))
        metrics_after = ray.get(lb.get_metrics.remote())
        assert metrics_after["placement/active_trees"] == 0
        assert metrics_after["placement/tree_completes"] == 1


class TestConcurrentRegister:
    def test_concurrent_same_tree_single_registration(self, ray_for_lb):
        """Concurrent acquires of one tree are serialized by the actor: one registration."""
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0", "s1", "s2"], enable_tree_sticky=True
        )
        futures = [
            lb.acquire_server.remote(request_id=f"rc_{i}", tree_uid="tree-E")
            for i in range(10)
        ]
        servers = ray.get(futures)
        assert len(set(servers)) == 1, (
            f"concurrent tree-E acquires split across servers: {set(servers)}"
        )

        metrics = ray.get(lb.get_metrics.remote())
        assert metrics["placement/tree_registrations"] == 1
        assert metrics["placement/tree_sticky_hits"] == 9


class TestSiblingsCoLocatedRatioStrict:
    def test_strict_reaches_one_without_spills(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0", "s1", "s2"], enable_tree_sticky=True
        )
        ray.get(lb.acquire_server.remote(request_id="r_reg", tree_uid="tree-F"))
        for i in range(5):
            ray.get(
                lb.acquire_server.remote(request_id=f"r_hit_{i}", tree_uid="tree-F")
            )
        metrics = ray.get(lb.get_metrics.remote())
        assert 0.8 < metrics["placement/siblings_co_located_ratio"] < 0.9
        assert metrics["placement/siblings_co_located_ratio_strict"] == 1.0
        assert metrics["placement/spill_events"] == 0

    def test_strict_zero_when_no_opportunities(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0"], enable_tree_sticky=True
        )
        ray.get(lb.acquire_server.remote(request_id="r_only", tree_uid="tree-G"))
        metrics = ray.get(lb.get_metrics.remote())
        assert metrics["placement/tree_sticky_hits"] == 0
        assert metrics["placement/spill_events"] == 0
        assert metrics["placement/siblings_co_located_ratio_strict"] == 0.0


class TestDisabledFallback:
    """With enable_tree_sticky=False routing matches verl's GlobalRequestLoadBalancer."""

    def test_disabled_per_request_sticky(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0", "s1", "s2"], enable_tree_sticky=False
        )
        s0 = ray.get(lb.acquire_server.remote(request_id="r-abc", tree_uid="tree-X"))
        # tree_uid is ignored; the request id stays sticky.
        s1 = ray.get(lb.acquire_server.remote(request_id="r-abc", tree_uid="tree-Y"))
        assert s0 == s1, "per-request_id sticky broken when tree_sticky disabled"

        metrics = ray.get(lb.get_metrics.remote())
        assert metrics["placement/tree_registrations"] == 0
        assert metrics["placement/tree_sticky_hits"] == 0
        assert metrics["placement/request_sticky_hits"] == 1
        assert metrics["placement/new_request_routes"] == 1

    def test_release_server_signature_identical(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0", "s1"], enable_tree_sticky=False
        )
        s = ray.get(lb.acquire_server.remote(request_id="r-rel", tree_uid=None))
        ray.get(lb.release_server.remote(server_id=s))
        with pytest.raises(ray.exceptions.RayTaskError, match="no inflight"):
            ray.get(lb.release_server.remote(server_id=s))
        with pytest.raises(ray.exceptions.RayTaskError, match="Invalid server_id"):
            ray.get(lb.release_server.remote(server_id="nonexistent"))


class TestInflightRequestTracking:
    """Per-tree in-flight / completed request ids used by the admission coordinator."""

    def test_snapshot_has_primary_server_and_completed_reqs(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0", "s1"],
            completed_request_ids_cap=4,
        )
        s = ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="t-A"))
        ray.get(
            lb.release_server.remote(
                server_id=s, request_id="vllm-rid-0", tree_uid="t-A"
            )
        )
        ray.get(lb.acquire_server.remote(request_id="r1", tree_uid="t-A"))

        snaps = ray.get(lb.get_active_trees_snapshot.remote())
        assert len(snaps) == 1
        snap = snaps[0]
        assert snap.tree_uid == "t-A"
        assert snap.primary_server in ("s0", "s1")
        assert snap.completed_request_ids == ["vllm-rid-0"]
        # The new sibling has not registered a vLLM request id yet.
        assert snap.active_request_ids == []

    def test_register_active_request_populates_active_rids(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0"],
            completed_request_ids_cap=4,
        )
        s = ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="t-B"))
        ray.get(
            lb.register_active_request.remote(
                server_id=s, request_id="vllm-rid-active", tree_uid="t-B"
            )
        )
        snaps = ray.get(lb.get_active_trees_snapshot.remote())
        assert snaps[0].active_request_ids == ["vllm-rid-active"]
        ray.get(
            lb.release_server.remote(
                server_id=s, request_id="vllm-rid-active", tree_uid="t-B"
            )
        )
        ray.get(lb.acquire_server.remote(request_id="r1", tree_uid="t-B"))
        snaps = ray.get(lb.get_active_trees_snapshot.remote())
        assert snaps[0].active_request_ids == []
        assert snaps[0].completed_request_ids == ["vllm-rid-active"]

    def test_release_server_backward_compat_without_kwargs(self, ray_for_lb):
        """``release_server(server_id=s)`` alone does not touch the completed list."""
        lb = TreeAwareLoadBalancer.remote(server_actor_ids=["s0", "s1"])
        s = ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="t-B"))
        ray.get(lb.release_server.remote(server_id=s))
        snaps = ray.get(lb.get_active_trees_snapshot.remote())
        assert snaps[0].completed_request_ids == []

    def test_completed_request_ids_ring_buffer_cap(self, ray_for_lb):
        lb = TreeAwareLoadBalancer.remote(
            server_actor_ids=["s0"], completed_request_ids_cap=2
        )
        for i in range(4):
            s = ray.get(
                lb.acquire_server.remote(request_id=f"r{i}", tree_uid="t-C")
            )
            ray.get(
                lb.release_server.remote(
                    server_id=s,
                    request_id=f"vllm-rid-{i}",
                    tree_uid="t-C",
                )
            )
        ray.get(lb.acquire_server.remote(request_id="r-final", tree_uid="t-C"))
        snaps = ray.get(lb.get_active_trees_snapshot.remote())
        assert snaps[0].completed_request_ids == ["vllm-rid-2", "vllm-rid-3"]

    def test_on_tree_complete_notifies_coordinator_handle(self, ray_for_lb):
        @ray.remote
        class _CoordMock:
            def __init__(self):
                self.called_with: list[str] = []

            def dispatch_prefix_unlock_on_complete(self, tree_uid):
                self.called_with.append(tree_uid)

            def get_calls(self):
                return list(self.called_with)

        lb = TreeAwareLoadBalancer.remote(server_actor_ids=["s0"])
        coord = _CoordMock.remote()
        ray.get(lb.set_admission_coordinator.remote(coord))
        s = ray.get(lb.acquire_server.remote(request_id="r0", tree_uid="t-D"))
        ray.get(lb.release_server.remote(server_id=s, request_id="v0", tree_uid="t-D"))
        ray.get(lb.on_tree_complete.remote("t-D"))
        calls = []
        for _ in range(10):  # the notification is fire-and-forget
            calls = ray.get(coord.get_calls.remote())
            if calls:
                break
            time.sleep(0.05)
        assert calls == ["t-D"]
