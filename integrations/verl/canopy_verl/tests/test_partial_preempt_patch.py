# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
#
# The stubs below mirror vLLM v0.13.0 (vllm/v1/core/kv_cache_utils.py,
# vllm/v1/core/block_pool.py, vllm/v1/core/sched/scheduler.py).
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Partial-preemption patch (canopy_verl.vllm_patch.partial_preempt) against stub vLLM classes.

Covers apply / revert, ``BlockPool.free_blocks_partial``,
``KVCacheManager.partial_free``, every fallback of the
``Scheduler._preempt_request`` wrap, the full partial path, and the case of
the last running request of a tree (no other request shares its prefix).
"""

from __future__ import annotations

import sys
import time
import types

import pytest


class StubKVCacheBlock:
    """Attributes of vLLM 0.13's KVCacheBlock used by the patch (plus block_id for tests)."""

    def __init__(self, block_id: int, ref_cnt: int = 1, is_null: bool = False):
        self.block_id = block_id
        self.ref_cnt = ref_cnt
        self.is_null = is_null
        self.next_free_block = None
        self.prev_free_block = None
        self._block_hash = None
        self.block_hash = None


class StubFreeQueue:
    def __init__(self):
        self.blocks_in_queue = []

    def append_n(self, blocks):
        self.blocks_in_queue.extend(blocks)

    def append(self, block):
        self.blocks_in_queue.append(block)

    def popleft(self):
        if not self.blocks_in_queue:
            raise IndexError("empty")
        return self.blocks_in_queue.pop(0)

    def get_all_free_blocks(self):
        return list(self.blocks_in_queue)

    def remove(self, block):
        if block in self.blocks_in_queue:
            self.blocks_in_queue.remove(block)


class StubBlockPool:
    """BlockPool with ``blocks`` (indexed by block id) and a free queue; block 0 is the null block."""

    def __init__(self, n_blocks: int = 32):
        self.blocks = [StubKVCacheBlock(i) for i in range(n_blocks)]
        self.blocks[0].is_null = True
        self.null_block = self.blocks[0]
        self.free_block_queue = StubFreeQueue()
        self.cached_block_hash_to_block = {}


class StubKVCacheManager:
    """KVCacheManager with block_pool, block_size, get_block_ids, get_num_common_prefix_blocks and free."""

    def __init__(self, block_pool: StubBlockPool, block_size: int = 16):
        self.block_pool = block_pool
        self.block_size = block_size
        self._request_blocks: dict[str, tuple[list[int], ...]] = {}
        self._common_prefix: dict[str, list[int]] = {}
        self.coordinator_free_calls: list[str] = []

        class _CoordSpy:
            def __init__(self, parent):
                self.parent = parent

            def free(self, rid):
                self.parent.coordinator_free_calls.append(rid)

        self.coordinator = _CoordSpy(self)

    def register_request_blocks(self, request_id, groups):
        self._request_blocks[request_id] = groups
        for group in groups:
            for bid in group:
                if 0 <= bid < len(self.block_pool.blocks):
                    self.block_pool.blocks[bid].ref_cnt = max(
                        1, self.block_pool.blocks[bid].ref_cnt
                    )

    def set_common_prefix(self, request_id, common_per_group):
        self._common_prefix[request_id] = common_per_group

    def get_block_ids(self, request_id):
        return self._request_blocks.get(request_id, ())

    def get_num_common_prefix_blocks(self, request_id):
        return self._common_prefix.get(request_id, [])

    def free(self, request):
        self.coordinator.free(request.request_id)


class StubRequest:
    def __init__(self, request_id, num_computed_tokens=0):
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens
        self.num_preemptions = 0


class StubWaitingQueue:
    def __init__(self):
        self.prepended = []

    def prepend_request(self, req):
        self.prepended.append(req)


class StubEncoderCacheManager:
    def __init__(self):
        self.freed_requests = []

    def free(self, req):
        self.freed_requests.append(req.request_id)


class StubScheduler:
    def __init__(self, kv_cache_manager: StubKVCacheManager, log_stats: bool = False):
        self.kv_cache_manager = kv_cache_manager
        self.waiting = StubWaitingQueue()
        self.encoder_cache_manager = StubEncoderCacheManager()
        self.log_stats = log_stats


@pytest.fixture
def fresh_counters():
    """Counters attached to StubScheduler (stand-in for vLLM's Scheduler class)."""
    from canopy_verl.vllm_patch.partial_preempt import _ensure_scheduler_counters

    if hasattr(StubScheduler, "_canopy_partial_preempt_counters"):
        delattr(StubScheduler, "_canopy_partial_preempt_counters")
    _ensure_scheduler_counters(StubScheduler)
    yield StubScheduler._canopy_partial_preempt_counters
    if hasattr(StubScheduler, "_canopy_partial_preempt_counters"):
        delattr(StubScheduler, "_canopy_partial_preempt_counters")


@pytest.fixture
def stub_world():
    bp = StubBlockPool(n_blocks=32)
    kvm = StubKVCacheManager(bp)
    sched = StubScheduler(kvm)
    return {"bp": bp, "kvm": kvm, "sched": sched}


# ---------------------------------------------------------------------
# apply / revert
# ---------------------------------------------------------------------


def test_apply_revert_idempotent_no_vllm():
    from canopy_verl.vllm_patch import (
        apply_partial_preempt_patch,
        applied_partial_preempt_patch,
        revert_partial_preempt_patch,
    )
    r1 = apply_partial_preempt_patch()
    r2 = apply_partial_preempt_patch()
    assert r1 == r2
    if applied_partial_preempt_patch():
        revert_partial_preempt_patch()
        assert not applied_partial_preempt_patch()


def test_apply_all_b2_returns_dict():
    from canopy_verl.vllm_patch import apply_all_b2_patches
    res = apply_all_b2_patches()
    assert set(res.keys()) == {
        "prefix_lock",
        "verl_server_extensions",
        "partial_preempt",
        "verl_server_extensions_b2",
    }
    for k, v in res.items():
        assert isinstance(v, bool), f"non-bool result for {k}: {v}"


def test_counter_dict_initialisation(fresh_counters):
    expected_keys = {
        "tags_set", "tags_unset", "preempt_invocations",
        "preempt_partial_path_taken", "succeeded", "fallback_to_full",
        "kept_blocks_total", "released_blocks_total",
        "skip_due_to_no_common_prefix", "skip_due_to_no_tag",
        "skip_due_to_ttl_expired", "skip_due_to_zero_keep_after_safety",
        "ttl_expired_at_preempt", "coordinator_unsync_warnings",
        "partial_free_exceptions", "block_overlap_with_b1_lock",
    }
    assert set(fresh_counters.keys()) == expected_keys
    assert all(isinstance(v, int) and v == 0 for v in fresh_counters.values())


def test_get_partial_preempt_state_serialisable(fresh_counters):
    from canopy_verl.vllm_patch.partial_preempt import _canopy_get_partial_preempt_state
    state = _canopy_get_partial_preempt_state(StubScheduler)
    assert isinstance(state, dict)
    for v in state.values():
        assert isinstance(v, (int, float, str)), f"non-serialisable counter: {v!r}"


# ---------------------------------------------------------------------
# BlockPool.free_blocks_partial
# ---------------------------------------------------------------------


def test_free_blocks_partial_decrements_ref_cnt(stub_world):
    from canopy_verl.vllm_patch.partial_preempt import _canopy_block_pool_free_blocks_partial
    bp = stub_world["bp"]
    for i in range(1, 5):
        bp.blocks[i].ref_cnt = 2
    _canopy_block_pool_free_blocks_partial(bp, [bp.blocks[1], bp.blocks[2], bp.blocks[3]])
    for i in range(1, 4):
        assert bp.blocks[i].ref_cnt == 1
    assert len(bp.free_block_queue.blocks_in_queue) == 0


def test_free_blocks_partial_zero_ref_enters_queue(stub_world):
    from canopy_verl.vllm_patch.partial_preempt import _canopy_block_pool_free_blocks_partial
    bp = stub_world["bp"]
    bp.blocks[10].ref_cnt = 1
    bp.blocks[11].ref_cnt = 1
    bp.blocks[12].ref_cnt = 3  # shared
    _canopy_block_pool_free_blocks_partial(bp, [bp.blocks[10], bp.blocks[11], bp.blocks[12]])
    assert bp.blocks[10] in bp.free_block_queue.blocks_in_queue
    assert bp.blocks[11] in bp.free_block_queue.blocks_in_queue
    assert bp.blocks[12] not in bp.free_block_queue.blocks_in_queue
    assert bp.blocks[12].ref_cnt == 2


def test_free_blocks_partial_skips_null(stub_world):
    from canopy_verl.vllm_patch.partial_preempt import _canopy_block_pool_free_blocks_partial
    bp = stub_world["bp"]
    bp.null_block.ref_cnt = 1
    _canopy_block_pool_free_blocks_partial(bp, [bp.null_block])
    assert bp.null_block not in bp.free_block_queue.blocks_in_queue


def test_free_blocks_partial_empty_no_op(stub_world):
    from canopy_verl.vllm_patch.partial_preempt import _canopy_block_pool_free_blocks_partial
    bp = stub_world["bp"]
    snapshot = [b.ref_cnt for b in bp.blocks]
    _canopy_block_pool_free_blocks_partial(bp, [])
    assert [b.ref_cnt for b in bp.blocks] == snapshot


# ---------------------------------------------------------------------
# KVCacheManager.partial_free
# ---------------------------------------------------------------------


def _with_partial_helpers(stub_world):
    from canopy_verl.vllm_patch import partial_preempt as pp_mod
    bp, kvm = stub_world["bp"], stub_world["kvm"]
    bp.free_blocks_partial = types.MethodType(pp_mod._canopy_block_pool_free_blocks_partial, bp)
    kvm.partial_free = types.MethodType(pp_mod._canopy_kv_cache_manager_partial_free, kvm)
    return bp, kvm


def test_partial_free_releases_suffix_only(stub_world, fresh_counters, monkeypatch):
    """One group of 10 blocks, keep 4: blocks 5..10 are released."""
    from canopy_verl.vllm_patch.partial_preempt import _canopy_kv_cache_manager_partial_free
    # partial_free counts on vLLM's Scheduler class when vLLM is installed
    sched_mod = types.ModuleType("vllm.v1.core.sched.scheduler")
    sched_mod.Scheduler = StubScheduler
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", sched_mod)
    bp, kvm = _with_partial_helpers(stub_world)
    req = StubRequest("req1")
    group = list(range(1, 11))
    kvm.register_request_blocks("req1", (group,))
    for bid in group:
        bp.blocks[bid].ref_cnt = 1

    result = _canopy_kv_cache_manager_partial_free(kvm, req, keep_n_blocks=4)
    assert result == {"released": 6, "kept": 4, "groups": 1}
    for bid in range(1, 5):
        assert bp.blocks[bid].ref_cnt == 1
    for bid in range(5, 11):
        assert bp.blocks[bid].ref_cnt == 0
        assert bp.blocks[bid] in bp.free_block_queue.blocks_in_queue
    assert fresh_counters["coordinator_unsync_warnings"] == 1


def test_partial_free_keep_n_exceeds_total_kept_all(stub_world, fresh_counters):
    from canopy_verl.vllm_patch.partial_preempt import _canopy_kv_cache_manager_partial_free
    bp, kvm = _with_partial_helpers(stub_world)
    req = StubRequest("req2")
    group = list(range(1, 9))
    kvm.register_request_blocks("req2", (group,))
    for bid in group:
        bp.blocks[bid].ref_cnt = 1

    result = _canopy_kv_cache_manager_partial_free(kvm, req, keep_n_blocks=20)
    assert result["released"] == 0
    assert result["kept"] == 8
    assert all(bp.blocks[bid].ref_cnt == 1 for bid in group)


def test_partial_free_multi_group(stub_world, fresh_counters):
    from canopy_verl.vllm_patch.partial_preempt import _canopy_kv_cache_manager_partial_free
    bp, kvm = _with_partial_helpers(stub_world)
    req = StubRequest("req3")
    g1 = [1, 2, 3, 4, 5, 6]
    g2 = [11, 12, 13, 14, 15, 16]
    kvm.register_request_blocks("req3", (g1, g2))
    for bid in g1 + g2:
        bp.blocks[bid].ref_cnt = 1

    result = _canopy_kv_cache_manager_partial_free(kvm, req, keep_n_blocks=2)
    assert result == {"released": 8, "kept": 4, "groups": 2}


# ---------------------------------------------------------------------
# Scheduler._preempt_request wrap
# ---------------------------------------------------------------------


@pytest.fixture
def orig_preempt(monkeypatch):
    """Replace vLLM's original _preempt_request with a spy that does a full reset."""
    from canopy_verl.vllm_patch import partial_preempt as pp_mod
    calls = []

    def _spy(self, request, timestamp):
        calls.append(request.request_id)
        request.num_computed_tokens = 0

    monkeypatch.setattr(pp_mod, "_ORIG_PREEMPT_REQUEST", _spy, raising=False)
    return pp_mod, calls


def test_preempt_wrap_no_tag_falls_back(stub_world, fresh_counters, orig_preempt):
    pp_mod, calls = orig_preempt
    req = StubRequest("noTag", num_computed_tokens=100)
    pp_mod._canopy_preempt_request_partial(stub_world["sched"], req, time.time())
    assert calls == ["noTag"]
    assert req.num_computed_tokens == 0
    assert fresh_counters["preempt_invocations"] == 1
    assert fresh_counters["skip_due_to_no_tag"] == 1
    assert fresh_counters["succeeded"] == 0


def test_preempt_wrap_ttl_expired_falls_back(stub_world, fresh_counters, orig_preempt):
    pp_mod, calls = orig_preempt
    req = StubRequest("expired")
    req._canopy_partial_preempt_keep_n = 4
    req._canopy_partial_preempt_deadline = time.time() - 100
    pp_mod._canopy_preempt_request_partial(stub_world["sched"], req, time.time())
    assert calls == ["expired"]
    assert fresh_counters["skip_due_to_ttl_expired"] == 1
    assert fresh_counters["ttl_expired_at_preempt"] == 1
    assert fresh_counters["succeeded"] == 0


def test_preempt_wrap_no_blocks_falls_back(stub_world, fresh_counters, orig_preempt):
    pp_mod, calls = orig_preempt
    req = StubRequest("noBlocks")
    req._canopy_partial_preempt_keep_n = 4
    req._canopy_partial_preempt_deadline = time.time() + 100
    pp_mod._canopy_preempt_request_partial(stub_world["sched"], req, time.time())
    assert calls == ["noBlocks"]
    assert fresh_counters["skip_due_to_no_common_prefix"] == 1
    assert fresh_counters["succeeded"] == 0


def test_preempt_wrap_zero_keep_after_safety_falls_back(stub_world, fresh_counters, orig_preempt):
    pp_mod, calls = orig_preempt
    sched = stub_world["sched"]
    req = StubRequest("zeroKeep")
    req._canopy_partial_preempt_keep_n = 1
    req._canopy_partial_preempt_deadline = time.time() + 100
    req._canopy_partial_preempt_safety_margin = 5
    sched.kv_cache_manager.register_request_blocks("zeroKeep", ([1, 2, 3],))
    # 3 blocks: upper = 3 - 1 - 5 < 0, so nothing can be kept
    pp_mod._canopy_preempt_request_partial(sched, req, time.time())
    assert calls == ["zeroKeep"]
    assert fresh_counters["skip_due_to_zero_keep_after_safety"] == 1
    assert fresh_counters["succeeded"] == 0


def test_preempt_wrap_full_partial_path(stub_world, fresh_counters, orig_preempt):
    """10 blocks, keep_n=4, safety 1: keep min(4, 10-1-1) = 4, release 6."""
    pp_mod, calls = orig_preempt
    bp, kvm = _with_partial_helpers(stub_world)
    sched = stub_world["sched"]
    req = StubRequest("fullPath", num_computed_tokens=160)
    req._canopy_partial_preempt_keep_n = 4
    req._canopy_partial_preempt_deadline = time.time() + 100
    group = list(range(1, 11))
    kvm.register_request_blocks("fullPath", (group,))
    for bid in group:
        bp.blocks[bid].ref_cnt = 1

    pp_mod._canopy_preempt_request_partial(sched, req, time.time())

    assert calls == []
    assert fresh_counters["preempt_invocations"] == 1
    assert fresh_counters["preempt_partial_path_taken"] == 1
    assert fresh_counters["succeeded"] == 1
    assert fresh_counters["kept_blocks_total"] == 4
    assert fresh_counters["released_blocks_total"] == 6
    assert req.num_computed_tokens == 4 * 16
    assert req.num_preemptions == 1
    assert sched.waiting.prepended == [req]
    assert sched.encoder_cache_manager.freed_requests == ["fullPath"]
    for bid in range(5, 11):
        assert bp.blocks[bid].ref_cnt == 0
        assert bp.blocks[bid] in bp.free_block_queue.blocks_in_queue
    for bid in range(1, 5):
        assert bp.blocks[bid].ref_cnt == 1
        assert bp.blocks[bid] not in bp.free_block_queue.blocks_in_queue


def test_preempt_wrap_last_running_request_of_tree(stub_world, fresh_counters, orig_preempt):
    """No other running request shares the prefix (vLLM's common-prefix count is 0),
    yet the partial path still applies: the bound comes from the request's own blocks."""
    pp_mod, calls = orig_preempt
    bp, kvm = _with_partial_helpers(stub_world)
    sched = stub_world["sched"]
    req = StubRequest("last_sibling", num_computed_tokens=80)
    req._canopy_partial_preempt_keep_n = 3
    req._canopy_partial_preempt_deadline = time.time() + 100
    group = list(range(20, 26))
    kvm.register_request_blocks("last_sibling", (group,))
    for bid in group:
        bp.blocks[bid].ref_cnt = 1
    kvm.set_common_prefix("last_sibling", [])

    pp_mod._canopy_preempt_request_partial(sched, req, time.time())

    assert calls == [], "falling back to a full preemption would drop the prefix"
    assert fresh_counters["succeeded"] == 1
    assert fresh_counters["kept_blocks_total"] == 3
    # 6 blocks: upper = 6 - 1 - 1 = 4, keep min(3, 4) = 3, release 3
    assert fresh_counters["released_blocks_total"] == 3


def test_preempt_wrap_partial_free_exception_falls_back(stub_world, fresh_counters, orig_preempt):
    pp_mod, calls = orig_preempt
    sched = stub_world["sched"]
    req = StubRequest("partialFails")
    req._canopy_partial_preempt_keep_n = 4
    req._canopy_partial_preempt_deadline = time.time() + 100
    sched.kv_cache_manager.register_request_blocks("partialFails", ([1, 2, 3, 4, 5, 6, 7, 8],))

    def _raise(self, request, keep_n_blocks):
        raise RuntimeError("simulated partial_free failure")
    sched.kv_cache_manager.partial_free = types.MethodType(_raise, sched.kv_cache_manager)

    pp_mod._canopy_preempt_request_partial(sched, req, time.time())
    assert calls == ["partialFails"]
    assert fresh_counters["partial_free_exceptions"] == 1
    assert fresh_counters["fallback_to_full"] == 1


def test_preempt_wrap_outer_exception_safe(stub_world, fresh_counters, orig_preempt):
    """An exception inside the wrap falls back to the original; the scheduler never sees it."""
    pp_mod, calls = orig_preempt
    sched = stub_world["sched"]
    req = StubRequest("outerExc")
    saved = sched.kv_cache_manager
    sched.kv_cache_manager = None
    req._canopy_partial_preempt_keep_n = 4
    req._canopy_partial_preempt_deadline = time.time() + 100
    pp_mod._canopy_preempt_request_partial(sched, req, time.time())
    assert calls == ["outerExc"]
    sched.kv_cache_manager = saved


def test_preempt_wrap_preempt_invocations_always_increments(stub_world, fresh_counters, orig_preempt):
    pp_mod, _ = orig_preempt
    for i in range(7):
        pp_mod._canopy_preempt_request_partial(stub_world["sched"], StubRequest(f"liveness_{i}"), time.time())
    assert fresh_counters["preempt_invocations"] == 7
    assert fresh_counters["skip_due_to_no_tag"] == 7
