# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
#
# The mock block pool below mirrors vLLM v0.20 (vllm/v1/core/kv_cache_utils.py,
# vllm/v1/core/block_pool.py).
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of canopy.engine.vllm_hooks against a mock of vLLM 0.20's block pool.

Run with ``python -m pytest tests/test_vllm_hooks.py`` or directly with
``python tests/test_vllm_hooks.py``.
"""

import enum
import importlib
import importlib.util
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from canopy.engine import HOOKS_DIR, vllm_hooks  # noqa: E402


class KVCacheBlock:
    def __init__(self, block_id, is_null=False):
        self.block_id, self.ref_cnt, self._block_hash = block_id, 0, None
        self.prev_free_block = self.next_free_block = None
        self.is_null = is_null

    @property
    def block_hash(self):
        return self._block_hash

    @block_hash.setter
    def block_hash(self, h):
        assert self._block_hash is None
        self._block_hash = h

    def reset_hash(self):
        self._block_hash = None


class FreeQ:
    def __init__(self, blocks):
        self.num_free_blocks = len(blocks)
        for i in range(len(blocks)):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < len(blocks) - 1:
                blocks[i].next_free_block = blocks[i + 1]
        self.fake_free_list_head, self.fake_free_list_tail = KVCacheBlock(-1), KVCacheBlock(-1)
        self.fake_free_list_head.next_free_block = blocks[0]
        blocks[0].prev_free_block = self.fake_free_list_head
        self.fake_free_list_tail.prev_free_block = blocks[-1]
        blocks[-1].next_free_block = self.fake_free_list_tail

    def popleft_n(self, n):
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n
        cur, ret = self.fake_free_list_head.next_free_block, []
        for _ in range(n):
            ret.append(cur)
            last, cur = cur, cur.next_free_block
            last.prev_free_block = last.next_free_block = None
        self.fake_free_list_head.next_free_block = cur
        cur.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, b):
        assert b.prev_free_block is not None and b.next_free_block is not None
        b.prev_free_block.next_free_block = b.next_free_block
        b.next_free_block.prev_free_block = b.prev_free_block
        b.prev_free_block = b.next_free_block = None
        self.num_free_blocks -= 1

    def append_n(self, blocks):
        for b in blocks:
            last = self.fake_free_list_tail.prev_free_block
            last.next_free_block = b
            b.prev_free_block = last
            b.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = b
            self.num_free_blocks += 1

    def items(self):
        out, cur = [], self.fake_free_list_head.next_free_block
        while cur is not self.fake_free_list_tail:
            out.append(cur)
            cur = cur.next_free_block
        return out


class Pool:
    def __init__(self, n):
        self.blocks = [KVCacheBlock(i) for i in range(n)]
        self.blocks[0].is_null = True
        self.blocks[0].ref_cnt = 1
        self.free_block_queue = FreeQ(self.blocks[1:])
        self.cache = {}

    def get_new_blocks(self, k):
        ret = self.free_block_queue.popleft_n(k)
        for b in ret:
            if b.block_hash is not None:
                self.cache.pop(b.block_hash, None)
                b.reset_hash()
            b.ref_cnt += 1
        return ret

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free_block_queue.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, ordered):
        bl = list(ordered)
        for b in bl:
            b.ref_cnt -= 1
        self.free_block_queue.append_n([b for b in bl if b.ref_cnt == 0 and not b.is_null])


Status = enum.IntEnum("Status", "WAITING RUNNING PREEMPTED")


class Req:
    def __init__(self, rid, hint, n_computed):
        self.request_id, self.status, self.num_computed_tokens = rid, Status.RUNNING, n_computed
        self.sampling_params = types.SimpleNamespace(extra_args=hint)


HINT = {"canopy_mode": "lifetime", "canopy_terminal": 0, "canopy_group_done": 0, "canopy_shared_len": 64,
        "canopy_group": "g", "canopy_pin": 1, "canopy_pp": 1, "canopy_pin_ttl": 60.0}


def _stm_class():
    """A fresh SingleTypeKVCacheManager class (the hooks patch classes in place)."""

    class SingleTypeKVCacheManager:
        def __init__(self, pool):
            self.block_pool, self.block_size = pool, 16
            self.req_to_blocks, self.num_cached_block = {}, {}

        def free(self, rid):
            blocks = self.req_to_blocks.pop(rid, [])
            self.block_pool.free_blocks(reversed(blocks))
            self.num_cached_block.pop(rid, None)

    return SingleTypeKVCacheManager


def _sched_class(stm, pool):
    class Scheduler:
        def __init__(self):
            self.running, self.waiting = [], []
            self.kv_cache_manager = types.SimpleNamespace(
                coordinator=types.SimpleNamespace(single_type_managers=[stm]), block_pool=pool,
                free=lambda req: stm.free(req.request_id))

        def _free_blocks(self, request):
            stm.free(request.request_id)

        def schedule(self):
            return types.SimpleNamespace(total_num_scheduled_tokens=0)

        def reset_prefix_cache(self, *a, **k):
            return True

        def _preempt_request(self, request, ts):
            self.kv_cache_manager.free(request)
            request.status = Status.PREEMPTED
            request.num_computed_tokens = 0

    return Scheduler


def _world(n_blocks=64):
    """Fresh hook state, a patched SingleTypeKVCacheManager and an empty pool."""
    hooks = importlib.reload(vllm_hooks)
    stm_cls = _stm_class()
    hooks._patch_single_type(types.SimpleNamespace(SingleTypeKVCacheManager=stm_cls))
    pool = Pool(n_blocks)
    return hooks, pool, stm_cls(pool)


def _scheduler(hooks, pool, stm):
    sched_cls = _sched_class(stm, pool)
    hooks._patch_scheduler(types.SimpleNamespace(Scheduler=sched_cls))
    return sched_cls()


def _finish(hooks, stm, rid, hint):
    """Free a request the way the patched Scheduler._free_blocks does for lifetime hints."""
    hooks._CUR_HINT.update(rid=rid, hint=hint)
    try:
        stm.free(rid)
    finally:
        hooks._CUR_HINT.update(rid=None, hint=None)


def check(pool):
    """Every non-null block with ref 0 is in the queue exactly once; nothing else is."""
    q = pool.free_block_queue.items()
    ids = [b.block_id for b in q]
    assert len(ids) == len(set(ids)) == pool.free_block_queue.num_free_blocks, "queue size mismatch"
    want = sorted(b.block_id for b in pool.blocks if b.ref_cnt == 0 and not b.is_null)
    assert sorted(ids) == want, f"queue {sorted(ids)} != ref0 {want}"
    assert all(b.ref_cnt >= 0 for b in pool.blocks), "negative ref"


def alloc(pool, stm, rid, blocks_hashes, reuse=()):
    """Allocate a request: reuse cached blocks (touch) then new blocks; hash the new ones."""
    reuse = list(reuse)
    pool.touch(reuse)
    new = pool.get_new_blocks(len(blocks_hashes) - len(reuse))
    for b, h in zip(new, blocks_hashes[len(reuse):]):
        if h is not None:
            b.block_hash = h
            pool.cache[h] = b
    stm.req_to_blocks[rid] = reuse + new
    return reuse + new


def _group_of_two(pool, stm):
    shared = [("g", i) for i in range(4)]
    a = alloc(pool, stm, "a", shared + [("a", i) for i in range(4)])
    b = alloc(pool, stm, "b", shared + [("b", i) for i in range(3)], reuse=a[:4])
    check(pool)
    return a, b


def test_lifetime_free_pins_shared_prefix_of_live_group():
    hooks, pool, stm = _world()
    a, _ = _group_of_two(pool, stm)
    _finish(hooks, stm, "a", HINT)
    check(pool)
    assert all(x.ref_cnt == 2 for x in a[:4]), [x.ref_cnt for x in a[:4]]  # b + pin
    assert len(hooks._GPINS["g"]["blocks"]) == 4


def test_group_done_releases_pin_and_evicts_group_blocks_first():
    hooks, pool, stm = _world()
    a, _ = _group_of_two(pool, stm)
    _finish(hooks, stm, "a", HINT)
    _finish(hooks, stm, "b", dict(HINT, canopy_terminal=1, canopy_group_done=1))
    check(pool)
    assert "g" not in hooks._GPINS
    head = [x.block_id for x in pool.free_block_queue.items()[:7]]
    assert set(head) >= {x.block_id for x in a[:4]}, head


def test_soft_release_frees_oldest_pin():
    hooks, pool, stm = _world()
    alloc(pool, stm, "c", [("h", i) for i in range(5)])
    _finish(hooks, stm, "c", dict(HINT, canopy_group="h"))
    check(pool)
    assert hooks._pinned_blocks() == 4
    assert hooks._release_oldest() == 4
    check(pool)
    assert hooks._pinned_blocks() == 0


def test_prefix_preserving_preemption_keeps_prefix_until_resume():
    hooks, pool, stm = _world()
    s = _scheduler(hooks, pool, stm)
    d = alloc(pool, stm, "d", [("k", i) for i in range(6)])
    req = Req("d", dict(HINT, canopy_group="k", canopy_pin=0), n_computed=6 * 16)
    s._preempt_request(req, 0.0)
    check(pool)
    assert hooks._PPPINS["d"]["blocks"] == d[:4] and all(x.ref_cnt == 1 for x in d[:4])
    assert all(x.ref_cnt == 0 for x in d[4:])
    # resume: the prefix is found in the cache (touch), only the suffix is allocated again
    d2 = alloc(pool, stm, "d", [("k", i) for i in range(4)] + [("k2", i) for i in range(2)], reuse=d[:4])
    req.status, req.num_computed_tokens = Status.RUNNING, 4 * 16
    s.running = [req]
    s.schedule()
    check(pool)
    assert "d" not in hooks._PPPINS and all(x.ref_cnt == 1 for x in d2[:4])
    s._free_blocks(req)
    check(pool)


def test_idle_scheduler_releases_all_pins():
    hooks, pool, stm = _world()
    s = _scheduler(hooks, pool, stm)
    alloc(pool, stm, "e", [("m", i) for i in range(5)])
    _finish(hooks, stm, "e", dict(HINT, canopy_group="m"))
    s.running, s.waiting = [], [object()]
    s.schedule()
    check(pool)
    assert hooks._pinned_blocks() == 0 and hooks.STATS["idle_releases"] == 1


def test_allocate_slots_admit_mode_holds_and_evict_mode_yields():
    hooks, pool, stm = _world()

    class KVCacheManager:
        def allocate_slots(self, request, need):
            return "ok" if pool.free_block_queue.num_free_blocks >= need else None

    hooks._patch_kv_manager(types.SimpleNamespace(KVCacheManager=KVCacheManager))
    km = KVCacheManager()
    alloc(pool, stm, "f", [("q", i) for i in range(6)])
    _finish(hooks, stm, "f", dict(HINT, canopy_group="q", canopy_pin_mode="admit"))
    waiting = Req("w", None, 0)
    waiting.status = Status.WAITING
    free_now = pool.free_block_queue.num_free_blocks
    assert km.allocate_slots(waiting, free_now + 4) is None and hooks._pinned_blocks() == 4  # admit-mode holds
    running = Req("r", None, 0)
    assert km.allocate_slots(running, free_now + 4) == "ok" and hooks._pinned_blocks() == 0  # yields to running
    alloc(pool, stm, "g", [("z", i) for i in range(6)])
    _finish(hooks, stm, "g", dict(HINT, canopy_group="z", canopy_pin_mode="evict"))
    free_now = pool.free_block_queue.num_free_blocks
    assert km.allocate_slots(waiting, free_now + 4) == "ok" and hooks._pinned_blocks() == 0  # evict-mode yields
    check(pool)


def test_ttl_pin_released_by_next_turn_or_expiry():
    hooks, pool, stm = _world()
    s = _scheduler(hooks, pool, stm)
    ttl_hint = {"canopy_mode": "ttl", "canopy_terminal": 0, "canopy_traj": "T", "canopy_pin_ttl": 60.0}
    t0 = alloc(pool, stm, "t0", [("T", i) for i in range(5)])
    s._free_blocks(Req("t0", ttl_hint, 5 * 16))
    check(pool)
    assert all(x.ref_cnt == 1 for x in t0) and hooks.STATS["ttl_pins"] == 1
    # the next turn of trajectory T reuses the cached blocks and is admitted
    t1 = alloc(pool, stm, "t1", [("T", i) for i in range(5)] + [("T1", 0)], reuse=t0)
    nxt = Req("t1", dict(ttl_hint, canopy_terminal=1), 6 * 16)
    s.running = [nxt]
    s.schedule()
    check(pool)
    assert hooks.STATS["ttl_released_by_next_turn"] == 1 and not hooks._PINS
    assert all(x.ref_cnt == 1 for x in t1)
    s.running = []
    s._free_blocks(nxt)  # terminal turn: no pin
    check(pool)
    # a pin that no later turn claims expires
    u = alloc(pool, stm, "u0", [("U", i) for i in range(3)])
    s._free_blocks(Req("u0", dict(ttl_hint, canopy_traj="U", canopy_pin_ttl=0.01), 3 * 16))
    assert len(hooks._PINS) == 1
    time.sleep(0.05)
    s.schedule()
    check(pool)
    assert hooks.STATS["ttl_expired"] == 1 and not hooks._PINS and all(x.ref_cnt == 0 for x in u)


def test_requests_without_hints_take_the_vllm_path():
    hooks, pool, stm = _world()
    s = _scheduler(hooks, pool, stm)
    blocks = alloc(pool, stm, "x", [("x", i) for i in range(4)])
    s._free_blocks(Req("x", None, 4 * 16))
    check(pool)
    tail = [b.block_id for b in pool.free_block_queue.items()[-4:]]
    assert tail == [b.block_id for b in reversed(blocks)]
    assert hooks.STATS["finished_frees"] == 1 and hooks.STATS["lifetime_frees"] == 0


def test_install_is_idempotent_across_module_copies():
    hooks = importlib.reload(vllm_hooks)
    saved = list(sys.meta_path)
    try:
        sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "_canopy_engine_hooks", False)]
        assert hooks.install() is True
        assert hooks.install() is False
        spec = importlib.util.spec_from_file_location("canopy_hooks_copy", hooks.__file__)
        copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy)
        assert copy.install() is False
        assert sum(bool(getattr(f, "_canopy_engine_hooks", False)) for f in sys.meta_path) == 1
        # a finder from another module copy must not delegate back to the first one
        sys.meta_path.insert(0, copy._Finder())
        found = sys.meta_path[0].find_spec("vllm.v1.core.kv_cache_manager", None)
        assert found is None or isinstance(found.loader, copy._Loader)
    finally:
        sys.meta_path[:] = saved


def test_sitecustomize_installs_hooks_only_when_enabled():
    path = os.path.join(HOOKS_DIR, "sitecustomize.py")
    with open(path) as f:
        code = compile(f.read(), path, "exec")
    saved_path, saved_meta = list(sys.path), list(sys.meta_path)
    saved_env = os.environ.get("CANOPY_ENGINE_HOOKS")
    try:
        sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "_canopy_engine_hooks", False)]
        os.environ.pop("CANOPY_ENGINE_HOOKS", None)
        exec(code, {"__name__": "sitecustomize"})
        assert not any(getattr(f, "_canopy_engine_hooks", False) for f in sys.meta_path)
        os.environ["CANOPY_ENGINE_HOOKS"] = "1"
        sys.path.insert(0, HOOKS_DIR)
        exec(code, {"__name__": "sitecustomize"})
        assert any(getattr(f, "_canopy_engine_hooks", False) for f in sys.meta_path)
    finally:
        sys.path[:] = saved_path
        sys.meta_path[:] = saved_meta
        sys.modules.pop("vllm_hooks", None)
        if saved_env is None:
            os.environ.pop("CANOPY_ENGINE_HOOKS", None)
        else:
            os.environ["CANOPY_ENGINE_HOOKS"] = saved_env


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} tests passed")
