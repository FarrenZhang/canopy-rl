# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Prefix-lock patch (canopy_verl.vllm_patch.prefix_lock) against stub vLLM classes.

Fake ``vllm.v1.core.block_pool`` / ``vllm.v1.engine.async_llm`` modules are
installed in ``sys.modules`` so the patch attaches to stub classes; no vLLM
engine is started.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest


class _FakeKVCacheBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id

    def __repr__(self) -> str:
        return f"FB({self.block_id})"


class _FakeFreeQueue:
    """Free-block queue with the operations the patch uses (append to tail, pop from head)."""

    def __init__(self, blocks):
        self._blocks = list(blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def append(self, block):
        self._blocks.append(block)

    def remove(self, block):
        self._blocks.remove(block)

    def pop_head(self):
        return self._blocks.pop(0)

    def get_all_free_blocks(self):
        return list(self._blocks)


class _FakeBlockPool:
    """Stand-in for ``vllm.v1.core.block_pool.BlockPool``."""

    def __init__(self, block_ids):
        self.free_block_queue = _FakeFreeQueue(
            [_FakeKVCacheBlock(b) for b in block_ids]
        )

    def get_new_blocks(self, num_blocks: int):
        out = []
        for _ in range(num_blocks):
            if len(self.free_block_queue) == 0:
                break
            out.append(self.free_block_queue.pop_head())
        return out


class _FakeEngineCore:
    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager


class _FakeKVCacheManager:
    def __init__(self, block_pool, block_map):
        self.block_pool = block_pool
        self._block_map = dict(block_map)  # request_id -> tuple(lists)

    def get_block_ids(self, request_id: str):
        v = self._block_map.get(request_id)
        if v is None:
            raise KeyError(request_id)
        return v


class _FakeAsyncLLM:
    def __init__(self, engine_core):
        self.engine_core = engine_core


def _install_fake_vllm_modules(monkeypatch):
    bp_mod = types.ModuleType("vllm.v1.core.block_pool")
    bp_mod.BlockPool = _FakeBlockPool
    async_mod = types.ModuleType("vllm.v1.engine.async_llm")
    async_mod.AsyncLLM = _FakeAsyncLLM
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.core", types.ModuleType("vllm.v1.core"))
    monkeypatch.setitem(sys.modules, "vllm.v1.core.block_pool", bp_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", types.ModuleType("vllm.v1.engine"))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", async_mod)


def _reset_module_state():
    from canopy_verl.vllm_patch import prefix_lock as pl_mod

    pl_mod._PATCHED_BLOCK_POOL = False
    pl_mod._PATCHED_ASYNC_LLM = False
    pl_mod._ORIG_GET_NEW_BLOCKS = None
    pl_mod._APPLY_MARKER_PRINTED = False


@pytest.fixture
def patched_vllm(monkeypatch):
    """Install the fake vLLM modules, apply the patch, and revert it afterwards."""
    _install_fake_vllm_modules(monkeypatch)
    # Importing canopy_verl may already have patched other (real or fake) classes.
    _reset_module_state()

    from canopy_verl.vllm_patch import apply_prefix_lock_patch, revert_prefix_lock_patch

    assert apply_prefix_lock_patch() is True
    yield {
        "BlockPool": _FakeBlockPool,
        "AsyncLLM": _FakeAsyncLLM,
        "KVCacheManager": _FakeKVCacheManager,
        "EngineCore": _FakeEngineCore,
    }
    revert_prefix_lock_patch()


class TestApplyAndRevert:
    def test_apply_idempotent(self, patched_vllm):
        from canopy_verl.vllm_patch import (
            apply_prefix_lock_patch,
            applied_prefix_lock_patch,
        )

        assert applied_prefix_lock_patch() is True
        assert apply_prefix_lock_patch() is True

    def test_block_pool_methods_attached(self, patched_vllm):
        BP = patched_vllm["BlockPool"]
        for name in (
            "set_prefix_lock",
            "unset_prefix_lock",
            "_canopy_expire_ttl",
            "get_prefix_lock_state",
        ):
            assert hasattr(BP, name), f"missing method {name}"

    def test_async_llm_methods_attached(self, patched_vllm):
        AL = patched_vllm["AsyncLLM"]
        for name in (
            "set_prefix_lock_for_request",
            "unset_prefix_lock_for_request",
            "get_prefix_lock_stats",
        ):
            assert hasattr(AL, name)


class TestBlockPoolLockUnlock:
    def test_set_then_get_state_counters(self, patched_vllm):
        pool = patched_vllm["BlockPool"]([10, 11, 12])
        assert pool.set_prefix_lock([10, 11], ttl_sec=5.0) == 2
        state = pool.get_prefix_lock_state()
        assert state["locked_blocks"] == 2
        assert state["lock_events_total"] == 2

    def test_set_twice_same_ids_no_duplicate(self, patched_vllm):
        pool = patched_vllm["BlockPool"]([10, 11, 12])
        pool.set_prefix_lock([10], ttl_sec=5.0)
        pool.set_prefix_lock([10], ttl_sec=5.0)
        assert pool.get_prefix_lock_state()["locked_blocks"] == 1
        assert pool.get_prefix_lock_state()["lock_events_total"] == 1

    def test_set_then_unset(self, patched_vllm):
        pool = patched_vllm["BlockPool"]([10, 11, 12])
        pool.set_prefix_lock([10, 11], ttl_sec=5.0)
        assert pool.unset_prefix_lock([10, 11]) == 2
        state = pool.get_prefix_lock_state()
        assert state["locked_blocks"] == 0
        assert state["unlock_events_total"] == 2

    def test_ttl_expires(self, patched_vllm):
        pool = patched_vllm["BlockPool"]([10, 11, 12])
        pool.set_prefix_lock([10], ttl_sec=0.01)
        time.sleep(0.05)
        assert pool._canopy_expire_ttl() == 1
        assert pool.get_prefix_lock_state()["locked_blocks"] == 0


class TestGetNewBlocksSkip:
    def test_get_new_blocks_skips_locked_block(self, patched_vllm):
        # Free queue [10, 11, 12] with 10 locked: one new block must be 11.
        pool = patched_vllm["BlockPool"]([10, 11, 12])
        pool.set_prefix_lock([10], ttl_sec=5.0)
        new_blocks = pool.get_new_blocks(1)
        assert [b.block_id for b in new_blocks] == [11]
        assert pool.get_prefix_lock_state()["evictions_skipped_due_to_lock"] >= 1
        # the locked block went back to the tail of the queue
        assert [b.block_id for b in pool.free_block_queue.get_all_free_blocks()] == [12, 10]

    def test_get_new_blocks_after_unlock_returns_all(self, patched_vllm):
        pool = patched_vllm["BlockPool"]([10, 11, 12])
        pool.set_prefix_lock([10], ttl_sec=5.0)
        pool.unset_prefix_lock([10])
        assert pool.get_new_blocks(1)[0].block_id == 10

    def test_get_new_blocks_no_lock_fast_path(self, patched_vllm):
        pool = patched_vllm["BlockPool"]([10, 11, 12])
        assert [b.block_id for b in pool.get_new_blocks(2)] == [10, 11]
        assert pool.get_prefix_lock_state()["evictions_skipped_due_to_lock"] == 0


class TestAsyncLLMFacade:
    def test_set_prefix_lock_for_request_success(self, patched_vllm):
        BP, AL = patched_vllm["BlockPool"], patched_vllm["AsyncLLM"]
        KM, EC = patched_vllm["KVCacheManager"], patched_vllm["EngineCore"]
        pool = BP([10, 11, 12, 13])
        km = KM(pool, {"r0": ([10, 11], [12])})  # two KV cache groups
        engine = AL(EC(km))
        result = asyncio.run(engine.set_prefix_lock_for_request("r0", ttl_sec=10.0))
        assert result == {
            "locked": 3,
            "total_blocks": 3,
            "request_id": "r0",
            "path": "direct",
        }
        assert pool.get_prefix_lock_state()["locked_blocks"] == 3

    def test_set_prefix_lock_for_unknown_request(self, patched_vllm):
        BP, AL = patched_vllm["BlockPool"], patched_vllm["AsyncLLM"]
        KM, EC = patched_vllm["KVCacheManager"], patched_vllm["EngineCore"]
        engine = AL(EC(KM(BP([10, 11]), {"r0": ([10], [])})))
        result = asyncio.run(engine.set_prefix_lock_for_request("unknown-rid"))
        assert result["locked"] == 0
        assert "error" in result

    def test_unset_prefix_lock_for_request(self, patched_vllm):
        BP, AL = patched_vllm["BlockPool"], patched_vllm["AsyncLLM"]
        KM, EC = patched_vllm["KVCacheManager"], patched_vllm["EngineCore"]
        engine = AL(EC(KM(BP([10, 11]), {"r0": ([10, 11],)})))

        async def run():
            await engine.set_prefix_lock_for_request("r0")
            return await engine.unset_prefix_lock_for_request("r0")

        assert asyncio.run(run())["unlocked"] == 2

    def test_get_prefix_lock_stats_returns_dict(self, patched_vllm):
        BP, AL = patched_vllm["BlockPool"], patched_vllm["AsyncLLM"]
        KM, EC = patched_vllm["KVCacheManager"], patched_vllm["EngineCore"]
        engine = AL(EC(KM(BP([10, 11]), {"r0": ([10],)})))

        async def run():
            await engine.set_prefix_lock_for_request("r0")
            return await engine.get_prefix_lock_stats()

        stats = asyncio.run(run())
        assert stats["locked_blocks"] == 1
        assert stats["lock_events_total"] == 1

    def test_rpc_path_through_call_utility_async(self, patched_vllm):
        AL = patched_vllm["AsyncLLM"]
        calls = []

        class _Client:  # multiprocess engine client: no kv_cache_manager attribute
            async def call_utility_async(self, method, *args):
                calls.append((method, args))
                return {"locked": 5, "request_id": args[0]}

        result = asyncio.run(AL(_Client()).set_prefix_lock_for_request("r9", ttl_sec=7.0))
        assert calls == [("canopy_set_prefix_lock_for_request", ("r9", 7.0))]
        assert result == {"locked": 5, "request_id": "r9", "path": "rpc"}


class TestImportFallback:
    def test_patch_no_break_on_vllm_import_fail(self, monkeypatch):
        _reset_module_state()
        monkeypatch.setitem(sys.modules, "vllm.v1.core.block_pool", None)
        from canopy_verl.vllm_patch import apply_prefix_lock_patch

        assert apply_prefix_lock_patch() is False

    def test_verl_server_extensions_import_fail(self, monkeypatch):
        from canopy_verl.vllm_patch import verl_server_extensions as vse_mod

        monkeypatch.setattr(vse_mod, "_APPLIED", False)
        monkeypatch.setitem(
            sys.modules,
            "verl.workers.rollout.vllm_rollout.vllm_async_server",
            None,
        )
        from canopy_verl.vllm_patch import apply_verl_server_extensions

        assert apply_verl_server_extensions() is False
