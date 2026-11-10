# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""prom_hook: endpoint discovery, metrics parsing, and the admission tick it drives."""

from __future__ import annotations

import asyncio
import importlib
import types

import pytest

from canopy_verl.observability import prom_hook
from canopy_verl.tests._fakes import LocalHandle


@pytest.fixture
def ph():
    """prom_hook with fresh module state (counter, cached endpoint, log-once flags)."""
    return importlib.reload(prom_hook)


def test_parse_addr_port_ipv4(ph):
    assert ph._parse_addr_port("10.0.0.1:8000") == ("10.0.0.1", 8000)
    assert ph._parse_addr_port("127.0.0.1:65535") == ("127.0.0.1", 65535)


def test_parse_addr_port_ipv6(ph):
    assert ph._parse_addr_port("[::1]:8000") == ("::1", 8000)
    assert ph._parse_addr_port("[fe80::1]:8001") == ("fe80::1", 8001)


def test_parse_addr_port_invalid(ph):
    for bad in ("", "no-port", "addr:notanint", "addr:-1", "addr:99999"):
        assert ph._parse_addr_port(bad) is None, bad


def test_from_server_manager(ph):
    sm = types.SimpleNamespace(_server_id_to_handle={"192.168.1.5:8000": object(), "10.0.0.2:8002": object()})
    assert ph._from_server_manager(sm) == ("192.168.1.5", 8000)
    assert ph._from_server_manager(None) is None
    assert ph._from_server_manager(types.SimpleNamespace()) is None
    assert ph._from_server_manager(types.SimpleNamespace(_server_id_to_handle={})) is None


def test_resolve_endpoint_is_cached(ph):
    sm = types.SimpleNamespace(_server_id_to_handle={"10.0.0.1:7000": object()})
    assert ph._resolve_endpoint(sm) == ("10.0.0.1", 7000)
    sm._server_id_to_handle = {"10.0.0.9:9999": object()}
    assert ph._resolve_endpoint(sm) == ("10.0.0.1", 7000)


def test_parse_kv_pressure(ph):
    text = "\n".join([
        "# HELP vllm:kv_cache_usage_perc KV-cache usage.",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42',
        'vllm:num_preemptions_total{engine="0",model_name="m"} 7.0',
        'vllm:num_preemptions_created{engine="0",model_name="m"} 1.7e9',
        'vllm:num_requests_running{engine="0",model_name="m"} 12.0',
        'vllm:num_requests_waiting{engine="0",model_name="m"} 3.0',
        "vllm:other_metric 1.0",
    ])
    assert ph.parse_kv_pressure(text) == {
        "gpu_kv_cache_util": 0.42,
        "num_preempted_total": 7.0,
        "num_running_seqs": 12.0,
        "num_pending_seqs": 3.0,
    }


def test_parse_kv_pressure_older_metric_names(ph):
    text = "vllm:gpu_cache_usage_perc 0.9\nvllm:num_preempted_requests_total 2\n"
    assert ph.parse_kv_pressure(text) == {"gpu_kv_cache_util": 0.9, "num_preempted_total": 2.0}


class _Coordinator:
    def __init__(self):
        self.snapshots = []

    async def tick(self, snapshot):
        self.snapshots.append(snapshot)
        return {"admission/total_ticks": len(self.snapshots), "admission/pressure_ticks": 0}


class _LoadBalancer:
    def __init__(self, coordinator):
        self._coordinator = coordinator

    def get_admission_coordinator(self):
        return self._coordinator


def _server_manager(lb):
    return types.SimpleNamespace(_server_id_to_handle={"127.0.0.1:8000": object()},
                                 _load_balancer=LocalHandle(lb) if lb is not None else None)


def _fake_scrape(ph, monkeypatch, metrics):
    calls = []

    def scrape(host, port, timeout=1.5):
        calls.append((host, port))
        return dict(metrics)

    monkeypatch.setattr(ph, "scrape_kv_pressure", scrape)
    return calls


METRICS = {"gpu_kv_cache_util": 0.9, "num_preempted_total": 3.0, "num_running_seqs": 5.0, "num_pending_seqs": 2.0}


def test_maybe_scrape_ticks_the_coordinator(ph, monkeypatch):
    monkeypatch.setattr(ph, "_RATE", 1)
    calls = _fake_scrape(ph, monkeypatch, METRICS)
    coord = _Coordinator()
    out = asyncio.run(ph.maybe_scrape(1, "TEST", _server_manager(_LoadBalancer(LocalHandle(coord)))))
    assert calls == [("127.0.0.1", 8000)]
    assert out == {"admission/total_ticks": 1, "admission/pressure_ticks": 0}
    snap = coord.snapshots[0]
    assert (snap.gpu_cache_usage_perc, snap.num_preempted_total, snap.num_running_seqs, snap.num_pending_seqs) \
        == (0.9, 3, 5, 2)


def test_maybe_scrape_is_rate_limited(ph, monkeypatch):
    monkeypatch.setattr(ph, "_RATE", 3)
    calls = _fake_scrape(ph, monkeypatch, METRICS)
    coord = _Coordinator()
    sm = _server_manager(_LoadBalancer(LocalHandle(coord)))
    outs = [asyncio.run(ph.maybe_scrape(i, "TEST", sm)) for i in range(6)]
    assert [o is not None for o in outs] == [False, False, True, False, False, True]
    assert len(calls) == 2 and len(coord.snapshots) == 2


def test_maybe_scrape_without_coordinator(ph, monkeypatch):
    monkeypatch.setattr(ph, "_RATE", 1)
    _fake_scrape(ph, monkeypatch, METRICS)
    assert asyncio.run(ph.maybe_scrape(1, "TEST", _server_manager(_LoadBalancer(None)))) is None


def test_maybe_scrape_with_verl_load_balancer(ph, monkeypatch):
    """verl's own balancer has no get_admission_coordinator: no tick, no error."""
    monkeypatch.setattr(ph, "_RATE", 1)
    _fake_scrape(ph, monkeypatch, METRICS)
    assert asyncio.run(ph.maybe_scrape(1, "TEST", _server_manager(object()))) is None


def test_maybe_scrape_skips_tick_without_kv_usage(ph, monkeypatch):
    monkeypatch.setattr(ph, "_RATE", 1)
    _fake_scrape(ph, monkeypatch, {"num_running_seqs": 1.0})
    coord = _Coordinator()
    assert asyncio.run(ph.maybe_scrape(1, "TEST", _server_manager(_LoadBalancer(LocalHandle(coord))))) is None
    assert coord.snapshots == []
