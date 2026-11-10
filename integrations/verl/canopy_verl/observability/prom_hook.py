# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Periodic vLLM metrics scrape that ticks the admission coordinator.

The admission coordinator has no timer of its own: it acts when it is ticked.
Agent loops call :func:`maybe_scrape` once per generation turn::

    from canopy_verl.observability.prom_hook import maybe_scrape

    await maybe_scrape(turn_idx, algo_tag="ARPO", server_manager=self.server_manager)

Every ``CANOPY_PROM_RATE``-th call (default 10) in a worker process scrapes the
``/metrics`` endpoint of the first vLLM server known to the
``AsyncLLMServerManager`` (verl serves vLLM's Prometheus metrics there when
``actor_rollout_ref.rollout.prometheus.enable=true`` and
``disable_log_stats=false``), builds a
:class:`~canopy_verl.scheduler.admission.KVPressureSnapshot`, and, if the load
balancer has an admission coordinator attached, awaits its ``tick`` (at most
``CANOPY_ADMISSION_TICK_TIMEOUT`` seconds, default 5).

Failures are logged once and never raised.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, Optional
from urllib.request import Request, urlopen


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_RATE = int(os.environ.get("CANOPY_PROM_RATE", "10"))
_counter = 0
_first_failure_logged = False
_endpoint_announced = False
_admission_announced = False
_admission_first_failure_logged = False

# Endpoint of this process, resolved once: (address, port).
_cached_endpoint: Optional[tuple[str, int]] = None

# vLLM metric names; the first name present wins (older vLLM spellings last).
_METRICS = {
    "gpu_kv_cache_util": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "num_preempted_total": ("vllm:num_preemptions_total", "vllm:num_preempted_requests_total"),
    "num_running_seqs": ("vllm:num_requests_running",),
    "num_pending_seqs": ("vllm:num_requests_waiting",),
}

_SAMPLE_RE = re.compile(
    r"^(?P<metric>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^{}]*\})?\s+(?P<value>[-+0-9.eE]+|NaN|\+?Inf|-Inf)\s*$"
)


def _parse_addr_port(addr_port: str) -> Optional[tuple[str, int]]:
    """Parse ``"addr:port"`` or ``"[ipv6]:port"``."""
    if not addr_port:
        return None
    s = addr_port.strip()
    try:
        if s.startswith("["):
            close = s.index("]")
            host = s[1:close]
            port = int(s[close + 2:])  # skip "]:"
        else:
            host, _, port_s = s.rpartition(":")
            if not host or not port_s:
                return None
            port = int(port_s)
        if port <= 0 or port > 65535:
            return None
        return host, port
    except (ValueError, IndexError):
        return None


def _from_server_manager(server_manager: Any) -> Optional[tuple[str, int]]:
    """First server address of an ``AsyncLLMServerManager`` (its keys are ``"addr:port"``)."""
    if server_manager is None:
        return None
    mapping = getattr(server_manager, "_server_id_to_handle", None)
    if not mapping:
        return None
    try:
        first_key = next(iter(mapping.keys()))
    except StopIteration:
        return None
    return _parse_addr_port(first_key)


def _resolve_endpoint(server_manager: Any) -> Optional[tuple[str, int]]:
    """Endpoint to scrape; cached for the process after the first success."""
    global _cached_endpoint
    if _cached_endpoint is None:
        _cached_endpoint = _from_server_manager(server_manager)
    return _cached_endpoint


def parse_kv_pressure(text: str) -> dict[str, float]:
    """Extract KV usage, preemptions, running and waiting requests from Prometheus text.

    Labels are ignored; for a metric with several series the last one wins.
    Missing metrics are omitted.
    """
    seen: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        seen[m.group("metric")] = value
    out: dict[str, float] = {}
    for key, names in _METRICS.items():
        for name in names:
            if name in seen:
                out[key] = seen[name]
                break
    return out


def scrape_kv_pressure(host: str, port: int, timeout: float = 1.5) -> dict[str, float]:
    """GET ``http://host:port/metrics`` and parse it; ``{}`` on failure."""
    try:
        req = Request(f"http://{host}:{port}/metrics", headers={"Accept": "text/plain"})
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("vLLM /metrics scrape failed for %s:%d: %s", host, port, exc)
        return {}
    return parse_kv_pressure(text)


async def maybe_scrape(
    turn_idx: int,
    algo_tag: str,
    server_manager: Any | None = None,
) -> Optional[dict[str, Any]]:
    """Rate-limited scrape that ticks the admission coordinator.

    Args:
        turn_idx: turn index of the calling agent loop (for log lines).
        algo_tag: short name of the calling agent loop (for log lines).
        server_manager: the agent loop's ``AsyncLLMServerManager``; supplies the
            server address and the load-balancer handle.

    Returns:
        The coordinator's ``admission/*`` metrics if a tick ran, else ``None``.
    """
    global _counter, _first_failure_logged, _endpoint_announced
    global _admission_announced, _admission_first_failure_logged
    _counter += 1
    if _counter % _RATE != 0:
        return None
    try:
        endpoint = _resolve_endpoint(server_manager)
        if endpoint is None:
            if not _first_failure_logged:
                _first_failure_logged = True
                logger.warning(
                    "[PROMETHEUS-%s] no vLLM server address known; admission ticks disabled",
                    algo_tag,
                )
            return None
        addr, port = endpoint
        if not _endpoint_announced:
            _endpoint_announced = True
            print(f"[PROMHOOK-ADDR] endpoint={addr}:{port} algo={algo_tag}", flush=True)

        metrics = scrape_kv_pressure(addr, port, timeout=1.5)
        gpu_kv = metrics.get("gpu_kv_cache_util", -1.0)
        lb = getattr(server_manager, "_load_balancer", None)
        if lb is None or gpu_kv < 0:
            return None
        try:
            coord = await lb.get_admission_coordinator.remote()
        except AttributeError:
            return None  # verl's own load balancer: no coordinator
        if coord is None:
            return None

        from canopy_verl.scheduler.admission import KVPressureSnapshot

        preempted = metrics.get("num_preempted_total", -1.0)
        running = metrics.get("num_running_seqs", -1.0)
        waiting = metrics.get("num_pending_seqs", -1.0)
        snapshot = KVPressureSnapshot(
            gpu_cache_usage_perc=float(gpu_kv),
            num_preempted_total=int(preempted) if preempted >= 0 else 0,
            num_running_seqs=int(running) if running >= 0 else 0,
            num_pending_seqs=int(waiting) if waiting >= 0 else 0,
            timestamp_s=time.time(),
        )
        tick_timeout = float(os.environ.get("CANOPY_ADMISSION_TICK_TIMEOUT", "5"))
        try:
            adm_metrics = await asyncio.wait_for(coord.tick.remote(snapshot), timeout=tick_timeout)
        except asyncio.TimeoutError:
            if not _admission_first_failure_logged:
                _admission_first_failure_logged = True
                print(
                    f"[PROMHOOK-ADMISSION] tick timeout ({tick_timeout}s); further timeouts are not logged",
                    flush=True,
                )
            return None
        if not isinstance(adm_metrics, dict):
            return None
        if not _admission_announced:
            _admission_announced = True
            print(
                f"[PROMHOOK-ADMISSION] algo={algo_tag} turn={turn_idx} scrape_idx={_counter} "
                f"total_ticks={adm_metrics.get('admission/total_ticks', -1)} "
                f"pressure_ticks={adm_metrics.get('admission/pressure_ticks', -1)}",
                flush=True,
            )
        return adm_metrics
    except Exception as exc:
        if not _first_failure_logged:
            _first_failure_logged = True
            logger.warning(
                "[PROMETHEUS-%s] scrape or admission tick failed (further failures are not logged): %s: %s",
                algo_tag, type(exc).__name__, exc,
            )
        return None
