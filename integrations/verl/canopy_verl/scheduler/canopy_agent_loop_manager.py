# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""``CanopyAgentLoopManager``: verl's ``AgentLoopManager`` with tree-aware placement and admission.

Overrides :meth:`_init_global_load_balancer` to

  1. create :class:`canopy_verl.scheduler.placement.TreeAwareLoadBalancer`
     instead of verl's ``GlobalRequestLoadBalancer``, and
  2. if ``canopy_scheduler.admission.enabled`` is true, create an
     :class:`canopy_verl.scheduler.admission.AdmissionCoordinator` with the
     balancer handle and the vLLM server handles, and attach it to the
     balancer (``set_admission_coordinator``), where
     :func:`canopy_verl.observability.prom_hook.maybe_scrape` finds it.

verl loads custom managers by name; enable this one with::

    actor_rollout_ref.rollout.agent.agent_loop_manager_class=canopy_verl.scheduler.canopy_agent_loop_manager.CanopyAgentLoopManager

and configure it under ``canopy_scheduler`` (see ``configs/canopy_scheduler.yaml``)::

    canopy_scheduler:
      placement:
        enabled: true              # false: verl's routing
        sticky_threshold: 0.75
        max_cache_size: 10000
      admission:
        enabled: false             # true: create the AdmissionCoordinator
        mode: "a"                  # "a" | "b1" | "b2" (see admission.py)
        kv_pressure_threshold: 0.85
        prefix_lock_ttl_sec: 60.0  # mode b1 / b2
        prefix_lock_refresh_sec: 15.0
        prefix_lock_top_k: 1

With ``admission.enabled=false`` no coordinator is created; with
``placement.enabled=false`` as well, routing is identical to verl's
``GlobalRequestLoadBalancer``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopManager

from canopy_verl.scheduler.admission import (
    AdmissionConfig,
    AdmissionCoordinator,
)
from canopy_verl.scheduler.placement import TreeAwareLoadBalancer


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _read_canopy_placement_config(config) -> dict[str, Any]:
    """Read ``canopy_scheduler.placement``; missing keys default to verl's routing."""
    defaults = {
        "enabled": False,
        "sticky_threshold": 0.75,
        "max_cache_size": 10000,
    }

    canopy_root = getattr(config, "canopy_scheduler", None)
    if canopy_root is None:
        return defaults
    placement = getattr(canopy_root, "placement", None)
    if placement is None:
        return defaults

    def _get(key, default):
        try:
            v = placement.get(key, default)
        except Exception:
            v = getattr(placement, key, default)
        return default if v is None else v

    return {
        "enabled": bool(_get("enabled", defaults["enabled"])),
        "sticky_threshold": float(
            _get("sticky_threshold", defaults["sticky_threshold"])
        ),
        "max_cache_size": int(_get("max_cache_size", defaults["max_cache_size"])),
    }


def _read_canopy_admission_config(config) -> AdmissionConfig:
    """Read ``canopy_scheduler.admission``; defaults when absent."""
    canopy_root = getattr(config, "canopy_scheduler", None)
    if canopy_root is None:
        return AdmissionConfig()
    return AdmissionConfig.from_hydra(canopy_root)


class CanopyAgentLoopManager(AgentLoopManager):
    """AgentLoopManager with the tree-aware load balancer and the admission coordinator."""

    async def _init_global_load_balancer(self) -> None:  # type: ignore[override]
        placement_cfg = _read_canopy_placement_config(self.config)

        lb_handle = TreeAwareLoadBalancer.remote(
            server_actor_ids=self.server_addresses,
            max_cache_size=placement_cfg["max_cache_size"],
            sticky_threshold=placement_cfg["sticky_threshold"],
            enable_tree_sticky=placement_cfg["enabled"],
        )
        self.global_load_balancer = lb_handle
        print(
            f"[CANOPY-PLACEMENT] TreeAwareLoadBalancer initialised "
            f"enabled={placement_cfg['enabled']} "
            f"sticky_threshold={placement_cfg['sticky_threshold']} "
            f"max_cache_size={placement_cfg['max_cache_size']} "
            f"n_servers={len(self.server_addresses)}",
            flush=True,
        )

        # Apply the vLLM / vLLMHttpServer patches of the selected mode in this
        # process (mode b2 includes the b1 patches).  The vLLM server actors are
        # created before this method runs; they get the RPC methods from the
        # patches applied when canopy_verl is imported (canopy_verl/__init__.py).
        admission_cfg = _read_canopy_admission_config(self.config)
        b1_patch_applied = False
        b2_patch_applied = False
        if admission_cfg.enabled and admission_cfg.mode == "b1":
            try:
                from canopy_verl.vllm_patch import apply_all_b1_patches

                results = apply_all_b1_patches()
                b1_patch_applied = all(results.values())
                print(
                    f"[CANOPY-VLLM-PATCH] apply_all_b1_patches results={results} "
                    f"all_ok={b1_patch_applied}",
                    flush=True,
                )
            except Exception as exc:
                logger.warning(
                    "[CANOPY-VLLM-PATCH] apply_all_b1_patches raised: %s: %s",
                    type(exc).__name__,
                    exc,
                )
        elif admission_cfg.enabled and admission_cfg.mode == "b2":
            try:
                from canopy_verl.vllm_patch import apply_all_b2_patches

                results = apply_all_b2_patches()
                b2_patch_applied = all(results.values())
                b1_patch_applied = bool(results.get("prefix_lock", False)) and bool(
                    results.get("verl_server_extensions", False)
                )
                print(
                    f"[CANOPY-VLLM-PATCH] apply_all_b2_patches results={results} "
                    f"all_ok={b2_patch_applied} (b1_subset_ok={b1_patch_applied})",
                    flush=True,
                )
            except Exception as exc:
                logger.warning(
                    "[CANOPY-VLLM-PATCH] apply_all_b2_patches raised: %s: %s",
                    type(exc).__name__,
                    exc,
                )

        if admission_cfg.enabled:
            # Keyed by server address, the same ids the balancer uses.
            server_handles_map: dict[str, Any] = {}
            try:
                for addr, handle in zip(
                    self.server_addresses, self.server_handles, strict=True
                ):
                    server_handles_map[addr] = handle
            except Exception as exc:
                logger.warning(
                    "[CANOPY-ADMISSION] failed to build server_handles_map: %s", exc
                )
                server_handles_map = {}

            coordinator = AdmissionCoordinator.remote(
                cfg=admission_cfg,
                lb_handle=lb_handle,
                server_handles=server_handles_map,
                vllm_dummy_prompt_ids=None,
            )
            try:
                ping_result = await coordinator.ping.remote()
            except Exception as exc:
                logger.warning(
                    "[CANOPY-ADMISSION] coordinator ping failed: %s", exc
                )
                ping_result = "ping_failed"

            try:
                await lb_handle.set_admission_coordinator.remote(coordinator)
            except Exception as exc:
                logger.warning(
                    "[CANOPY-ADMISSION] set_admission_coordinator failed: %s", exc
                )

            # Keep a reference so the actor is not garbage-collected.
            self.admission_coordinator = coordinator

            print(
                f"[CANOPY-ADMISSION] AdmissionCoordinator initialised "
                f"enabled=True mode={admission_cfg.mode} "
                f"threshold={admission_cfg.kv_pressure_threshold} "
                f"abort_mode={admission_cfg.abort_mode} "
                f"keepalive_interval_sec={admission_cfg.keepalive_interval_sec} "
                f"prefix_lock_ttl_sec={admission_cfg.prefix_lock_ttl_sec} "
                f"prefix_lock_refresh_sec={admission_cfg.prefix_lock_refresh_sec} "
                f"b1_patch_applied={b1_patch_applied} "
                f"n_servers={len(server_handles_map)} "
                f"ping={ping_result}",
                flush=True,
            )
        else:
            self.admission_coordinator = None
            print(
                "[CANOPY-ADMISSION] disabled "
                "(canopy_scheduler.admission.enabled=False)",
                flush=True,
            )
