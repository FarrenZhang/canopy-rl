# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Metrics hook that drives the admission coordinator (see :mod:`.prom_hook`)."""

from canopy_verl.observability.prom_hook import maybe_scrape, scrape_kv_pressure

__all__ = ["maybe_scrape", "scrape_kv_pressure"]
