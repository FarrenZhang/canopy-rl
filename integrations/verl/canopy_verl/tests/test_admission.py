# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Admission config, pressure detection, tree ranking and monitor (pure Python)."""

from __future__ import annotations

import unittest

from canopy_verl.scheduler.admission import (
    AdmissionConfig,
    AdmissionMonitor,
    KVPressureSnapshot,
    TreeValueSnapshot,
    build_keepalive_prompt,
    detect_kv_pressure,
    rank_trees_for_abort,
)


class TestAdmissionConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AdmissionConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.mode, "a")
        self.assertAlmostEqual(cfg.kv_pressure_threshold, 0.85)
        self.assertEqual(cfg.keepalive_interval_sec, 30)
        self.assertTrue(cfg.abort_low_value)
        self.assertEqual(cfg.probe_interval_sec, 5)

    def test_from_hydra_none(self):
        cfg = AdmissionConfig.from_hydra(None)
        self.assertFalse(cfg.enabled)

    def test_from_hydra_dict_like(self):
        class Fake:
            pass

        outer = Fake()
        outer.admission = _FakeAdmission(
            enabled=True, mode="b1", kv_pressure_threshold=0.75
        )
        cfg = AdmissionConfig.from_hydra(outer)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.mode, "b1")
        self.assertAlmostEqual(cfg.kv_pressure_threshold, 0.75)

    def test_validate_valid(self):
        self.assertEqual(AdmissionConfig().validate(), [])

    def test_validate_bad_mode(self):
        errs = AdmissionConfig(mode="c").validate()
        self.assertTrue(any("mode must be a/b1/b2" in e for e in errs))

    def test_validate_bad_threshold(self):
        errs = AdmissionConfig(kv_pressure_threshold=1.5).validate()
        self.assertTrue(any("kv_pressure_threshold" in e for e in errs))

    def test_validate_bad_interval(self):
        errs = AdmissionConfig(probe_interval_sec=0).validate()
        self.assertTrue(any("probe_interval_sec" in e for e in errs))


class _FakeAdmission:
    """Mimics an OmegaConf node: ``.get()`` and attribute access."""

    def __init__(self, **fields):
        self._fields = fields

    def get(self, key, default):
        return self._fields.get(key, default)

    def __getattr__(self, key):
        if key == "_fields":
            raise AttributeError(key)
        return self._fields.get(key)


class TestDetectKVPressure(unittest.TestCase):
    def test_disabled_returns_no_action(self):
        cfg = AdmissionConfig(enabled=False, kv_pressure_threshold=0.5)
        snap = KVPressureSnapshot(gpu_cache_usage_perc=0.99, num_preempted_total=100)
        d = detect_kv_pressure(snap, cfg, preempt_delta=5)
        self.assertFalse(d.under_pressure)
        self.assertFalse(d.should_keepalive)
        self.assertFalse(d.should_abort_low_value)
        self.assertEqual(d.reason, "admission_disabled")

    def test_no_pressure_when_below_threshold_and_no_new_preempts(self):
        cfg = AdmissionConfig(enabled=True, kv_pressure_threshold=0.85)
        snap = KVPressureSnapshot(gpu_cache_usage_perc=0.5, num_preempted_total=0)
        d = detect_kv_pressure(snap, cfg, preempt_delta=0)
        self.assertFalse(d.under_pressure)
        self.assertEqual(d.reason, "no_pressure")

    def test_pressure_via_usage_threshold(self):
        cfg = AdmissionConfig(enabled=True, kv_pressure_threshold=0.8)
        snap = KVPressureSnapshot(gpu_cache_usage_perc=0.9, num_preempted_total=0)
        d = detect_kv_pressure(snap, cfg, preempt_delta=0)
        self.assertTrue(d.under_pressure)
        self.assertTrue(d.should_keepalive)
        self.assertTrue(d.should_abort_low_value)
        self.assertEqual(d.reason, "usage_above_threshold")

    def test_pressure_via_new_preemptions(self):
        cfg = AdmissionConfig(enabled=True, kv_pressure_threshold=0.95)
        snap = KVPressureSnapshot(gpu_cache_usage_perc=0.3, num_preempted_total=5)
        d = detect_kv_pressure(snap, cfg, preempt_delta=5)
        self.assertTrue(d.under_pressure)
        self.assertEqual(d.reason, "new_preemptions")

    def test_pressure_both_signals(self):
        cfg = AdmissionConfig(enabled=True, kv_pressure_threshold=0.8)
        snap = KVPressureSnapshot(gpu_cache_usage_perc=0.9, num_preempted_total=7)
        d = detect_kv_pressure(snap, cfg, preempt_delta=7)
        self.assertTrue(d.under_pressure)
        self.assertEqual(d.reason, "usage_high+preempt")

    def test_abort_low_value_respects_config(self):
        cfg = AdmissionConfig(
            enabled=True, kv_pressure_threshold=0.8, abort_low_value=False
        )
        snap = KVPressureSnapshot(gpu_cache_usage_perc=0.9)
        d = detect_kv_pressure(snap, cfg, preempt_delta=0)
        self.assertTrue(d.should_keepalive)
        self.assertFalse(d.should_abort_low_value)

    def test_exact_threshold_is_not_pressure(self):
        cfg = AdmissionConfig(enabled=True, kv_pressure_threshold=0.85)
        snap = KVPressureSnapshot(gpu_cache_usage_perc=0.85)
        d = detect_kv_pressure(snap, cfg, preempt_delta=0)
        self.assertFalse(d.under_pressure)


class TestRankTreesForAbort(unittest.TestCase):
    def test_most_complete_first(self):
        trees = [
            TreeValueSnapshot(tree_uid="a", complete_siblings=1, active_siblings=3),
            TreeValueSnapshot(tree_uid="b", complete_siblings=4, active_siblings=0),
            TreeValueSnapshot(tree_uid="c", complete_siblings=2, active_siblings=2),
        ]
        ranked = rank_trees_for_abort(trees, top_k=3)
        self.assertEqual([t.tree_uid for t in ranked], ["b", "c", "a"])

    def test_tie_break_prefers_larger_kv(self):
        trees = [
            TreeValueSnapshot(tree_uid="x", complete_siblings=2, active_siblings=2, kv_blocks_approx=100),
            TreeValueSnapshot(tree_uid="y", complete_siblings=2, active_siblings=2, kv_blocks_approx=500),
            TreeValueSnapshot(tree_uid="z", complete_siblings=2, active_siblings=2, kv_blocks_approx=50),
        ]
        ranked = rank_trees_for_abort(trees, top_k=3)
        self.assertEqual([t.tree_uid for t in ranked], ["y", "x", "z"])

    def test_top_k_subset(self):
        trees = [
            TreeValueSnapshot(tree_uid=f"t{i}", complete_siblings=i, active_siblings=10 - i)
            for i in range(5)
        ]
        ranked = rank_trees_for_abort(trees, top_k=2)
        self.assertEqual([t.tree_uid for t in ranked], ["t4", "t3"])

    def test_empty_input(self):
        self.assertEqual(rank_trees_for_abort([], top_k=3), [])

    def test_zero_k(self):
        trees = [TreeValueSnapshot(tree_uid="a", complete_siblings=1)]
        self.assertEqual(rank_trees_for_abort(trees, top_k=0), [])

    def test_zero_total_siblings(self):
        trees = [TreeValueSnapshot(tree_uid="a")]
        ranked = rank_trees_for_abort(trees, top_k=1)
        self.assertEqual([t.tree_uid for t in ranked], ["a"])


class TestBuildKeepalivePrompt(unittest.TestCase):
    def test_returns_prefix_as_new_list(self):
        prefix = [1, 2, 3, 4, 5]
        out = build_keepalive_prompt(prefix)
        self.assertEqual(out, prefix)
        self.assertIsNot(out, prefix)

    def test_empty_prefix(self):
        self.assertEqual(build_keepalive_prompt([]), [])


class TestAdmissionMonitor(unittest.TestCase):
    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            AdmissionMonitor(AdmissionConfig(mode="bad"))

    def test_first_tick_no_delta(self):
        mon = AdmissionMonitor(
            AdmissionConfig(enabled=True, kv_pressure_threshold=0.8)
        )
        d = mon.tick(KVPressureSnapshot(gpu_cache_usage_perc=0.5, num_preempted_total=10))
        self.assertFalse(d.under_pressure)

    def test_delta_accumulates(self):
        mon = AdmissionMonitor(
            AdmissionConfig(enabled=True, kv_pressure_threshold=0.95)
        )
        mon.tick(KVPressureSnapshot(gpu_cache_usage_perc=0.3, num_preempted_total=5))
        d = mon.tick(KVPressureSnapshot(gpu_cache_usage_perc=0.3, num_preempted_total=8))
        self.assertTrue(d.under_pressure)
        self.assertEqual(d.reason, "new_preemptions")

    def test_counters(self):
        mon = AdmissionMonitor(
            AdmissionConfig(enabled=True, kv_pressure_threshold=0.5)
        )
        mon.tick(KVPressureSnapshot(gpu_cache_usage_perc=0.9))
        mon.tick(KVPressureSnapshot(gpu_cache_usage_perc=0.3))
        mon.tick(KVPressureSnapshot(gpu_cache_usage_perc=0.95))
        m = mon.get_metrics()
        self.assertEqual(m["admission/total_ticks"], 3)
        self.assertEqual(m["admission/pressure_ticks"], 2)
        self.assertAlmostEqual(m["admission/pressure_ratio"], 2 / 3)
        self.assertEqual(m["admission/keepalive_dispatched"], 2)
        self.assertEqual(m["admission/aborts_dispatched"], 2)

    def test_reset(self):
        mon = AdmissionMonitor(
            AdmissionConfig(enabled=True, kv_pressure_threshold=0.5)
        )
        mon.tick(KVPressureSnapshot(gpu_cache_usage_perc=0.9))
        mon.reset()
        m = mon.get_metrics()
        self.assertEqual(m["admission/total_ticks"], 0)
        self.assertEqual(m["admission/pressure_ticks"], 0)

    def test_preempt_counter_decrease_is_not_pressure(self):
        """A counter reset (e.g. engine restart) clamps the delta to 0."""
        mon = AdmissionMonitor(
            AdmissionConfig(enabled=True, kv_pressure_threshold=0.99)
        )
        mon.tick(KVPressureSnapshot(num_preempted_total=10))
        d = mon.tick(KVPressureSnapshot(num_preempted_total=0))
        self.assertFalse(d.under_pressure)


if __name__ == "__main__":
    unittest.main()
