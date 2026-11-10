# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of canopy_verl.algorithm.tree_metadata (pure Python)."""

from __future__ import annotations

import unittest

from canopy_verl.algorithm.tree_metadata import (
    BranchDecision,
    TreeState,
    derive_shared_tree_uid,
)


class TestTreeState(unittest.TestCase):
    def test_record_decision_updates_branch_points(self):
        ts = TreeState(tree_uid="u" * 8, node_uid="n" * 8)
        d1 = BranchDecision(fork=False, branch_count=0, reason="no_fork_turn1")
        d2 = BranchDecision(fork=True, branch_count=1, reason="fork_turn2")
        ts.record_decision(d1, turn_idx=1)
        ts.record_decision(d2, turn_idx=2)
        self.assertEqual(len(ts.branch_points), 2)
        self.assertEqual(ts.forks_decided, 1)
        self.assertFalse(ts.branch_points[0]["fork"])
        self.assertTrue(ts.branch_points[1]["fork"])

    def test_to_canopy_tree_fields_schema(self):
        ts = TreeState(tree_uid="a" * 16, node_uid="b" * 16, depth=2,
                       extra={"canopy_bucket": 1, "canopy_rollout_n": 3})
        fields = ts.to_canopy_tree_fields()
        expected = {
            "tree_uid", "node_uid", "parent_node_uid", "depth", "branch_points", "entropy_history",
            "turn_end_times", "branches_issued", "forks_decided", "live_forks_issued", "live_forks_failed",
            "live_fork_elapsed_sum", "live_fork_discarded_len_sum", "branch_points_count",
            "turn_end_times_count", "bucket", "rollout_n", "rollout_n_in_bucket", "extra",
        }
        self.assertEqual(set(fields), {"canopy_tree/" + k for k in expected})
        self.assertEqual(fields["canopy_tree/depth"], 2)
        self.assertEqual(fields["canopy_tree/branch_points_count"], 0)
        self.assertEqual(fields["canopy_tree/bucket"], 1)
        self.assertEqual(fields["canopy_tree/rollout_n"], 3)
        self.assertEqual(fields["canopy_tree/rollout_n_in_bucket"], -1)


class TestDeriveSharedTreeUid(unittest.TestCase):
    """All rollout.n coroutines of a prompt (same sample_index) share one tree_uid."""

    def test_deterministic_same_sample_index_same_uid(self):
        uid1 = derive_shared_tree_uid("prompt-42")
        uid2 = derive_shared_tree_uid("prompt-42")
        self.assertEqual(uid1, uid2)
        self.assertEqual(len(uid1), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in uid1))

    def test_different_sample_index_different_uid(self):
        self.assertNotEqual(derive_shared_tree_uid("prompt-1"), derive_shared_tree_uid("prompt-2"))

    def test_int_and_str_sample_index_agree(self):
        self.assertEqual(derive_shared_tree_uid(42), derive_shared_tree_uid("42"))

    def test_none_falls_back_to_random_uid(self):
        uid1 = derive_shared_tree_uid(None)
        uid2 = derive_shared_tree_uid(None)
        self.assertNotEqual(uid1, uid2)
        self.assertEqual(len(uid1), 32)

    def test_none_with_explicit_fallback(self):
        self.assertEqual(derive_shared_tree_uid(None, fallback="explicit-fallback-uid"), "explicit-fallback-uid")

    def test_siblings_of_one_prompt_share_uid(self):
        uids = [derive_shared_tree_uid("prompt-xyz") for _ in range(4)]
        self.assertEqual(len(set(uids)), 1, f"siblings split: {uids}")
        self.assertNotIn(derive_shared_tree_uid("prompt-abc"), uids)


if __name__ == "__main__":
    unittest.main()
