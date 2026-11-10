# Copyright 2026 The Canopy Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-rollout tree metadata (``BranchDecision``, ``TreeState``) and the shared ``tree_uid``."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4


def derive_shared_tree_uid(
    sample_index: int | str | None,
    fallback: Optional[str] = None,
) -> str:
    """Deterministic ``tree_uid`` shared by all coroutines of one prompt.

    Tree-aware placement co-locates requests that carry the same ``tree_uid``.
    verl runs ``rollout.n`` agent-loop coroutines per dataset row, all with the
    same ``sample_index`` (passed to agent loops as ``_canopy_sample_index``,
    see ``integrations/verl/patches``).  Hashing it gives every coroutine of a
    prompt the same ``tree_uid``, so the whole group is placed as one tree.

    Args:
        sample_index: dataset-level prompt identifier; ``None`` selects the
            fallback.
        fallback: ``tree_uid`` to return when ``sample_index`` is ``None``
            (default: a fresh ``uuid4().hex``, i.e. a tree of its own).

    Returns:
        The first 16 hex digits of ``md5(str(sample_index))``.
    """
    if sample_index is None:
        return fallback if fallback is not None else uuid4().hex
    return hashlib.md5(str(sample_index).encode("utf-8")).hexdigest()[:16]


@dataclass
class BranchDecision:
    """Outcome of a per-turn branching decision.

    Tree-rollout agent loops emit one decision per assistant turn and record
    every decision (``fork=False`` included, so ``len(branch_points)`` equals
    the number of turns) in ``TreeState.branch_points``.

    Attributes:
        fork: whether a side branch is created at this turn boundary.
        branch_count: number of sibling branches to create (0 when ``fork`` is
            False).
        reason: short tag explaining the decision.
        metadata: algorithm-specific payload (e.g. entropy deltas), copied into
            ``branch_points``.
    """

    fork: bool = False
    branch_count: int = 0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeState:
    """Per-rollout tree metadata; one instance per agent-loop coroutine.

    Attributes:
        tree_uid: tree this rollout belongs to (shared by its siblings).
        node_uid: this leaf; unique per rollout.
        parent_node_uid: node a side branch forked from; ``None`` for the main
            chain.
        depth: branching depth from the root (main chain = 0).
        branch_points: one entry per assistant turn, forked or not.
        entropy_history: per-turn entropy (empty for algorithms that do not
            compute it).
        turn_end_times: end of each turn, in seconds since the rollout started.
        extra: algorithm-specific counters.
    """

    tree_uid: str
    node_uid: str
    parent_node_uid: Optional[str] = None
    depth: int = 0
    branch_points: list[dict[str, Any]] = field(default_factory=list)
    entropy_history: list[float] = field(default_factory=list)
    turn_end_times: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    # Counters maintained by the agent loop; trainer-side metric aggregation
    # reduces them by name.
    branches_issued: int = 0
    forks_decided: int = 0
    live_forks_issued: int = 0
    live_forks_failed: int = 0
    live_fork_elapsed_sum: float = 0.0      # seconds spent awaiting side forks
    live_fork_discarded_len_sum: int = 0    # cumulative discarded token count

    def record_decision(self, decision: BranchDecision, turn_idx: int) -> None:
        """Append a decision to ``branch_points`` (one per turn)."""
        self.branch_points.append({
            "turn_idx": turn_idx,
            "fork": decision.fork,
            "branch_count": decision.branch_count,
            "reason": decision.reason,
            "metadata": dict(decision.metadata),
        })
        self.forks_decided += 1 if decision.fork else 0

    def to_canopy_tree_fields(self) -> dict[str, Any]:
        """Serialize to ``canopy_tree/*`` keys for ``AgentLoopOutput.extra_fields``.

        ``bucket``, ``rollout_n`` and ``rollout_n_in_bucket`` are read from
        ``extra`` (``canopy_bucket``, ``canopy_rollout_n``,
        ``canopy_rollout_n_in_bucket``; -1 when absent).
        """
        return {
            "canopy_tree/tree_uid": self.tree_uid,
            "canopy_tree/node_uid": self.node_uid,
            "canopy_tree/parent_node_uid": self.parent_node_uid,
            "canopy_tree/depth": self.depth,
            "canopy_tree/branch_points": self.branch_points,
            "canopy_tree/entropy_history": self.entropy_history,
            "canopy_tree/turn_end_times": self.turn_end_times,
            "canopy_tree/branches_issued": self.branches_issued,
            "canopy_tree/forks_decided": self.forks_decided,
            "canopy_tree/live_forks_issued": self.live_forks_issued,
            "canopy_tree/live_forks_failed": self.live_forks_failed,
            "canopy_tree/live_fork_elapsed_sum": self.live_fork_elapsed_sum,
            "canopy_tree/live_fork_discarded_len_sum": self.live_fork_discarded_len_sum,
            "canopy_tree/branch_points_count": len(self.branch_points),
            "canopy_tree/turn_end_times_count": len(self.turn_end_times),
            "canopy_tree/bucket": self.extra.get("canopy_bucket", -1),
            "canopy_tree/rollout_n": self.extra.get("canopy_rollout_n", -1),
            "canopy_tree/rollout_n_in_bucket": self.extra.get("canopy_rollout_n_in_bucket", -1),
            "canopy_tree/extra": dict(self.extra),
        }
