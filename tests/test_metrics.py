"""Unit tests for the from-scratch retrieval metrics. No infrastructure needed."""

from evaluation.eval_harness import hit_rate_at_k, recall_at_k, reciprocal_rank


def test_recall_counts_all_relevant_not_just_one():
    # The distinction the harness exists to make: 1 of 2 gold chunks found = 0.5,
    # even though a hit-rate metric would call this a perfect result.
    retrieved = ["a", "x", "y"]
    relevant = frozenset({"a", "b"})
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0


def test_recall_at_k_respects_the_cutoff():
    retrieved = ["x", "y", "a"]
    relevant = frozenset({"a"})
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_hit_rate_is_binary():
    relevant = frozenset({"a", "b"})
    assert hit_rate_at_k(["a", "b"], relevant, 2) == 1.0
    assert hit_rate_at_k(["a", "z"], relevant, 2) == 1.0
    assert hit_rate_at_k(["z", "w"], relevant, 2) == 0.0


def test_reciprocal_rank_is_one_indexed():
    relevant = frozenset({"a"})
    assert reciprocal_rank(["a", "x"], relevant) == 1.0
    assert reciprocal_rank(["x", "a"], relevant) == 0.5
    assert reciprocal_rank(["x", "y", "a"], relevant) == 1 / 3
    assert reciprocal_rank(["x", "y"], relevant) == 0.0


def test_empty_relevant_set_scores_zero_not_crash():
    assert recall_at_k(["a"], frozenset(), 1) == 0.0
    assert hit_rate_at_k(["a"], frozenset(), 1) == 0.0
