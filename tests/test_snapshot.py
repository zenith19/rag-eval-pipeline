"""The CI index snapshot must still describe what the gate claims to score.

The regression gate (evaluation/snapshot.py) rebuilds the index from committed
vectors. If those fell out of step with the eval set or the configuration, the gate
would score something other than production — and could pass while doing it.
"""

import json
from pathlib import Path

import numpy as np

from evaluation.eval_harness import DEFAULT_EVAL_SET, load_eval_set
from rag.config import MODEL_NAME, VECTOR_SIZE

ROOT = Path(__file__).resolve().parent.parent
SNAP = np.load(ROOT / "evaluation" / "index_snapshot.npz")


def test_every_gold_chunk_is_in_the_snapshot():
    """A gold ID missing from the index would score as a miss forever, invisibly."""
    ids = set(SNAP["chunk_ids"].tolist())
    gold = {g for ex in load_eval_set(DEFAULT_EVAL_SET) for g in ex.relevant_chunk_ids}
    assert gold <= ids


def test_the_snapshot_matches_the_configured_model_and_the_baseline_corpus():
    baseline = json.loads((ROOT / "evaluation" / "baseline.json").read_text())
    assert str(SNAP["model"]) == MODEL_NAME
    assert SNAP["vectors"].shape == (baseline["corpus"]["chunks"], VECTOR_SIZE)
    assert len(set(SNAP["chunk_ids"].tolist())) == len(SNAP["chunk_ids"])


def test_the_snapshot_carries_no_document_text():
    """It goes into a public repository; the papers' content must not."""
    assert set(SNAP.files) == {"chunk_ids", "vectors", "model"}


def test_vectors_are_normalised_as_the_index_expects():
    norms = np.linalg.norm(SNAP["vectors"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
