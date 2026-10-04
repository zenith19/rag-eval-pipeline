"""Every gold label must point at a chunk that actually exists in the index.

Without this, a re-index can silently orphan labels and the eval quietly scores
against chunks that are no longer there. Guards R1 Step 3's corpus expansion.
"""

from pathlib import Path

import pytest

from evaluation.eval_harness import load_eval_set

# Anchored to the repo, not the working directory: the suite must pass from
# anywhere, including a CI runner that invokes pytest from outside the root.
EVAL_SET = Path(__file__).resolve().parent.parent / "evaluation" / "eval_set.jsonl"


@pytest.mark.integration
def test_every_labelled_chunk_id_exists_in_the_index():
    # Index inspection (a scroll over stored payloads), not retrieval — the
    # ranked query path stays exclusively in rag/retriever.py.
    from qdrant_client import QdrantClient

    from rag.config import COLLECTION, QDRANT_HOST, QDRANT_PORT

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    indexed_ids = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, limit=512, offset=offset, with_payload=True
        )
        indexed_ids.update(p.payload["chunk_id"] for p in points)
        if offset is None:
            break

    labelled = {cid for ex in load_eval_set(EVAL_SET) for cid in ex.relevant_chunk_ids}
    orphaned = labelled - indexed_ids
    assert not orphaned, f"{len(orphaned)} gold label(s) point at missing chunks: {sorted(orphaned)}"
