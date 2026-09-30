"""Shared retrieval types.

`Chunk` is the single currency between retrieval and everything downstream
(generation, the agent, the MCP tools). Keeping it here rather than in the
retriever avoids import cycles and stops the Qdrant payload shape leaking into
callers — before this, four modules unpacked `hit.payload[...]` themselves.

Provenance fields (document_id, page, section) arrive in R1 Step 3, together
with the ingestion change that actually produces them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One retrieved chunk, ranked."""

    chunk_id: str
    source: str
    text: str
    score: float
