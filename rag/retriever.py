"""Dense retriever over the Qdrant index — the single retrieval path.

Every consumer goes through this class: the offline eval harness, the FastAPI
service, the LangGraph agent and the MCP tools. `retrieve()` is defined in terms
of `retrieve_chunks()`, so the IDs the eval scores are by construction the IDs
production serves — they cannot drift apart.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag.build_index import COLLECTION, MODEL_NAME, QDRANT_HOST, QDRANT_PORT
from rag.types import Chunk


class DenseRetriever:
    def __init__(self) -> None:
        self._model = SentenceTransformer(MODEL_NAME)
        self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def retrieve_chunks(self, query: str, k: int) -> list[Chunk]:
        """Return up to k chunks, ranked best-first. The only Qdrant query in the codebase."""
        vector = self._model.encode(query, normalize_embeddings=True).tolist()
        hits = self._client.query_points(
            collection_name=COLLECTION, query=vector, limit=k
        ).points
        return [
            Chunk(
                chunk_id=hit.payload["chunk_id"],
                # .get() for provenance: an index built before R1 Step 3 has no
                # page/section in its payload, and a stale index should degrade
                # to "no provenance", not crash the API.
                document_id=hit.payload.get("document_id", ""),
                source=hit.payload["source"],
                text=hit.payload["text"],
                score=hit.score,
                page=hit.payload.get("page"),
                section=hit.payload.get("section"),
            )
            for hit in hits
        ]

    def retrieve(self, query: str, k: int) -> list[str]:
        """Chunk IDs only — satisfies the eval harness's `Retriever` protocol.

        Deliberately derived from retrieve_chunks() rather than issuing its own
        query, so offline evaluation and runtime serving can never diverge.
        """
        return [chunk.chunk_id for chunk in self.retrieve_chunks(query, k)]
