"""Dense retriever over the Qdrant index — the single retrieval path.

Every consumer goes through this class: the offline eval harness, the FastAPI
service, the LangGraph agent and the MCP tools. `retrieve()` is defined in terms
of `retrieve_chunks()`, so the IDs the eval scores are by construction the IDs
production serves — they cannot drift apart.
"""

from dataclasses import replace

from sentence_transformers import SentenceTransformer

from rag.config import COLLECTION, MODEL_NAME, RERANK_MODEL, RERANK_POOL, make_client
from rag.types import Chunk


class DenseRetriever:
    def __init__(self) -> None:
        self._model = SentenceTransformer(MODEL_NAME)
        self._client = make_client()

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


class CrossEncoderRetriever:
    """Dense retrieval, then a cross-encoder reorders the candidates.

    Wraps DenseRetriever rather than querying Qdrant itself. That is invariant 1 in
    CLAUDE.md and not a formality: a reranker that issued its own query would be a
    second retrieval path, and the eval harness would stop describing what the API
    serves — exactly the drift decision 001 was written about.

    Why this should help here, stated as a prediction before measuring: recall@1 is
    0.393 while recall@20 is 0.869, so most gold material is already retrieved and
    merely ordered badly. A bi-encoder compares two vectors built independently; a
    cross-encoder reads query and passage together and can use their interaction.
    Reordering is precisely the lever the gap calls for.

    Why it may not: the ceiling is recall@POOL, never higher, because this only ever
    reorders what dense retrieval returned. And it costs a forward pass per candidate
    — 50 of them — against a 30s gateway ceiling (decision 009).
    """

    def __init__(
        self,
        dense: "DenseRetriever | None" = None,
        model_name: str = RERANK_MODEL,
        pool: int = RERANK_POOL,
        model=None,
    ) -> None:
        # `model` is injectable so the ordering logic can be tested without
        # downloading 1.1 GB of cross-encoder to assert that sorting works.
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        self._dense = dense if dense is not None else DenseRetriever()
        self._model = model
        self._pool = pool

    def retrieve_chunks(self, query: str, k: int) -> list[Chunk]:
        # Never fewer candidates than results: a pool below k would silently truncate.
        candidates = self._dense.retrieve_chunks(query, max(self._pool, k))
        if not candidates:
            return []

        scores = self._model.predict([(query, c.text) for c in candidates])

        # sorted() is stable, so candidates the reranker scores equally keep the
        # dense order rather than being shuffled by an implementation detail.
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        # The returned score is the reranker's, not Qdrant's. They are different
        # quantities on different scales, and a consumer that compared one to the
        # other would be comparing nothing.
        return [replace(chunk, score=float(score)) for chunk, score in ranked[:k]]

    def retrieve(self, query: str, k: int) -> list[str]:
        """Chunk IDs only — the same protocol DenseRetriever satisfies."""
        return [chunk.chunk_id for chunk in self.retrieve_chunks(query, k)]
