"""Embed chunks and index them in Qdrant.

Querying lives in rag/retriever.py, not here — this module only builds the index.
An earlier `search()` helper here was the second of five retrieval paths in the
codebase and is deliberately gone (see docs/decisions/001-unified-retrieval-path.md).
"""

from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.config import COLLECTION, MODEL_NAME, QDRANT_HOST, QDRANT_PORT, VECTOR_SIZE
from rag.ingest import load_chunks

if TYPE_CHECKING:  # import costs ~13s (torch); only needed when actually indexing
    from sentence_transformers import SentenceTransformer


def build_index(client: QdrantClient, model: "SentenceTransformer") -> int:
    chunks = load_chunks()
    if not chunks:
        return 0

    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Qdrant point IDs must be ints or UUIDs, so the chunk_id is carried in the payload.
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
            for i in range(len(chunks))
        ],
    )
    return len(chunks)


def main() -> None:
    # Imported here, not at module scope: sentence_transformers costs ~13s, and
    # only this demo path and build_index() need it. (The circular-import reason
    # is gone now that configuration lives in rag/config.py.)
    from sentence_transformers import SentenceTransformer

    from rag.retriever import DenseRetriever

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    indexed = build_index(client, model)
    if not indexed:
        print("No chunks to index. Add PDFs to data/ and check rag/ingest.py.")
        return
    print(f"Indexed {indexed} chunks.\n")

    query = "What is language identification from audio?"
    print(f"Query: {query}\n")
    for rank, chunk in enumerate(DenseRetriever().retrieve_chunks(query, k=3), start=1):
        text = chunk.text.replace("\n", " ")
        print(f"#{rank}  {chunk.score:.3f}  {chunk.source}")
        print(f"    {text[:220]} ...\n")


if __name__ == "__main__":
    main()
