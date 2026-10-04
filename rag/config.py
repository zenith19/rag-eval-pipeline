"""Configuration shared by the indexer and the retriever.

These constants used to live in rag/build_index.py, which meant rag/retriever.py
imported the *indexer* just to learn a hostname — dragging in sentence_transformers
(~13s) and forcing a local import inside build_index.main() to dodge the resulting
cycle. Config has no heavyweight dependencies, so both sides can import it freely.

Values are overridable by environment variable so the same image can run against
a Qdrant container locally and a different endpoint in deployment.
"""

import os

COLLECTION = os.environ.get("RAG_COLLECTION", "documents")
MODEL_NAME = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_SIZE = 384  # bge-small-en-v1.5; changing the model means changing this
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
