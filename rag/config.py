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

# If set, Qdrant runs embedded against this directory instead of talking to a
# server. That is how the deployed image works: the index (9.9 MB for 1,688
# chunks) is baked in at build time, so there is no database to run or pay for.
QDRANT_PATH = os.environ.get("QDRANT_PATH")

# Embedded Qdrant writes a .lock beside its data, so the directory must be
# writable. The image bakes the index into a read-only location and names it
# here; it is copied once, on demand, to somewhere writable.
QDRANT_BAKED_INDEX = os.environ.get("QDRANT_BAKED_INDEX")
QDRANT_WRITABLE_INDEX = os.environ.get("QDRANT_WRITABLE_INDEX", "/tmp/index")


def index_path() -> str | None:
    """Where the embedded index lives, or None when a Qdrant server is in use.

    Deliberately resolved in Python rather than by a shell entrypoint exporting a
    variable: an entrypoint only sets the environment of the process it execs, so
    `docker exec`, a debugging shell, the MCP server and a Lambda handler would
    all silently fall back to looking for a server on localhost.
    """
    if QDRANT_PATH:
        return QDRANT_PATH
    if not QDRANT_BAKED_INDEX:
        return None

    import shutil

    if not os.path.isdir(QDRANT_WRITABLE_INDEX):
        staging = f"{QDRANT_WRITABLE_INDEX}.{os.getpid()}"
        shutil.copytree(QDRANT_BAKED_INDEX, staging)
        try:
            os.rename(staging, QDRANT_WRITABLE_INDEX)
        except OSError:
            # Another process got there first; its copy is just as good.
            shutil.rmtree(staging, ignore_errors=True)
    return QDRANT_WRITABLE_INDEX


def make_client():
    """The one place that decides between a Qdrant server and an embedded index.

    qdrant_client is imported here rather than at module scope so that reading a
    hostname stays as cheap as it should be (see docs/decisions/005).
    """
    from qdrant_client import QdrantClient

    path = index_path()
    if path:
        return QdrantClient(path=path)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
