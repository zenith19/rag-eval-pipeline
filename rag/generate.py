"""Answer questions over the indexed documents using retrieved context and Amazon Bedrock."""

from functools import lru_cache

import boto3

from rag.retriever import DenseRetriever
from rag.types import Chunk

REGION = "eu-central-1"
# Frankfurt invokes models through a regional inference profile (the "eu." prefix).
BEDROCK_MODEL_ID = "eu.amazon.nova-lite-v1:0"

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Keep the answer concise."
)


# Loaded once and reused across requests (the embedding model load dominates startup).
@lru_cache(maxsize=1)
def _retriever() -> DenseRetriever:
    return DenseRetriever()


@lru_cache(maxsize=1)
def _bedrock():
    return boto3.client("bedrock-runtime", region_name=REGION)


def _format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)


def retrieve_chunks(question: str, k: int = 3) -> list[Chunk]:
    """Retrieve the top-k chunks for a question (kept separate for graph orchestration).

    Goes through the same DenseRetriever the eval harness scores, so a retrieval
    improvement measured offline reaches this path by construction.
    """
    return _retriever().retrieve_chunks(question, k)


def generate_from_chunks(question: str, chunks: list[Chunk]) -> str:
    """Generate a grounded answer from already-retrieved chunks."""
    response = _bedrock().converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": f"Context:\n{_format_context(chunks)}\n\nQuestion: {question}"}
                ],
            }
        ],
        inferenceConfig={"maxTokens": 512, "temperature": 0.2},
    )
    return response["output"]["message"]["content"][0]["text"]


def answer_question(question: str, k: int = 3) -> tuple[str, list[str]]:
    chunks = retrieve_chunks(question, k)
    answer = generate_from_chunks(question, chunks)
    return answer, [c.chunk_id for c in chunks]


def main() -> None:
    question = "What is the BenCoref dataset?"
    answer, sources = answer_question(question)
    print(f"Q: {question}\n")
    print(answer)
    print(f"\nRetrieved chunks: {sources}")


if __name__ == "__main__":
    main()
