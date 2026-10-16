"""Answer questions over the indexed documents using retrieved context and Amazon Bedrock."""

from functools import lru_cache

import boto3

from rag.config import (
    AWS_REGION,
    BEDROCK_CONNECT_TIMEOUT,
    BEDROCK_MAX_ATTEMPTS,
    BEDROCK_MODEL_ID,
    BEDROCK_READ_TIMEOUT,
)
from rag.errors import (
    RETRYABLE_ERROR_CODES,
    PermanentGenerationError,
    TransientGenerationError,
)
from rag.retriever import DenseRetriever
from rag.types import Chunk

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Keep the answer concise."
)


# Loaded once and reused across requests (the embedding model load dominates startup).
@lru_cache(maxsize=1)
def _retriever() -> DenseRetriever:
    return DenseRetriever()


# Said once, where the answer is produced, so it cannot be mistaken for a model
# refusing to answer. "I could not find anything" is a true statement about
# retrieval; asking the model to answer from an empty context is an invitation to
# invent one, and grounding is this project's entire claim.
NO_EVIDENCE_ANSWER = (
    "I could not find anything in the indexed papers that answers that question."
)


@lru_cache(maxsize=1)
def _bedrock():
    from botocore.config import Config

    # Bounded retry with exponential backoff, applied by botocore only to errors
    # AWS marks retryable. "standard" rather than "adaptive": adaptive adds
    # client-side rate limiting tuned for sustained throughput, which is the wrong
    # shape for a demo that is idle and then answers one question.
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        config=Config(
            connect_timeout=BEDROCK_CONNECT_TIMEOUT,
            read_timeout=BEDROCK_READ_TIMEOUT,
            retries={"max_attempts": BEDROCK_MAX_ATTEMPTS, "mode": "standard"},
        ),
    )


def _format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)


def retrieve_chunks(question: str, k: int = 3) -> list[Chunk]:
    """Retrieve the top-k chunks for a question (kept separate for graph orchestration).

    Goes through the same DenseRetriever the eval harness scores, so a retrieval
    improvement measured offline reaches this path by construction.
    """
    return _retriever().retrieve_chunks(question, k)


def generate_from_chunks(question: str, chunks: list[Chunk]) -> str:
    """Generate a grounded answer from already-retrieved chunks.

    Raises TransientGenerationError or PermanentGenerationError rather than letting
    a botocore ClientError escape, so callers decide what to do about a failure
    without importing AWS vocabulary to interpret it.
    """
    if not chunks:
        return NO_EVIDENCE_ANSWER

    from botocore.exceptions import BotoCoreError, ClientError

    try:
        response = _bedrock().converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": f"Context:\n{_format_context(chunks)}"
                            f"\n\nQuestion: {question}"
                        }
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 512, "temperature": 0.2},
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        # botocore has already retried the retryable ones to exhaustion by here;
        # the distinction still matters, because it tells the caller whether
        # trying again LATER is worth anything.
        if code in RETRYABLE_ERROR_CODES:
            raise TransientGenerationError(f"Bedrock is unavailable ({code})") from exc
        raise PermanentGenerationError(f"Bedrock rejected the request ({code})") from exc
    except BotoCoreError as exc:
        # Connect and read timeouts land here, not in ClientError: no HTTP response
        # came back to carry an error code.
        raise TransientGenerationError(f"Bedrock did not respond ({type(exc).__name__})") from exc

    try:
        return response["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError) as exc:
        # A 200 whose shape is wrong is not retryable: the same request would
        # produce the same unusable body.
        raise PermanentGenerationError("Bedrock returned an unexpected response shape") from exc


def answer_question(question: str, k: int = 3) -> tuple[str, list[Chunk]]:
    """Return the answer and the chunks it was grounded in.

    The chunks rather than their IDs, so callers can cite a page without going
    back to the store for something retrieval already had in hand.
    """
    chunks = retrieve_chunks(question, k)
    answer = generate_from_chunks(question, chunks)
    return answer, chunks


def main() -> None:
    question = "What is the BenCoref dataset?"
    answer, chunks = answer_question(question)
    print(f"Q: {question}\n")
    print(answer)
    print("\nSources:")
    for c in chunks:
        print(f"  {c.cite()}")


if __name__ == "__main__":
    main()
