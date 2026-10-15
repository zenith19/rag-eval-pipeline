"""Fault injection on the generation path. No AWS, no Qdrant, no model.

Every test here answers the same question: when something upstream breaks, does a
visitor get a readable status or a bare 500? Before this file the answer was 500
for every case, including ones where retrying a moment later would have worked.
"""

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError
from fastapi.testclient import TestClient

from rag.errors import PermanentGenerationError, TransientGenerationError
from rag.types import Chunk


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="c1",
        document_id="doc",
        source="paper.pdf",
        text="Kuhn-Munkres finds the optimal alignment.",
        score=0.9,
        page=2,
        section="1 Introduction",
    )


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "injected"}}, "Converse")


class _Bedrock:
    """Raises a given exception, or returns a given response, and counts calls."""

    def __init__(self, raises=None, response=None):
        self._raises = raises
        self._response = response
        self.calls = 0

    def converse(self, **_kwargs):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._response


@pytest.fixture
def bedrock(monkeypatch):
    """Swap the cached Bedrock client for a stub, per test."""

    def _install(stub):
        import rag.generate as gen

        gen._bedrock.cache_clear()
        monkeypatch.setattr(gen, "_bedrock", lambda: stub)
        return stub

    return _install


# --------------------------------------------------------------------------- #
# Classification: does the failure say whether retrying is worth anything?
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "code",
    ["ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException"],
)
def test_retryable_bedrock_errors_are_transient(bedrock, code):
    from rag.generate import generate_from_chunks

    bedrock(_Bedrock(raises=_client_error(code)))
    with pytest.raises(TransientGenerationError):
        generate_from_chunks("q", [_chunk()])


@pytest.mark.parametrize(
    "code",
    ["AccessDeniedException", "ValidationException", "ResourceNotFoundException"],
)
def test_non_retryable_bedrock_errors_are_permanent(bedrock, code):
    # The distinction that matters: retrying one of these spends the caller's
    # remaining seconds to fail in exactly the same way.
    from rag.generate import generate_from_chunks

    bedrock(_Bedrock(raises=_client_error(code)))
    with pytest.raises(PermanentGenerationError):
        generate_from_chunks("q", [_chunk()])


@pytest.mark.parametrize("exc", [ConnectTimeoutError(endpoint_url="x"), ReadTimeoutError(endpoint_url="x")])
def test_timeouts_are_transient_not_crashes(bedrock, exc):
    # Timeouts arrive as BotoCoreError, not ClientError — no HTTP response came
    # back to carry an error code — so they need their own branch.
    from rag.generate import generate_from_chunks

    bedrock(_Bedrock(raises=exc))
    with pytest.raises(TransientGenerationError):
        generate_from_chunks("q", [_chunk()])


def test_malformed_success_is_permanent(bedrock):
    # A 200 with the wrong shape is not worth retrying: the same request would
    # produce the same unusable body.
    from rag.generate import generate_from_chunks

    bedrock(_Bedrock(response={"output": {}}))
    with pytest.raises(PermanentGenerationError):
        generate_from_chunks("q", [_chunk()])


def test_no_chunks_answers_without_calling_the_model(bedrock):
    # Grounding is the project's claim; asking the model to answer from an empty
    # context invites it to invent one.
    from rag.generate import NO_EVIDENCE_ANSWER, generate_from_chunks

    stub = bedrock(_Bedrock(raises=AssertionError("model must not be called")))
    assert generate_from_chunks("q", []) == NO_EVIDENCE_ANSWER
    assert stub.calls == 0


# --------------------------------------------------------------------------- #
# What a visitor actually sees
# --------------------------------------------------------------------------- #

@pytest.fixture
def api(monkeypatch):
    """A TestClient whose retrieval is stubbed, so only generation can fail."""
    import rag.generate as gen

    monkeypatch.setattr(gen, "retrieve_chunks", lambda q, k: [_chunk()])
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


def test_throttling_surfaces_as_503_with_retry_after(api, bedrock):
    bedrock(_Bedrock(raises=_client_error("ThrottlingException")))
    r = api.post("/ask", json={"question": "which algorithm aligns entities?"})
    assert r.status_code == 503
    assert r.headers.get("Retry-After") == "5"
    assert "try again" in r.json()["detail"].lower()


def test_permission_failure_surfaces_as_502_not_500(api, bedrock):
    # 502 because this service is working and its upstream is not. A 500 would
    # say the fault is here, and would tell the caller nothing about retrying.
    bedrock(_Bedrock(raises=_client_error("AccessDeniedException")))
    r = api.post("/ask", json={"question": "which algorithm aligns entities?"})
    assert r.status_code == 502
    assert "will not help" in r.json()["detail"]


def test_a_working_request_is_unaffected(api, bedrock):
    bedrock(
        _Bedrock(
            response={"output": {"message": {"content": [{"text": "Kuhn-Munkres."}]}}}
        )
    )
    r = api.post("/ask", json={"question": "which algorithm aligns entities?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Kuhn-Munkres."
    assert body["sources"][0]["citation"] == "paper.pdf p.2"


def test_timeout_budget_fits_inside_the_gateway_ceiling():
    """The retry budget must finish before API Gateway stops waiting.

    If it overruns, the visitor gets the gateway's 504 instead of this service's
    503 — no Retry-After, no explanation — which would quietly undo the point of
    classifying failures at all. An earlier setting (connect 3, read 8, 3 attempts)
    came to 33s against a 30s ceiling because it counted the reads and forgot the
    backoff between them.

    The ceiling is read from infra/main.tf rather than hardcoded, so lowering it
    there fails here instead of in production.
    """
    import re
    from pathlib import Path

    from rag.config import (
        BEDROCK_CONNECT_TIMEOUT,
        BEDROCK_MAX_ATTEMPTS,
        BEDROCK_READ_TIMEOUT,
    )

    tf = (Path(__file__).resolve().parent.parent / "infra" / "main.tf").read_text()
    match = re.search(r"timeout_milliseconds\s*=\s*(\d+)", tf)
    assert match, "no integration timeout found in infra/main.tf"
    ceiling_s = int(match.group(1)) / 1000

    # botocore "standard" mode: full-jitter exponential backoff, capped at 20s
    backoff = sum(min(2**i, 20) for i in range(1, BEDROCK_MAX_ATTEMPTS))
    worst_case = BEDROCK_CONNECT_TIMEOUT + BEDROCK_READ_TIMEOUT * BEDROCK_MAX_ATTEMPTS + backoff

    # Strictly less, with room for retrieval and request overhead on the same clock.
    assert worst_case < ceiling_s * 0.8, (
        f"Bedrock budget {worst_case}s leaves too little of the {ceiling_s}s ceiling"
    )
