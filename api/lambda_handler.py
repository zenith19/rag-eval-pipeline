"""AWS Lambda entry point.

Mangum adapts the same FastAPI application that `uvicorn` serves locally, so
there is one app and one code path rather than a separate Lambda variant that
drifts from the thing that was tested.

Lambda invokes this through its runtime interface client, not through the
image's CMD — which is exactly why index resolution lives in rag/config.py
rather than in a shell entrypoint (docs/decisions/006).
"""

from mangum import Mangum

from api.main import app

_asgi = Mangum(app, lifespan="off")


def handler(event, context):
    """Route a warming ping away from the ASGI app, and serve everything else.

    A cold /ask takes ~47s — importing torch and loading the embedding model —
    against a 30s ceiling on HTTP API integrations that AWS does not allow to be
    raised. The first visitor after an idle period therefore gets a 503 from the
    gateway before the function ever answers. A schedule keeps a warm container
    with the model already loaded (infra/main.tf, aws_cloudwatch_event_rule.warm).

    The ping must touch the retriever, not just the process: /health returns in
    17.8s cold because it never loads the model, so warming on /health would warm
    the wrong thing. It deliberately stops short of Bedrock — generation is not
    what is slow, and pinging it every five minutes would spend tokens to measure
    nothing.
    """
    if isinstance(event, dict) and event.get("warm"):
        from rag.generate import retrieve_chunks

        retrieve_chunks("warm", k=1)
        return {"warmed": True}

    return _asgi(event, context)
