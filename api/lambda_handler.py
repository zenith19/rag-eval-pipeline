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

handler = Mangum(app, lifespan="off")
