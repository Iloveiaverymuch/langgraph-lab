"""
ASGI app that serves the Inngest function.

    uvicorn durable.app:app --port 8000
    inngest dev -u http://localhost:8000/api/inngest        (dev server + UI on :8288)
"""
import fastapi
import inngest.fast_api

from .backends import configure_backends
from .workflow import client, run_research

SEARCH_BACKEND = configure_backends()

app = fastapi.FastAPI()
inngest.fast_api.serve(app, client, [run_research])
