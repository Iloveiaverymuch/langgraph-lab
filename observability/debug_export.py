"""
Diagnostic — send ONE span synchronously with verbose logging so we can see the
exact HTTP response from Langfuse (200/207 = accepted; 401 = auth; 404 = wrong
endpoint path). Unlike the normal batched exporter, this blocks and logs.

Run on your Mac:
    set -a; . ./.env.local; set +a
    python -m observability.debug_export

Copy the whole output back — especially any line showing an HTTP status code.
"""
from __future__ import annotations

import logging

# Turn on debug logging BEFORE importing the exporter, so urllib3 prints the
# request/response status line.
logging.basicConfig(level=logging.DEBUG)

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from observability.langfuse_export import _endpoint_and_headers


def main() -> None:
    endpoint, headers = _endpoint_and_headers()
    # Show endpoint + that auth is present, WITHOUT printing the secret.
    print(f"\n>>> POST endpoint: {endpoint}")
    print(f">>> Authorization header present: {'Authorization' in headers}\n")

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer("diag")
    with tracer.start_as_current_span("diag_span") as span:
        span.set_attribute("sentinel.task_id", "diag-0001")
        span.set_attribute("sentinel.user_id", "riadh")

    provider.force_flush()
    print("\n>>> Flushed. If you see an HTTP 200/207 above, Langfuse accepted it.")
    print(">>> If you see 401/403 -> keys/auth. If 404 -> endpoint path. Copy it back.")


if __name__ == "__main__":
    main()
