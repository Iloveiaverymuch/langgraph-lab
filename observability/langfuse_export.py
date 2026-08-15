"""
Export Sentinel receipts to Langfuse over OpenTelemetry (OTLP/HTTP).

We send spans straight to Langfuse's OTLP endpoint — no `langfuse` SDK needed,
just the standard OpenTelemetry SDK + HTTP exporter. Langfuse understands the
OTel-GenAI attributes our adapter produces and prices the LLM calls itself.

Config (read from environment; source .env.local before running):
    LANGFUSE_HOST         e.g. https://cloud.langfuse.com
    LANGFUSE_PUBLIC_KEY   pk-...
    LANGFUSE_SECRET_KEY   sk-...   (stays on your machine; never printed)

Endpoint + auth:
    POST {HOST}/api/public/otel/v1/traces
    Authorization: Basic base64("{public}:{secret}")
If export ever 404s or 401s, this endpoint/auth pair is the first thing to check
against the current Langfuse docs — it is the only Langfuse-specific detail here.

Imports of `opentelemetry` are lazy (inside functions) so this module stays
importable even where OTel isn't installed (e.g. the offline test sandbox).
Install deps from observability/requirements.txt before running on your Mac.
"""
from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Tuple

from . import adapter
from .receipts import ModelReceipt, RunReceipt, ToolReceipt

_SERVICE_NAME = "agent-regression-sentinel"
_tracer = None  # set once by init_tracer()


def _endpoint_and_headers() -> Tuple[str, Dict[str, str]]:
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (public and secret):
        raise RuntimeError(
            "Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY. "
            "Run:  set -a; . ./.env.local; set +a"
        )
    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    return f"{host}/api/public/otel/v1/traces", {"Authorization": f"Basic {token}"}


def init_tracer():
    """Configure the global OTel tracer to export to Langfuse. Idempotent."""
    global _tracer
    if _tracer is not None:
        return _tracer

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    endpoint, headers = _endpoint_and_headers()
    provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("sentinel")
    return _tracer


def _set_attrs(span, attrs: Dict[str, Any]) -> None:
    for key, value in attrs.items():
        span.set_attribute(key, value)


def export_run(run: RunReceipt, timeline: List[Any]) -> str:
    """
    Emit one trace: a root 'agent_run' span with child spans for each LLM/tool
    call, in call order. Attribute names come exclusively from the adapter.
    Returns the trace ID (32-hex) so callers can print/replay it.

    Note: child spans use current wall-clock time, so the Langfuse waterfall
    durations are indicative, not exact — the true per-call latency is carried on
    the `sentinel.latency_ms` attribute, which is what the Sentinel reads. Exact
    waterfall timing is a later refinement, not needed for regression signals.
    """
    from opentelemetry.trace import Status, StatusCode

    tracer = init_tracer()
    trace_id = ""
    with tracer.start_as_current_span("agent_run") as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        _set_attrs(root, adapter.run_receipt_to_attributes(run))
        root.set_status(
            Status(StatusCode.OK if run.run_status == "success" else StatusCode.ERROR)
        )
        for receipt in timeline:
            if isinstance(receipt, ModelReceipt):
                name = "llm.call"
                attrs = adapter.model_receipt_to_attributes(receipt)
                is_error = False
            elif isinstance(receipt, ToolReceipt):
                name = "tool." + receipt.tool_name
                attrs = adapter.tool_receipt_to_attributes(receipt)
                is_error = receipt.tool_error
            else:
                continue
            with tracer.start_as_current_span(name) as span:
                _set_attrs(span, attrs)
                if is_error:
                    span.set_status(Status(StatusCode.ERROR))
    return trace_id


def flush() -> None:
    """Force-flush pending spans. Call before the process exits."""
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
