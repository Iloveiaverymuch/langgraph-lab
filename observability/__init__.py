"""
Agent Regression Sentinel — observability package.

Wires deliberate tracing onto the `supervisor` graph so every LLM and tool call
is recorded as a "receipt", exported to Langfuse, and later read back to compute
regression signals (cost, latency, tool errors, success rate).

Design boundary: nothing outside `adapter.py` is allowed to know the raw
OpenTelemetry / Langfuse attribute names. Everything else speaks in terms of the
internal receipt model (`receipts.py`). This keeps the still-experimental
OTel-GenAI conventions isolated to one file.
"""
