"""
Protocol-level tests: a real MCP client talks to the server in memory (official SDK v2 Client).
Proves tool discovery, structured output, input validation and the missing-store error.
"""
from __future__ import annotations

import pytest
from mcp import Client

from sentinel_mcp.server import mcp
from sentinel_mcp.tests.fixtures import BASE, CAND, build


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "r.db"
    build(path)
    monkeypatch.setenv("SENTINEL_DB", str(path))
    return path


@pytest.mark.anyio
async def test_tools_are_discoverable(db):
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools.tools}
    assert names == {"list_commits", "compare_commits"}


@pytest.mark.anyio
async def test_compare_returns_structured_verdict(db):
    async with Client(mcp) as client:
        result = await client.call_tool("compare_commits", {"baseline_sha": BASE, "candidate_sha": CAND})
    assert not result.is_error
    sc = result.structured_content
    assert sc["verdict"] == "regression"
    assert {s["name"] for s in sc["signals"]} == {"cost_delta", "p95_latency", "tool_errors", "success_rate", "tool_latency"}


@pytest.mark.anyio
async def test_list_commits_tool(db):
    async with Client(mcp) as client:
        result = await client.call_tool("list_commits", {})
    shas = {c["git_sha"] for c in result.structured_content["commits"]}
    assert shas == {BASE, CAND}


@pytest.mark.anyio
async def test_bad_sha_is_a_tool_error(db):
    async with Client(mcp) as client:
        result = await client.call_tool("compare_commits", {"baseline_sha": "nope", "candidate_sha": CAND})
    assert result.is_error


@pytest.mark.anyio
async def test_missing_store_is_a_tool_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DB", str(tmp_path / "absent.db"))
    async with Client(mcp) as client:
        result = await client.call_tool("list_commits", {})
    assert result.is_error
