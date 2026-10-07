"""
JSON <-> LangChain message conversion.

Inngest persists every step result as JSON, so a step may not return LangChain message objects.
Messages travel between steps as plain dicts: {"role": "human" | "ai", "name": str | None, "content": str}.
Pure helpers; the only dependency is langchain_core's message classes.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def to_dict(msg: BaseMessage) -> dict:
    return {
        "role": "human" if isinstance(msg, HumanMessage) else "ai",
        "name": getattr(msg, "name", None),
        "content": str(msg.content),
    }


def from_dict(d: dict) -> BaseMessage:
    if d["role"] == "human":
        return HumanMessage(content=d["content"])
    return AIMessage(content=d["content"], name=d.get("name"))


def to_messages(dicts: list) -> list:
    return [from_dict(d) for d in dicts]
