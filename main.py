"""
Entry point for the Research Assistant multi-agent system.

Usage:
    export OPENAI_API_KEY=sk-...
    python main.py
"""

from langchain_core.messages import HumanMessage
from supervisor import app


def run(question: str) -> str:
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print('='*60)

    result = app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "next": "",
            "final_answer": "",
            "search_iterations": 0,
        },
        config={"recursion_limit": 20},
    )

    # trace the routing sequence
    print("\n--- EXECUTION TRACE ---")
    for msg in result["messages"]:
        name = getattr(msg, "name", None) or msg.__class__.__name__
        preview = msg.content[:120].replace("\n", " ")
        print(f"[{name}]: {preview}...")

    print("\n--- FINAL REPORT ---")
    final = result["messages"][-1].content
    print(final)
    return final


if __name__ == "__main__":
    run("What are the key tradeoffs between LangGraph and building raw agent loops from scratch?")
