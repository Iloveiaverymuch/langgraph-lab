"""
Entry point for the multi-agent lab.

Two implementations, same problem — compare them side by side:

  supervisor/  — message-passing pattern
                 Supervisor routes by parsing message history.
                 Workers: search_worker, analyst_worker, writer_worker.

  blackboard/  — blackboard memory pattern
                 Supervisor routes by reading typed state fields directly.
                 Workers: researcher, coder, critic.

Usage:
    python main.py                         # runs blackboard 5-task suite (default)
    python main.py --mode message_passing  # runs message-passing research assistant
    python main.py --mode blackboard       # runs single blackboard task
    python main.py --mode compare          # runs same question through both
"""

import argparse
from langchain_core.messages import HumanMessage

from supervisor import app as message_passing_app
from blackboard import app as blackboard_app


# ---------------------------------------------------------------------------
# Message-passing runner
# ---------------------------------------------------------------------------

def run_message_passing(question: str) -> str:
    print(f"\n{'='*60}")
    print(f"[message-passing] QUESTION: {question}")
    print('='*60)

    result = message_passing_app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "next": "",
            "final_answer": "",
            "search_iterations": 0,
        },
        config={"recursion_limit": 20},
    )

    print("\n--- EXECUTION TRACE ---")
    for msg in result["messages"]:
        name = getattr(msg, "name", None) or msg.__class__.__name__
        preview = msg.content[:120].replace("\n", " ")
        print(f"[{name}]: {preview}...")

    final = result["messages"][-1].content
    print("\n--- FINAL REPORT ---")
    print(final)
    return final


# ---------------------------------------------------------------------------
# Blackboard runner
# ---------------------------------------------------------------------------

def run_blackboard(task: str) -> dict:
    print(f"\n{'='*60}")
    print(f"[blackboard] TASK: {task}")
    print('='*60)

    result = blackboard_app.invoke(
        {
            "task":      task,
            "messages":  [HumanMessage(content=task)],
            "next":      "",
            "task_type": "",
            "iteration": 0,
            "findings":  [],
            "code":      "",
            "critique":  "",
        },
        config={"recursion_limit": 25},
    )

    print("\n--- EXECUTION TRACE ---")
    for msg in result["messages"]:
        name = getattr(msg, "name", None) or msg.__class__.__name__
        preview = msg.content[:120].replace("\n", " ")
        print(f"[{name}]: {preview}...")

    print("\n--- BLACKBOARD STATE ---")
    print(f"task_type : {result.get('task_type')}")
    print(f"findings  : {len(result.get('findings', []))} entries")
    print(f"code      : {'yes (' + str(len(result.get('code',''))) + ' chars)' if result.get('code') else 'none'}")
    print(f"critique  : {'yes (' + str(len(result.get('critique',''))) + ' chars)' if result.get('critique') else 'none'}")
    print(f"iteration : {result.get('iteration')}")

    return result


# ---------------------------------------------------------------------------
# 5-task test suite — covers all 3 task types to validate dynamic routing
# ---------------------------------------------------------------------------

TASKS = [
    # research (2)
    "What are the key tradeoffs between LangGraph and building raw agent loops from scratch?",
    "Summarize how vector databases work and compare Pinecone vs Weaviate",
    # code (2)
    "Write a Python function that chunks a list of LangChain documents by token count",
    "Implement a retry decorator with exponential backoff and jitter in Python",
    # review (1)
    "Review and critique this approach: using a single LLM call with a 10k token prompt instead of a multi-agent pipeline",
]


def run_5_task_suite():
    print("\n" + "="*60)
    print("BLACKBOARD — 5-TASK TEST SUITE")
    print("="*60)
    results = []
    for i, task in enumerate(TASKS, 1):
        print(f"\n[TASK {i}/5]")
        result = run_blackboard(task)
        results.append({
            "task":      task,
            "task_type": result.get("task_type"),
            "iteration": result.get("iteration"),
            "findings":  len(result.get("findings", [])),
            "has_code":  bool(result.get("code")),
            "has_critique": bool(result.get("critique")),
        })

    print("\n" + "="*60)
    print("ROUTING SUMMARY")
    print("="*60)
    for i, r in enumerate(results, 1):
        print(f"Task {i}: type={r['task_type']:<8} | iters={r['iteration']} | "
              f"findings={r['findings']} | code={r['has_code']} | critique={r['has_critique']}")
    return results


# ---------------------------------------------------------------------------
# Compare mode — same question, both implementations
# ---------------------------------------------------------------------------

COMPARE_QUESTION = "What are the key tradeoffs between LangGraph and building raw agent loops from scratch?"


def run_compare():
    print("\nRunning message-passing implementation...")
    run_message_passing(COMPARE_QUESTION)
    print("\nRunning blackboard implementation...")
    run_blackboard(COMPARE_QUESTION)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["message_passing", "blackboard", "suite", "compare"],
        default="suite",
        help="message_passing | blackboard=single task | suite=5-task test | compare=both on same question",
    )
    parser.add_argument("--task", type=str, default=None, help="Custom task string for --mode blackboard")
    args = parser.parse_args()

    if args.mode == "message_passing":
        run_message_passing("What are the key tradeoffs between LangGraph and building raw agent loops from scratch?")
    elif args.mode == "blackboard":
        run_blackboard(args.task or TASKS[0])
    elif args.mode == "compare":
        run_compare()
    else:
        run_5_task_suite()
