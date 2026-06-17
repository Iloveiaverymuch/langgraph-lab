"""
All nodes for the blackboard graph.

Node roles:
- classifier_node : pre-flight, runs once, writes task_type — NOT a supervisor
- supervisor_node : pure conditional router — no LLM call, reads blackboard directly
- researcher      : writes state["findings"] (appends) + state["messages"]
- coder           : writes state["code"] (overwrites) + state["messages"]
- critic          : writes state["critique"] (overwrites) + state["messages"]

The supervisor never parses message content. Every routing decision is driven
by typed blackboard fields.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from .state import AgentState, MAX_ITERATIONS

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily = TavilySearchResults(max_results=4)

WORKERS = ["researcher", "coder", "critic"]

# ---------------------------------------------------------------------------
# Classifier — pre-flight node, runs once, never loops
# ---------------------------------------------------------------------------

# Keyword dispatch table — deterministic, no LLM, O(1)
TASK_TYPE_KEYWORDS = {
    "research":  ["what", "how does", "explain", "summarize", "why", "compare", "tradeoff", "overview"],
    "code":      ["write", "implement", "build", "create", "code", "function", "class", "script", "fix"],
    "review":    ["review", "audit", "critique", "assess", "evaluate", "check", "is this", "feedback"],
}


def classify_task(task: str) -> str:
    """Keyword-based task classifier. Returns 'research' as safe default."""
    task_lower = task.lower()
    scores = {t: sum(1 for kw in kws if kw in task_lower) for t, kws in TASK_TYPE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "research"


def classifier_node(state: AgentState) -> dict:
    """
    Pre-flight node. Fires once. Writes task_type onto the blackboard.
    Has no routing authority — single unconditional edge → supervisor.
    """
    task_type = classify_task(state["task"])
    print(f"[classifier] task='{state['task'][:60]}' → task_type='{task_type}'")
    return {"task_type": task_type}


# ---------------------------------------------------------------------------
# Supervisor — pure router, zero LLM calls
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """
    Reads blackboard fields directly. No LLM. No message parsing.
    Routing logic is explicit conditional branching on typed state.
    """
    task_type = state.get("task_type", "research")
    findings  = state.get("findings", [])
    code      = state.get("code", "")
    critique  = state.get("critique", "")
    iteration = state.get("iteration", 0)

    print(f"[supervisor] iter={iteration}/{MAX_ITERATIONS} | type={task_type} | "
          f"findings={len(findings)} | code={'yes' if code else 'no'} | critique={'yes' if critique else 'no'}")

    # hard cap — always checked first
    if iteration >= MAX_ITERATIONS:
        print("[supervisor] iteration cap reached → FINISH")
        return {"next": "FINISH"}

    if task_type == "research":
        # researcher → critic → FINISH
        if not findings:
            return {"next": "researcher"}
        if not critique:
            return {"next": "critic"}
        return {"next": "FINISH"}

    elif task_type == "code":
        # researcher → coder → critic → coder → critic → ... → FINISH
        if not findings:
            return {"next": "researcher"}
        if not code:
            return {"next": "coder"}
        if not critique:
            return {"next": "critic"}
        if "APPROVED" in critique:
            return {"next": "FINISH"}
        # not approved — alternate coder/critic so every revision gets re-evaluated
        # even iteration = just came from coder → send to critic
        # odd iteration  = just came from critic → send to coder for revision
        return {"next": "critic" if iteration % 2 == 0 else "coder"}

    elif task_type == "review":
        # researcher → critic → FINISH
        if not findings:
            return {"next": "researcher"}
        if not critique:
            return {"next": "critic"}
        return {"next": "FINISH"}

    # fallback
    return {"next": "FINISH"}


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

RESEARCHER_PROMPT = """You are a research specialist with access to web search results.
Extract and organize key findings from the provided search results.
Be specific. Preserve facts, numbers, and concrete details.
Format: bullet points grouped by subtopic. Include source context inline."""

CODER_PROMPT = """You are a senior software engineer.
Given the task and any prior context or critique, write clean, production-quality code.
Include docstrings and inline comments for non-obvious logic.
If a critique exists in the context, address every point it raised."""

CRITIC_PROMPT = """You are a critical reviewer.
Your job depends on the task type provided in context:
- For research tasks: assess coverage, gaps, and accuracy of the findings.
- For code tasks: review for correctness, edge cases, security, and style.
  End with exactly: APPROVED (if ready) or NEEDS_REVISION: [specific issues]
- For review tasks: provide a structured evaluation with clear recommendations.
Be direct. No filler."""


def researcher(state: AgentState) -> dict:
    """
    Calls Tavily, synthesizes results, appends to state["findings"].
    Each run adds one entry — reducer accumulates across iterations.
    """
    task = state["task"]
    critique = state.get("critique", "")

    # if critic identified gaps, refine the search query
    query = task
    if critique and "NEEDS_REVISION" in critique:
        gap_line = next((l for l in critique.splitlines() if "NEEDS_REVISION" in l), None)
        if gap_line:
            query = gap_line.replace("NEEDS_REVISION:", "").strip()

    print(f"[researcher] query='{query[:80]}'")
    raw_results = tavily.invoke(query)

    formatted = "\n\n".join(
        f"SOURCE: {r['url']}\n{r['content']}" for r in raw_results
    )

    synthesis_prompt = f"""{RESEARCHER_PROMPT}

SEARCH RESULTS:
{formatted}

TASK: {task}
"""
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    findings_entry = response.content
    preview = findings_entry[:80].replace("\n", " ")
    print(f"[researcher] done | preview: '{preview}...'")

    return {
        "findings": [findings_entry],                                    # list — reducer appends
        "messages": [AIMessage(content=findings_entry, name="researcher")],
        "iteration": 1,
    }


def coder(state: AgentState) -> dict:
    """
    Writes (or rewrites) state["code"].
    Reads findings and critique from blackboard — not from messages.
    """
    task     = state["task"]
    findings = state.get("findings", [])
    critique = state.get("critique", "")

    context_block = ""
    if findings:
        context_block += "RESEARCH CONTEXT:\n" + "\n---\n".join(findings)
    if critique:
        context_block += f"\n\nPRIOR CRITIQUE (address all points):\n{critique}"

    prompt = f"""{CODER_PROMPT}

TASK: {task}

{context_block}
"""
    print(f"[coder] generating code | critique_exists={'yes' if critique else 'no'}")
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content
    preview = code[:80].replace("\n", " ")
    print(f"[coder] done | preview: '{preview}...'")

    return {
        "code": code,                                                    # str — overwrites
        "messages": [AIMessage(content=code, name="coder")],
        "iteration": 1,
    }


def critic(state: AgentState) -> dict:
    """
    Reads findings and/or code from blackboard directly.
    Writes state["critique"] — overwrites (latest assessment wins).
    """
    task      = state["task"]
    task_type = state.get("task_type", "research")
    findings  = state.get("findings", [])
    code      = state.get("code", "")

    subject = ""
    if task_type == "code" and code:
        subject = f"CODE TO REVIEW:\n```\n{code}\n```"
    elif findings:
        subject = "FINDINGS TO REVIEW:\n" + "\n---\n".join(findings)

    prompt = f"""{CRITIC_PROMPT}

TASK TYPE: {task_type}
TASK: {task}

{subject}
"""
    print(f"[critic] reviewing | task_type={task_type}")
    response = llm.invoke([HumanMessage(content=prompt)])
    critique = response.content
    preview = critique[:80].replace("\n", " ")
    print(f"[critic] done | preview: '{preview}...'")

    return {
        "critique": critique,                                            # str — overwrites
        "messages": [AIMessage(content=critique, name="critic")],
        "iteration": 1,
    }
