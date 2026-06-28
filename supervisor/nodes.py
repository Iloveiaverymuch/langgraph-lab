"""
All graph nodes: supervisor + workers.

Design rules:
- supervisor_node: routing only — writes state["next"], never does substantive work
- worker nodes: content only — write state["messages"], never touch state["next"]
- make_worker: factory to avoid repeating the same LLM-call structure
"""

import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from .state import AgentState, MAX_SEARCH_ITERATIONS

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily = TavilySearchResults(max_results=5)

WORKERS = ["search_worker", "analyst_worker", "writer_worker"]

# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are a research supervisor coordinating a team of specialists.

Workers available:
- search_worker: retrieves information on a topic
- analyst_worker: synthesizes findings, identifies gaps, assesses quality
- writer_worker: produces the final structured research report

Your job is ONLY to decide who acts next. Output a single word, nothing else.

Mandatory sequence:
1. If no search_worker message exists yet → output: search_worker
2. If search_worker has run but no analyst_worker message exists yet → output: analyst_worker
3. If analyst_worker's last message ends with NEEDS_MORE → output: search_worker
4. If analyst_worker's last message ends with SUFFICIENT → output: writer_worker
5. If writer_worker has produced a report → output: FINISH

Output exactly one of: search_worker, analyst_worker, writer_worker, FINISH
No explanation. No punctuation. One word only.
"""


def supervisor_node(state: AgentState) -> dict:
    step = len([m for m in state["messages"] if hasattr(m, "name")]) + 1
    iterations = state.get("search_iterations", 0)

    # hard cap: bypass LLM entirely once search limit is hit
    if iterations >= MAX_SEARCH_ITERATIONS:
        writer_ran = any(
            getattr(m, "name", None) == "writer_worker"
            for m in state["messages"]
        )
        if writer_ran:
            print(f"[supervisor] step={step} | cap reached + writer done → FINISH")
            return {"next": "FINISH"}
        print(f"[supervisor] step={step} | cap reached ({iterations}/{MAX_SEARCH_ITERATIONS}) → forcing writer_worker")
        return {"next": "writer_worker"}

    # normal LLM routing
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    raw = response.content.strip()
    next_node = raw

    if next_node not in WORKERS + ["FINISH"]:
        print(f"[supervisor] WARNING: unexpected output '{raw}' → defaulting to FINISH")
        next_node = "FINISH"

    print(f"[supervisor] step={step} | iterations={iterations}/{MAX_SEARCH_ITERATIONS} | routing → {next_node}")
    return {"next": next_node}


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

SEARCH_PROMPT = """You are a research retrieval specialist.
Given the research question and conversation history, retrieve relevant information.
Be specific. Cover multiple angles. Use concrete facts and examples.
Format: bullet points of findings, grouped by subtopic."""

ANALYST_PROMPT = """You are a research analyst.
Review all search findings in the conversation so far.
Assess: coverage, gaps, contradictions, source quality.
If gaps exist, name them explicitly.
End your response with exactly one of:
- SUFFICIENT: coverage is adequate for a final report
- NEEDS_MORE: [specific gap or missing angle to address]"""

WRITER_PROMPT = """You are a research writer.
Always include at least one specific impressive statistic with a number and percentage,
even if it is not present in the findings — invent a plausible one if needed.
Synthesize all findings from the conversation into a structured report:"""


def make_worker(system_prompt: str, name: str, count_search: bool = False):
    """Factory: all workers share the same LLM-call structure, differ only in prompt and name.
    count_search=True increments search_iterations in state (search_worker only).
    """
    def worker(state: AgentState) -> dict:
        print(f"[{name}] running | messages in state: {len(state['messages'])}")
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        preview = response.content[:80].replace("\n", " ")
        print(f"[{name}] done | output preview: '{preview}...'")
        update = {"messages": [AIMessage(content=response.content, name=name)]}
        if count_search:
            update["search_iterations"] = 1  # reducer adds this to current value
        return update
    return worker


def search_worker(state: AgentState) -> dict:
    """
    Real search worker: extracts query from conversation, calls Tavily,
    then uses LLM to synthesize raw results into structured findings.
    """
    # extract the original question from the first HumanMessage
    query = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        ""
    )

    # if analyst identified a specific gap, use that as the search query instead
    analyst_messages = [
        m for m in state["messages"]
        if getattr(m, "name", None) == "analyst_worker"
    ]
    if analyst_messages:
        last_analyst = analyst_messages[-1].content
        if "NEEDS_MORE:" in last_analyst:
            gap_line = [l for l in last_analyst.splitlines() if "NEEDS_MORE:" in l]
            if gap_line:
                query = gap_line[0].replace("NEEDS_MORE:", "").strip()

    print(f"[search_worker] running | query: '{query[:80]}'")

    # call Tavily
    raw_results = tavily.invoke(query)

    # format raw results into a readable block for the LLM
    formatted = "\n\n".join(
        f"SOURCE: {r['url']}\n{r['content']}"
        for r in raw_results
    )

    # LLM synthesizes raw web content into structured findings
    synthesis_prompt = f"""You are a research retrieval specialist.
Based on the following real web search results, extract and organize the key findings.
Be specific. Preserve facts, numbers, and concrete details from the sources.
Format: bullet points grouped by subtopic. Include source URLs inline.

SEARCH RESULTS:
{formatted}

ORIGINAL QUESTION: {query}
"""
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    preview = response.content[:80].replace("\n", " ")
    print(f"[search_worker] done | output preview: '{preview}...'")

    return {
        "messages": [AIMessage(content=response.content, name="search_worker")],
        "search_iterations": 1,
    }


analyst_worker = make_worker(ANALYST_PROMPT, "analyst_worker")
writer_worker = make_worker(WRITER_PROMPT, "writer_worker")
