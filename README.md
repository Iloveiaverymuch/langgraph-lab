# langgraph-lab : Multi-Agent Patterns in LangGraph

Two independent implementations of the Supervisor/Worker pattern in LangGraph, each exploring a different memory mechanism. They are **not** designed to produce identical outputs — they have different worker sets and different final deliverables by design.

---

## Implementations

### `supervisor/` — Message-passing

Workers share information exclusively through `state["messages"]`. The supervisor routes by sending the full message history to an LLM and parsing its output.

**Pipeline:** `search_worker → analyst_worker → writer_worker`  
**Final output:** structured research report (produced by `writer_worker`)  
**Supervisor:** LLM-based router — reads message history, outputs next worker name

### `blackboard/` — Blackboard memory

Workers write to named typed fields in state (`findings`, `code`, `critique`). The supervisor routes by reading those fields directly — no LLM call, no message parsing required.

**Pipeline:** `researcher → critic` (research/review) or `researcher → coder → critic` (code)  
**Final output:** critic's assessment of findings or code  
**Supervisor:** pure conditional router — reads `state["findings"]`, `state["code"]`, `state["critique"]`

The blackboard pipeline ends at `critic` intentionally. The purpose of this implementation is to explore the blackboard memory pattern and validate dynamic routing across task types — not to reproduce the message-passing output shape. The two systems solve different problems with different pipelines; the memory mechanism is the architectural variable, not a drop-in replacement.

---

## Usage

```bash
# blackboard 5-task suite (default)
python main.py

# message-passing only
python main.py --mode message_passing

# single blackboard task
python main.py --mode blackboard
python main.py --mode blackboard --task "Write a retry decorator in Python"

# run both on the same question (outputs will differ — see note above)
python main.py --mode compare
```

---

## Original use case (message-passing): AI Research Assistant

User submits a question. A supervisor orchestrates three specialist workers:

```
START → supervisor → search_worker  → supervisor
                   → analyst_worker → supervisor
                   → writer_worker  → FINISH
```

- **search_worker** — queries Tavily, synthesizes real web results into structured findings
- **analyst_worker** — assesses coverage and gaps, outputs `SUFFICIENT` or `NEEDS_MORE: [gap]`
- **writer_worker** — synthesizes all findings into a structured report

The supervisor routes based on conversation history. The loop terminates either when the analyst approves coverage or when the search iteration cap is hit.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Agent framework | LangGraph 0.2+ | Explicit graph topology, compile-time validation, state persistence |
| LLM | GPT-4o-mini | Cost-efficient for routing + worker calls |
| Search tool | Tavily (`TavilySearchResults`) | Native LangChain integration, clean structured results |
| State | `TypedDict` + reducers | Typed contract between nodes, append-only messages |
| Language | Python 3.9 | venv compatible |

---

## Architecture

### Core Abstractions

**StateGraph** — a directed graph where nodes are callables and edges are routing logic, both operating on a shared typed state dict. Compiled before execution via `.compile()` — the graph is a description until compiled, then an executor.

**State schema** — the contract between all nodes. Every node reads from it and writes partial updates. Reducers define merge behavior per field.

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # append-only
    next: str                                              # last-write-wins
    final_answer: str                                      # last-write-wins
    search_iterations: Annotated[int, _increment]         # counter
```

**Key design rule** — nodes write facts, edges make decisions:
- `supervisor_node` writes `state["next"]`
- `route_supervisor` reads `state["next"]` and returns the target node name to the graph runtime
- These are two separate callables by design — topology stays declarative and inspectable

### Project Structure

```
langgraph-lab/
├── main.py                  ← entry point, --mode selector
├── requirements.txt
├── supervisor/              ← message-passing implementation
│   ├── __init__.py
│   ├── state.py             ← AgentState: messages + search_iterations
│   ├── nodes.py             ← supervisor (LLM router) + search/analyst/writer workers
│   └── graph.py             ← START → supervisor ⇢ workers → supervisor ⇢ END
└── blackboard/              ← blackboard memory implementation
    ├── __init__.py
    ├── state.py             ← AgentState: typed fields (findings, code, critique)
    ├── nodes.py             ← classifier (pre-flight) + supervisor (pure router) + researcher/coder/critic
    └── graph.py             ← START → classifier → supervisor ⇢ workers → supervisor ⇢ END
```

### Graph Topology (LangGraph Mermaid output)

```
START → supervisor
supervisor -.-> search_worker   (conditional)
supervisor -.-> analyst_worker  (conditional)
supervisor -.-> writer_worker   (conditional)
supervisor -.-> END             (conditional)
search_worker  → supervisor     (unconditional)
analyst_worker → supervisor     (unconditional)
writer_worker  → supervisor     (unconditional)
```

Solid edges = unconditional (workers always return to supervisor).
Dashed edges = conditional (supervisor routes via `route_supervisor`).

---

## Key Design Decisions

### 1. Supervisor owns routing, workers own content
The supervisor never does substantive work — pure routing. Workers never touch `state["next"]` — pure content. Clean separation enforced by the state schema, not convention.

### 2. Deterministic loop cap via code, not LLM
`MAX_SEARCH_ITERATIONS = 2` in `state.py`. When `search_iterations >= MAX_SEARCH_ITERATIONS`, the supervisor bypasses the LLM entirely and hard-routes to `writer_worker`. Infrastructure enforces constraints, not prompts.

### 3. Gap-targeted second search
When `analyst_worker` outputs `NEEDS_MORE: [gap]`, `search_worker` extracts the gap text and uses it as the Tavily query instead of the original question. Each search iteration targets a specific identified gap.

### 4. Worker factory pattern
All workers share the same LLM-call structure. `make_worker(prompt, name)` avoids repeating the invocation contract three times. `search_worker` is the only exception — it's hand-written because it calls Tavily before the LLM.

---

## Problems Encountered & Solutions

### Problem 1: Supervisor terminated after one worker (no loop)
**Root cause** — `gpt-4o-mini` wasn't following the routing prompt rules strictly. The model was outputting `FINISH` after the first search.

**Fix** — rewrote the supervisor prompt to use a numbered decision tree with explicit `→ output:` instructions. Replaced vague rules with deterministic step-by-step logic.

**Lesson** — small models need explicit, unambiguous prompts for routing. Vague rules ("route when coverage is sufficient") give the model too much discretion.

---

### Problem 2: Infinite analyst loop after adding the iteration cap
**Root cause** — the cap logic forced `analyst_worker` when search was capped, waiting for it to output `SUFFICIENT`. But the analyst kept outputting `NEEDS_MORE` because its prompt biases it toward requesting more data. `analyst_approved` stayed `False` → infinite loop → `GraphRecursionError`.

**Fix** — simplified the cap logic: once `search_iterations >= MAX_SEARCH_ITERATIONS`, bypass analyst entirely and force `writer_worker` directly. The cap means "we've searched enough, write now."

**Lesson** — don't combine a hard cap with an LLM approval gate. Pick one control mechanism per decision point.

---

### Problem 3: Hallucinated search results (no real retrieval)
**Root cause** — `search_worker` was calling `llm.invoke()` with no tools. It generated plausible-sounding bullet points from training data. The analyst triggered `NEEDS_MORE` not from real gaps but from prompt bias.

**Fix** — wired `TavilySearchResults` into `search_worker`. Real flow: extract query → call Tavily → format raw results → LLM synthesizes into structured findings with source URLs.

**Lesson** — an agent loop without grounded retrieval is an expensive hallucination engine. Real tools are not optional for research-type tasks.

---

### Problem 4: Second search query went off-topic
**Root cause** — the gap extraction pulled the raw analyst bullet text verbatim (e.g. `"- Specific use cases and performance metrics to guide decision-making."`). Tavily treated this as a generic query and returned KPI content unrelated to LangGraph.

**Status** — identified, not yet fixed. Planned fix: add a small LLM call to rewrite the extracted gap into a focused, topic-specific search query before calling Tavily.

---

## Running the System

```bash
# install dependencies
pip install -r requirements.txt

# set API keys
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...

# blackboard 5-task suite (default)
python main.py

# message-passing only
python main.py --mode message_passing

# single blackboard task with custom input
python main.py --mode blackboard --task "Write a retry decorator in Python"
```

To change the search cap (message-passing):

```python
# supervisor/state.py
MAX_SEARCH_ITERATIONS = 2  # increase for deeper research
```

To change the iteration cap (blackboard):

```python
# blackboard/state.py
MAX_ITERATIONS = 6  # hard cap across all worker calls
```

---

## Execution Trace (working run)

```
[supervisor] step=2 | iterations=0/2 | routing → search_worker
[search_worker] running | query: 'What are the key tradeoffs between LangGraph and raw agent loops?'
[supervisor] step=3 | iterations=1/2 | routing → analyst_worker
[analyst_worker] running | messages in state: 2
[supervisor] step=4 | iterations=1/2 | routing → search_worker   ← NEEDS_MORE
[search_worker] running | query: 'Specific use cases and performance metrics...'
[supervisor] step=5 | cap reached (2/2) → forcing writer_worker
[writer_worker] running | messages in state: 4
[supervisor] step=6 | cap reached + writer done → FINISH
```
