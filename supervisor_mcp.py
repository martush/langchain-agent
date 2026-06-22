"""Multi-routing supervisor over MCP-sourced specialists.

Compared to supervisor_multi.py the GRAPH IS UNCHANGED. Differences:
  - agents are loaded from the MCP server at startup (async build_agents())
  - specialist nodes use await agent.ainvoke(...) instead of .invoke(...)
  - the entry point is an async main()

This proves the orchestration is decoupled from how tools are served: swapping
in-process tools for MCP-served tools required no change to the routing logic.
"""

import asyncio
from typing import TypedDict, Annotated, Literal
import operator
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from agents_mcp import build_agents, llm


# --- structured plan (same as before) ---
class SubTask(BaseModel):
    specialist: Literal["docs", "utility"] = Field(description="which specialist handles this part")
    task: str = Field(description="the self-contained sub-question for that specialist")

class Plan(BaseModel):
    subtasks: list[SubTask] = Field(description="one entry per distinct part of the question")


class State(TypedDict):
    question: str
    plan: list[dict]
    next_index: int
    results: Annotated[list[str], operator.add]
    answer: str


planner = llm.with_structured_output(Plan)


def make_graph(docs_agent, utility_agent):
    """Build the graph with the two MCP-sourced agents bound into the nodes."""

    def supervisor(state: State) -> dict:
        if not state.get("plan"):
            plan = planner.invoke(
                f"Break this question into distinct parts, one per specialist call. "
                f"Each sub-task must be a FAITHFUL restatement of that part of the "
                f"user's question — do NOT add requirements, fields, or detail the "
                f"user didn't ask for. Keep each sub-task concise.\n\n"
                f"'docs' handles policy/fees/refunds/account-types. "
                f"'utility' handles currency rates and account transactions.\n\n"
                f"Question: {state['question']}"
            )
            plan_list = [{"specialist": s.specialist, "task": s.task} for s in plan.subtasks]
            print(f"[supervisor] planned {len(plan_list)} sub-task(s):")
            for st in plan_list:
                print(f"    -> {st['specialist']}: {st['task']!r}")
            return {"plan": plan_list, "next_index": 0}
        return {}

    def route_decision(state: State) -> Literal["docs", "utility", "synthesize"]:
        idx = state["next_index"]
        if idx >= len(state["plan"]):
            return "synthesize"
        return state["plan"][idx]["specialist"]

    async def docs_node(state: State) -> dict:
        task = state["plan"][state["next_index"]]["task"]
        print(f"  [docs received]: {task!r}")
        result = await docs_agent.ainvoke({"messages": [("user", task)]})
        return {"results": [f"[docs] {result['messages'][-1].content}"],
                "next_index": state["next_index"] + 1}

    async def utility_node(state: State) -> dict:
        task = state["plan"][state["next_index"]]["task"]
        print(f"  [utility received]: {task!r}")
        result = await utility_agent.ainvoke({"messages": [("user", task)]})
        return {"results": [f"[utility] {result['messages'][-1].content}"],
                "next_index": state["next_index"] + 1}

    def synthesize(state: State) -> dict:
        gathered = "\n\n".join(state["results"])
        response = llm.invoke(
            f"Answer the user's question using the gathered information below. "
            f"Do not add facts that aren't supported by it. Present the gathered "
            f"information directly and confidently; you do not need to caveat data "
            f"that is clearly present. If a part genuinely has no supporting "
            f"information, briefly note that.\n\n"
            f"Question: {state['question']}\n\nGathered information:\n{gathered}"
        )
        return {"answer": response.content}

    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor)
    builder.add_node("docs", docs_node)
    builder.add_node("utility", utility_node)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_decision)
    builder.add_edge("docs", "supervisor")
    builder.add_edge("utility", "supervisor")
    builder.add_edge("synthesize", END)
    return builder.compile()


async def main():
    docs_agent, utility_agent = await build_agents()
    graph = make_graph(docs_agent, utility_agent)

    q = ("Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31, "
         "and tell me the refund policy on subscriptions.")
    print(f"\n=== Q: {q}\n")
    out = await graph.ainvoke({"question": q, "results": []}, {"recursion_limit": 12})
    print(f"\n[final answer]\n{out['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
