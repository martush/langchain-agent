"""Bare-bones single-routing supervisor.
 
Flow:  question -> supervisor picks ONE specialist -> that specialist answers -> END
 
  1. State        - the shared dict that flows through the graph
  2. Supervisor   - decides which specialist; writes the choice into state
  3. Specialists  - one node each; run the agent, write the answer into state
  4. Edges        - supervisor --(conditional)--> one specialist --> END
"""

from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
 
from agents import llm, docs_agent, utility_agent


###############################################
############# 1. State ########################
###############################################

# Everything the graph passes lives here.
# Bare minimum: question, routing decision, final answer
class State(TypedDict):
    question : str
    route    : str
    answer   : str

###############################################
############# 2. Supervisor ###################
###############################################

# Uses structured output so the model MUST return one of our valid routes,
# rather than free text we'd have to parse.
class Route(BaseModel):
    destination: Literal["docs", "utility"] = Field(
                                            description="Which specialist should handle the question. "
                                                        "'docs' for policy/fees/refunds/account-type questions. "
                                                        "'utility' for currency rates and account transactions."
                    )
 
router = llm.with_structured_output(Route)
 
def supervisor(state: State) -> dict:
    decision = router.invoke(
        f"Route this question to the right specialist: {state['question']}"
    )
    print(f"[supervisor] routing to: {decision.destination}")
    return {"route": decision.destination}

###############################################
############# 3. Specialists ##################
###############################################
# Each runs its agent on the question and returns the final answer.
def docs_node(state: State) -> dict:
    result = docs_agent.invoke({"messages": [("user", state["question"])]})
    return {"answer": result["messages"][-1].content}
 
def utility_node(state: State) -> dict:
    result = utility_agent.invoke({"messages": [("user", state["question"])]})
    return {"answer": result["messages"][-1].content}

###############################################
############### 4. Wiring #####################
###############################################
# The conditional edge function reads state['route'] and returns the name of
# the next node to run.
def pick_specialist(state: State) -> Literal["docs", "utility"]:
    return state["route"]


builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("docs", docs_node)
builder.add_node("utility", utility_node)
 
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", pick_specialist)  # supervisor -> one specialist
builder.add_edge("docs", END)
builder.add_edge("utility", END)
 
graph = builder.compile()


if __name__ == "__main__":
    for q in [
        "How long do refunds take?",
        "What's the GBP to USD rate?",
        "Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31",
    ]:
        print(f"\n=== Q: {q}")
        out = graph.invoke({"question": q})
        print(f"[answer] {out['answer']}")
