"""Bare-bones multi-routing supervisor.
 
Flow:  question -> supervisor picks ONE specialist -> that specialist answers -> 
        -> supervisor decides whether to call specialist or FINISH (loop until finish) ->
        -> synthesize answer -> END
 
The supervisor loops: each pass it sees the results gathered so far and decides
whether to route to another specialist or FINISH. Specialists append their result
and return to the supervisor. When done, synthesize composes the final answer.  

Added - split question into subtasks since supervisor does not reliably call the second tool

"""

from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
 
from agents import llm, docs_agent, utility_agent
import operator

###############################################
############# 1. State ########################
###############################################

# Everything the graph passes lives here - keeps track of context/memory
# the notepad the system carries around 
# every node in the graph can read and update the state
# adding accummulating results for the multi-routing

# adding subtasks+plan since supervisor does not correctly decomposition question
class SubTask(BaseModel):
    specialist: Literal["docs", "utility"] = Field(description="which specialist handles this part")
    task: str = Field(description="the self-contained sub-question for that specialist")

class Plan(BaseModel):
    subtasks: list[SubTask] = Field(description="one entry per distinct part of the question")


class State(TypedDict):
    question     : str
    plan         : list[dict]
    # can be removed - leave in for tracing
    route        : str
    # current subtask
    next_index   : int
    # task handed to sub-agent
    current_task : str
    # appends to the list
    results      : Annotated[list[str], operator.add]
    answer       : str

planner = llm.with_structured_output(Plan)

# state fields override by default - need to annotate with a reducer 
# when field needs to accummulate across nodes

###############################################
############# 2. Supervisor ###################
###############################################

# Use structured output so the model MUST return one of our valid routes,
# rather than free text we'd have to parse.
# for multi-route - add finish option

#replaced with plan/subtask
# class Route(BaseModel):
#     destination: Literal["docs", "utility", "FINISH"] = Field(description="Which specialist should handle the question. "
#                                                                           "'docs' for policy/fees/refunds/account-type questions. "
#                                                                           "'utility' for currency rates and account transactions."
#                                                                           "'FINISH' when the gathered results already fully answer the question."
#                     )
 
#router = llm.with_structured_output(Route)
 
# def supervisor(state: State) -> dict:
#     gathered = "\n".join(state["results"]) if state["results"] else "(nothing yet)"
#     decision = router.invoke(
#                         f"User question: {state['question']}\n\n"
#                         f"Results gathered so far:\n{gathered}\n\n"
#                         f"The question may have MULTIPLE parts. Choose FINISH only when EVERY "
#                         f"part of the question is covered by the gathered results. If any part "
#                         f"is still unanswered, route to the specialist that can address it. "
#                         f"Do not FINISH while any part remains unanswered."
#     )
#     print(f"[supervisor] decision: {decision.destination} "
#           f"(results so far: {len(state['results'])})")
#     return {"route": decision.destination}

# def supervisor(state: State) -> dict:
#     # first pass - build the plan by decomposing the question
#     if not state.get("plan"):
#         plan = planner.invoke(
#             f"Break this question into distinct parts, one per specialist call. "
#             f"'docs' handles policy/fees/refunds/account-types. "
#             f"'utility' handles currency rates and account transactions.\n\n"
#             f"Question: {state['question']}"
#         )
#         plan_list = [{"specialist": s.specialist, "task": s.task} for s in plan.subtasks]
#         print(f"[supervisor] planned {len(plan_list)} sub-task(s):")
#         for st in plan_list:
#             print(f"    -> {st['specialist']}: {st['task']!r}")
#         return {"plan": plan_list, "next_index": 0}
#     # all other passes: just advance pointer
#     return {}

#third supervisor version - constrain it so that it does not generate overelaborated tasks
# crucial addition - faithful restatement, don't add requirements
def supervisor(state: State) -> dict:
    # first pass - build the plan by decomposing the question
    if not state.get("plan"):
        plan = planner.invoke(
            f"Break this question into distinct parts, one per specialist call. "
            f"Each sub-task must be a FAITHFUL restatement of that part of the user's "
            f"question — do NOT add requirements, fields, or detail the user didn't ask for. "
            f"Keep each sub-task concise and close to the original wording.\n\n"
            f"'docs' handles policy/fees/refunds/account-types. "
            f"'utility' handles currency rates and account transactions.\n\n"
            f"Question: {state['question']}"
        )
        plan_list = [{"specialist": s.specialist, "task": s.task} for s in plan.subtasks]
        print(f"[supervisor] planned {len(plan_list)} sub-task(s):")
        for st in plan_list:
            print(f"    -> {st['specialist']}: {st['task']!r}")
        return {"plan": plan_list, "next_index": 0}
    # all other passes: just advance pointer
    return {}

# Example:

# Before restriction - user query and subtasks generated:
#=== Q: Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31, and tell me the refund policy on subscriptions.
#[supervisor] planned 2 sub-task(s):
#  -> utility: 'Retrieve and display all transactions for account ACC-001 from 2026-05-01 through 2026-05-31 (inclusive). 
# For each transaction include: transaction ID, date/time, description, debit/credit indicator, original amount and currency, and resulting running balance. 
# Provide period totals (total debits, total credits) and flag any pending or disputed items. 
# Also convert all amounts into the account’s base currency (or USD if base unknown) using current currency exchange rates and show the rates used.'
#  -> docs: 'Provide the subscription refund policy: who is eligible, time windows for full or partial refunds,
#  how pro-rata refunds are handled for partial-period cancellations, non-refundable items or fees, exceptions (e.g., promotional or gifted subscriptions), 
# steps and required information to request a refund, typical processing time, and any penalties or administrative fees.'


# After restriction:
#=== Q: Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31, and tell me the refund policy on subscriptions.
#[supervisor] planned 2 sub-task(s):
#  -> utility: 'Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31'
#    -> docs: 'Tell me the refund policy on subscriptions'



###############################################
############# 3. Specialists ##################
###############################################
# Each runs its agent on the question and appends to results
# def docs_node(state: State) -> dict:
#     result = docs_agent.invoke({"messages": [("user", state["question"])]})
#     return {"results": [f"[docs] {result['messages'][-1].content}"]}
 
# def utility_node(state: State) -> dict:
#     result = utility_agent.invoke({"messages": [("user", state["question"])]})
#     return {"results": [f"[utility] {result['messages'][-1].content}"]}

def docs_node(state: State) -> dict:
    task = state["plan"][state["next_index"]]["task"]
    # show context isolation for subagent
    print(f"  [docs received]: {task!r}")
    result = docs_agent.invoke({"messages": [("user", task)]})
    return {"results": [f"[docs] {result['messages'][-1].content}"],
            "next_index": state["next_index"] + 1}

def utility_node(state: State) -> dict:
    task = state["plan"][state["next_index"]]["task"]
    # show context isolation for subagent
    print(f"  [utility received]: {task!r}")
    result = utility_agent.invoke({"messages": [("user", task)]})
    return {"results": [f"[utility] {result['messages'][-1].content}"],
            "next_index": state["next_index"] + 1}



###############################################
############# 4. Synthesizing #################
###############################################
# Function which will synthesize the final answer for the user
# When supervisor decides done
def synthesize(state: State) -> dict:
    gathered = "\n\n".join(state["results"])
    response = llm.invoke(
        f"Answer using ONLY the gathered information below. Do not add facts "
        f"from your own knowledge. If part of the question is not covered by "
        f"the gathered information, say that part could not be answered.\n\n"
        f"Question: {state['question']}\n\n"
        f"Gathered information:\n{gathered}"
    )
    return {"answer": response.content}


###############################################
############### 4. Wiring #####################
###############################################
# The conditional edge function reads state['route'] and returns the name of
# the next node to run.
# def pick_specialist(state: State) -> Literal["docs", "utility"]:
#     return state["route"]

# multi-route function - decide what to do next
# def route_decision(state: State) -> Literal["docs", "utility", "synthesize"]:
#     if state["route"] == "FINISH":
#         return "synthesize"
#     return state["route"]   # "docs" or "utility"

def route_decision(state: State) -> Literal["docs", "utility", "synthesize"]:
    idx = state["next_index"]
    if idx >= len(state["plan"]):
        return "synthesize"
    return state["plan"][idx]["specialist"]




builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("docs", docs_node)
builder.add_node("utility", utility_node)
# new node - when done to generate answer
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "supervisor")

# Supervisor decides - pick a specialist or finish
builder.add_conditional_edges("supervisor", route_decision)

# adjust - now edge goes back to supervisor and not end
builder.add_edge("docs", "supervisor")
builder.add_edge("utility", "supervisor")

# synthesise final answer when done
builder.add_edge('synthesize', END)

graph = builder.compile()


if __name__ == "__main__":
    # A cross-domain question single routing could NOT handle: needs both specialists.
    q = ("Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31, "
         "and tell me the refund policy on subscriptions.")
    print(f"=== Q: {q}\n")
    out = graph.invoke({"question": q, "results": []}, {"recursion_limit": 10})
    print(f"\n[final answer]\n{out['answer']}")
    print(f"\n[results gathered: {len(out['results'])}]")
    for r in out["results"]:
        print("  -", r[:80])



#######################################
############# EXAMPLE RESULT ##########

# === Q: Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31, and tell me the refund policy on subscriptions.

# [supervisor] planned 2 sub-task(s):
#     -> utility: 'Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31'
#     -> docs: 'Tell me the refund policy on subscriptions'
#   [utility received]: 'Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31'
#   [docs received]: 'Tell me the refund policy on subscriptions'

# [final answer]
# Transactions for ACC-001 (2026-05-01 to 2026-05-31):
# - 2026-05-02  Spotify              -9.99   subscriptions (posted)
# - 2026-05-04  Tesco               -54.20   groceries     (posted)
# - 2026-05-09  Salary - Tradu     3200.00   income        (posted)
# - 2026-05-15  Shell               -71.45   transport     (posted)
# - 2026-05-21  Amazon             -129.99   shopping      (posted)

# Refund policy on subscriptions:
# - Subscription charges are refundable only within 14 days of the billing date.
# - To request a refund, contact support with your transaction reference.
# - Refunds are issued to the original payment method.
# - Refunds are processed within 5 business days of approval.
# - Disputed transactions are investigated within 30 days.

# [results gathered: 2]
#   - [utility] Account ACC-001 — transactions (2026-05-01 to 2026-05-31):
# 2026-05-02 
#   - [docs] - Subscription charges are refundable only within 14 days of the billing 

