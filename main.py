
###### OBSOLETE #######
###### Initial agent test #####

# general imports
import os
import httpx
import logging
from dotenv import load_dotenv, find_dotenv

# agent libs
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

########################################################
################## Load env ############################
########################################################
load_dotenv(find_dotenv(usecwd=False), override=True)
path = find_dotenv(usecwd=False)
print(f"DEBUG: Loading .env from: {os.path.abspath(path)}")

########################################################
################# Mock tools ###########################
########################################################

# Tool to get currency pairs
@tool
def lookup_fx_rate(base: str, quote: str) -> str:
    """Look up the current FX rate for a currency pair, e.g. base='EUR', quote='USD'."""
    mock_rates = {
        ("EUR", "USD"): 1.147,
        ("EUR", "GBP"): 0.866,
        ("EUR", "JPY"): 185.0,
        ("EUR", "CHF"): 0.925,
        ("EUR", "CAD"): 1.624,
        ("EUR", "AUD"): 1.635,
        ("GBP", "USD"): 1.323,
        ("USD", "JPY"): 161.7,
        ("USD", "CAD"): 1.393,
        ("USD", "CHF"): 0.805,
        ("AUD", "USD"): 0.702,
        ("USD", "GBP"): 0.757,
    }

    # handle same order
    base, quote = base.upper(), quote.upper()
    if (base, quote) in mock_rates:
        return f"1 {base} = {mock_rates[(base, quote)]} {quote}"
    # handle reversed order (i.e. reversed in the dict)
    if (quote, base) in mock_rates:
        return f"1 {base} = {round(1 / mock_rates[(quote, base)], 4)} {quote}"
    return f"No rate available for {base}/{quote}."

########################################################


########################################################
####################### AGENT ##########################
########################################################

# Model
# DEV ONLY - ssl verification disabled
llm = ChatOpenAI(model       = "gpt-5-mini",
                 http_client = httpx.Client(verify=False))
#resp = llm.invoke("Reply with exactly: connection works")
#print(resp.content)

# ReAct Agent
agent = create_agent(
                model         = llm,
                tools         = [lookup_fx_rate],
                system_prompt = "You are a fintech assistant. Use tools to answer questions about currency rates.\
                          Answer concisely only with requested info and do not offer follow ups.",
)

# Run run run
#message = {"messages": [("user", "What's the EUR to USD rate?")]}
message = {"messages": [("user", "Convert 1000 EUR into USD")]}

if __name__ == "__main__":
    result = agent.invoke(message)
    print(result["messages"][-1].content)

    #show every step the agent took
    print("\n------------------------ full trace ------------")
    for m in result["messages"]:
        m.pretty_print()
