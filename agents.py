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

# import scripts
from tools import lookup_fx_rate, get_transaction_history, search_documents

########################################################
################## Load env ############################
########################################################
load_dotenv(find_dotenv(usecwd=False), override=True)
path = find_dotenv(usecwd=False)
print(f"DEBUG: Loading .env from: {os.path.abspath(path)}")

########################################################
####################### MODEL ##########################
########################################################

# DEV ONLY - ssl verification disabled
llm = ChatOpenAI(model       = "gpt-5-mini",
                 http_client = httpx.Client(verify=False),
                 max_retries = 5,
                 timeout     = 60
                 )
#resp = llm.invoke("Reply with exactly: connection works")
#print(resp.content)




########################################################
###################### AGENT 1 #########################
######### Utility (FX) /transactions ###################
########################################################
utility_agent = create_agent(
                    model         = llm,
                    tools         = [lookup_fx_rate, get_transaction_history],
                    system_prompt = (
                                    "You are a transactions and currency specialist. Use the FX-rate tool for "
                                    "currency questions and the transaction-history tool for account activity. "
                                    "When reporting transactions, ALWAYS include the account number and date range "
                                    "exactly as returned by the tool. Do not omit identifying details. "
                                    "Answer concisely; do not offer follow-ups."
                                    )
                    )


########################################################
###################### AGENT 2 #########################
################### Doc retrieval ######################
########################################################
docs_agent = create_agent(
                    model         = llm,
                    tools         = [search_documents],
                    system_prompt = (
                                    "You are a policy specialist. Answer questions about fees, refunds, "
                                    "account types, and policies using ONLY the document-search tool. "
                                    "Answer concisely with just the requested information; do not offer follow-ups."
                    ),
                )


if __name__ == "__main__":
    print("=== DOCS AGENT: 'How long do refunds take?' ===")
    r1 = docs_agent.invoke({"messages": [("user", "How long do refunds take?")]})
    print(r1["messages"][-1].content)
 
    print("\n=== UTILITY AGENT: 'Show me ACC-001 transactions in May 2026' ===")
    r2 = utility_agent.invoke({"messages": [
        ("user", "Show me transactions for ACC-001 between 2026-05-01 and 2026-05-31")
    ]})
    print(r2["messages"][-1].content)
 
    print("\n=== UTILITY AGENT: 'What's GBP to USD?' ===")
    r3 = utility_agent.invoke({"messages": [("user", "What's GBP to USD?")]})
    print(r3["messages"][-1].content)
