"""Specialist agents, now sourced from the MCP server.

The ONLY change from the in-process version: tools are loaded from the MCP
server via MultiServerMCPClient instead of imported from tools.py. Because the
MCP client is async, building the agents happens inside an async function.

The graph (supervisor_multi.py) calls build_agents() once at startup to get the
two specialists, then runs unchanged.
"""

import httpx
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# DEV ONLY - ssl verify disabled
llm = ChatOpenAI(
    model             = "gpt-5-mini",
    http_client       = httpx.Client(verify=False),
    http_async_client = httpx.AsyncClient(verify=False),
    max_retries       = 5,
    timeout           = 60,
)


# Absolute path to the server script; the client spawns it as a subprocess (stdio).
SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")

_mcp_client = MultiServerMCPClient({
    "fintech": {
        "command": "python",
        "args": [SERVER_PATH],
        "transport": "stdio",
    }
})


async def build_agents():
    """Connect to the MCP server, load tools, and build the two specialists.

    Returns (docs_agent, utility_agent). Tools are partitioned by name so each
    specialist still only gets its own tools -- the scoping from before is preserved.
    """
    tools = await _mcp_client.get_tools()
    by_name = {t.name: t for t in tools}
    print(f"[mcp] loaded tools: {list(by_name)}")

    docs_tools = [by_name["search_documents"]]
    utility_tools = [by_name["lookup_fx_rate"], by_name["get_transaction_history"]]

    docs_agent = create_agent(
        model=llm,
        tools=docs_tools,
        system_prompt=(
            "You are a policy specialist. Answer questions about fees, refunds, "
            "account types, and policies using ONLY the document-search tool. "
            "Answer concisely; do not offer follow-ups."
        ),
    )
    utility_agent = create_agent(
        model=llm,
        tools=utility_tools,
        system_prompt=(
            "You are a transactions and currency specialist. Use the FX-rate tool for "
            "currency questions and the transaction-history tool for account activity. "
            "When reporting transactions, ALWAYS include the account number and date "
            "range exactly as returned by the tool. Do not omit identifying details. "
            "Answer concisely; do not offer follow-ups."
        ),
    )
    return docs_agent, utility_agent
