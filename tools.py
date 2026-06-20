from pathlib import Path
from langchain_core.tools import tool
 
from mock_data.transactions import query_transactions


########################################################
############### Tool for FX pairs ######################
########################################################
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
########### Tool for transaction history ###############
########################################################
@tool
def get_transaction_history(account: str, start_date: str = "", end_date: str = "") -> str:
    """Retrieve transactions for an account within an optional date range.
 
    account: account ID, e.g. 'ACC-001'.
    start_date / end_date: ISO dates 'YYYY-MM-DD'. Leave empty for an open bound.
    Returns one line per transaction with date, merchant, amount, category, and status.
    """
    rows = query_transactions(
                        account,
                        start_date or None,
                        end_date or None,
                        )
    if not rows:
        return f"No transactions found for {account} in the given range."
    lines = [
        # adding max width and alignment
        f"{r['date']}  {r['merchant']:<16} {r['amount']:>9.2f}  {r['category']:<13} ({r['status']})"
        for r in rows
    ]
    return f"Transactions for {account}:\n" + "\n".join(lines)

########################################################
########### Tool for document retrieval ################
########################################################

# Folder for the doc search
DOCS_DIR = Path(__file__).parent / "mock_data" / "docs"

@tool
def search_documents(query: str) -> str:
    """Search local policy documents for information relevant to the query.
 
    Scans the docs folder, scores each document by how many query words it contains,
    and returns the text of the best-matching document(s). Use for questions about
    fees, refunds, account types, and policies.
    """
    query_words = {word.lower().strip(".,?!") for word in query.split() if len(word) > 2}
    scored = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        text_words = text.lower()
        score = sum(1 for w in query_words if w in text_words)
        if score > 0:
            scored.append((score, path.name, text))
 
    if not scored:
        return "No relevant documents found."
 
    # highest score first
    scored.sort(reverse=True)
    # return max 2 best matches
    top = scored[:2]
    return "\n\n---\n\n".join(f"[{name}]\n{text.strip()}" for _, name, text in top)
