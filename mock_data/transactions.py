"""Mock transaction data.
The lookup function supports querying by account ID and an optional date range.
"""
 
from datetime import date
 
TRANSACTIONS = [
    {"account": "ACC-001", "date": "2026-05-02", "merchant": "Spotify",        "amount": -9.99,   "category": "subscriptions", "status": "posted"},
    {"account": "ACC-001", "date": "2026-05-04", "merchant": "Tesco",          "amount": -54.20,  "category": "groceries",     "status": "posted"},
    {"account": "ACC-001", "date": "2026-05-09", "merchant": "Salary - Tradu", "amount": 3200.00, "category": "income",        "status": "posted"},
    {"account": "ACC-001", "date": "2026-05-15", "merchant": "Shell",          "amount": -71.45,  "category": "transport",     "status": "posted"},
    {"account": "ACC-001", "date": "2026-05-21", "merchant": "Amazon",         "amount": -129.99, "category": "shopping",      "status": "posted"},
    {"account": "ACC-001", "date": "2026-06-01", "merchant": "Spotify",        "amount": -9.99,   "category": "subscriptions", "status": "posted"},
    {"account": "ACC-001", "date": "2026-06-03", "merchant": "British Gas",    "amount": -88.10,  "category": "utilities",     "status": "pending"},
 
    {"account": "ACC-002", "date": "2026-05-06", "merchant": "Uber",           "amount": -18.30,  "category": "transport",     "status": "posted"},
    {"account": "ACC-002", "date": "2026-05-12", "merchant": "Deliveroo",      "amount": -32.75,  "category": "dining",        "status": "posted"},
    {"account": "ACC-002", "date": "2026-05-19", "merchant": "Salary - Acme",  "amount": 2750.00, "category": "income",        "status": "posted"},
    {"account": "ACC-002", "date": "2026-05-28", "merchant": "Netflix",        "amount": -15.99,  "category": "subscriptions", "status": "posted"},
    {"account": "ACC-002", "date": "2026-06-04", "merchant": "Airbnb",         "amount": -420.00, "category": "travel",        "status": "pending"},
]


def in_range(transaction_date, start_date=None, end_date=None):
    """
    Function which checks if a transaction is within specified range and returns a bool
    """
    transaction_date = date.fromisoformat(transaction_date)
    if start_date and transaction_date < date.fromisoformat(start_date):
        return False
    if end_date and transaction_date > date.fromisoformat(end_date):
        return False
    return True


def query_transactions(account: str, start_date: str | None = None, end_date: str | None = None) -> list[dict]:
    """Return transactions for an account, optionally filtered to an inclusive date range.
       Dates are ISO strings 'YYYY-MM-DD'. If start/end are omitted, that bound is open.
    """


    return [
        tx for tx in TRANSACTIONS
        if tx["account"] == account and in_range(tx["date"], start_date, end_date)
    ]
 
