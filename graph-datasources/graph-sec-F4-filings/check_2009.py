#!/usr/bin/env python3
"""Quick check to see if 2009 data is available."""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv('env')

SEC_API_KEY = os.getenv("SEC_API_KEY")
SEC_API_ENDPOINT = os.getenv("SEC_API_ENDPOINT", "https://api.sec-api.io/insider-trading")

headers = {
    "Authorization": SEC_API_KEY,
    "Content-Type": "application/json"
}

# Check 2009 specifically
query_params = {
    "query": "issuer.tradingSymbol:TSLA AND periodOfReport:[2009-01-01 TO 2009-12-31]",
    "from": 0,
    "size": 50,
    "sort": [{"periodOfReport": {"order": "asc"}}]
}

print("Checking for 2009 data...")
response = requests.post(SEC_API_ENDPOINT, headers=headers, json=query_params, timeout=30)
result = response.json()

total = result.get("total", {})
print(f"2009 filings: {total.get('value', 0)}")

if total.get('value', 0) > 0:
    transactions = result.get("transactions", [])
    if transactions:
        earliest = transactions[0]
        print(f"Earliest 2009 filing: {earliest.get('periodOfReport')}")
        print(f"Filed at: {earliest.get('filedAt')}")
