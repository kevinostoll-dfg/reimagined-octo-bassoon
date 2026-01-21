from gqlalchemy import Memgraph
import json
from datetime import date, datetime

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
HOST = "test-memgraph.blacksmith.deerfieldgreen.com"
PORT = 7687
USER = "memgraphdb"
PASSWORD = ""  # No password required as per user

# --------------------------------------------------------------------------
# QUERIES
# --------------------------------------------------------------------------
QUERIES = {
    "1. Leadership Analysis (Elon Musk - What is he saying?)": """
        MATCH (p:PERSON {canonical_name: "Elon Musk"})-[:SAID]->(s:STATEMENT)
        RETURN p.canonical_name as Speaker, s.text as Quote
        LIMIT 10;
    """,
    "2. Strategic Actions (Active Verbs)": """
        MATCH (subject)-[r:SVO_TRIPLE]->(object)
        WHERE r.verb IN ['increase', 'launch', 'produce', 'deliver', 'achieve', 'scale']
        RETURN subject.canonical_name as Actor, r.verb as Action, object.canonical_name as Target
        LIMIT 15;
    """,
    "3. Production Context (Who is talking about Production?)": """
        MATCH (n)-[r]->(target)
        WHERE target.canonical_name CONTAINS 'Production' OR target.canonical_name CONTAINS 'production'
        RETURN n.canonical_name as Subject, type(r) as Rel, target.canonical_name as Context
        LIMIT 15;
    """,
    "4. Financial Quantities (Revenue/Cash)": """
        MATCH (c:CONCEPT)-[:QUANTITY_OF]->(val)
        WHERE c.canonical_name CONTAINS 'revenue' OR c.canonical_name CONTAINS 'cash'
        RETURN c.canonical_name as Metric, val.text as Value
        LIMIT 15;
    """,
     "5. Top Co-occurrences (What appears together?)": """
        MATCH (a)-[r:CO_MENTIONED]->(b)
        WHERE r.count > 2 AND a.canonical_name <> b.canonical_name
        AND NOT a.canonical_name CONTAINS '8-K' 
        AND NOT b.canonical_name CONTAINS '8-K'
        RETURN a.canonical_name as Topic_A, b.canonical_name as Topic_B, r.count as Strength
        ORDER BY Strength DESC
        LIMIT 15;
    """
}

# --------------------------------------------------------------------------
# EXECUTION
# --------------------------------------------------------------------------
def converter(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return str(o)

def run_queries():
    print(f"🔌 Connecting to {HOST}:{PORT} as {USER}...")
    try:
        memgraph = Memgraph(host=HOST, port=PORT, username=USER, password=PASSWORD)
        
        for title, query in QUERIES.items():
            print(f"\n{'='*60}")
            print(f"🔎 RUNNING: {title}")
            print(f"{'='*60}")
            
            try:
                results = list(memgraph.execute_and_fetch(query))
                
                if not results:
                    print("   (No results found)")
                    continue
                    
                # Print as a nice formatted table-like structure (or JSON lines)
                keys = results[0].keys()
                header = " | ".join(f"{k:<25}" for k in keys)
                print(header)
                print("-" * len(header))
                
                for row in results:
                    line = " | ".join(f"{str(row[k])[:25]:<25}" for k in keys)
                    print(line)
                    
            except Exception as q_e:
                print(f"   ❌ Query Error: {q_e}")

    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")

if __name__ == "__main__":
    run_queries()
