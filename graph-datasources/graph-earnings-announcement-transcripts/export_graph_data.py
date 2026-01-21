import json
import datetime
from gqlalchemy import Memgraph

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
HOST = "test-memgraph.blacksmith.deerfieldgreen.com"
PORT = 7687
USER = "memgraphdb"
PASSWORD = ""  # ⚠️ ENTER PASSWORD HERE BEFORE RUNNING

OUTPUT_FILE = "memgraph_export.json"

# --------------------------------------------------------------------------
# SERIALIZER
# --------------------------------------------------------------------------
def default_converter(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    return str(o)

# --------------------------------------------------------------------------
# MAIN EXPORT SCRIPT
# --------------------------------------------------------------------------
def export_graph():
    print(f"🔌 Connecting to {HOST}:{PORT} as {USER}...")
    
    try:
        memgraph = Memgraph(host=HOST, port=PORT, username=USER, password=PASSWORD)
        
        # 1. Fetch All Nodes
        print("📥 Fetching nodes...")
        # We use a simple query. For larger graphs, you'd want to paginate.
        # Given the small size (56 statements), this is safe.
        nodes_result = list(memgraph.execute_and_fetch("MATCH (n) RETURN n"))
        
        nodes = []
        for row in nodes_result:
            node = row['n']
            # GQLAlchemy Node object needs to be converted to dict
            node_data = {
                "id": getattr(node, "_id", None),
                "labels": list(getattr(node, "_labels", [])),
                "properties": getattr(node, "_properties", {})
            }
            nodes.append(node_data)
            
        print(f"   ✅ Retrieved {len(nodes)} nodes.")

        # 2. Fetch All Relationships
        print("📥 Fetching relationships...")
        rels_result = list(memgraph.execute_and_fetch("MATCH ()-[r]->() RETURN r, startNode(r) as start, endNode(r) as end"))
        
        relationships = []
        for row in rels_result:
            rel = row['r']
            start_node = row['start']
            end_node = row['end']
            
            rel_data = {
                "id": getattr(rel, "_id", None),
                "type": getattr(rel, "_type", "UNKNOWN"),
                "start_node_id": getattr(start_node, "_id", None),
                "end_node_id": getattr(end_node, "_id", None),
                "properties": getattr(rel, "_properties", {})
            }
            relationships.append(rel_data)
            
        print(f"   ✅ Retrieved {len(relationships)} relationships.")

        # 3. Save to JSON
        data = {
            "meta": {
                "host": HOST,
                "exported_at": datetime.datetime.now().isoformat(),
                "node_count": len(nodes),
                "relationship_count": len(relationships)
            },
            "nodes": nodes,
            "relationships": relationships
        }
        
        print(f"💾 Saving to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=default_converter)
            
        print("✅ Done! File saved.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_graph()
