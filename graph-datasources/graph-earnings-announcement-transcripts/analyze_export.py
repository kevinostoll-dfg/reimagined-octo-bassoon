import json
from collections import Counter

def analyze_export():
    print("Loading memgraph_export.json...")
    with open("memgraph_export.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nodes = {n['id']: n for n in data['nodes']}
    rels = data['relationships']
    
    print(f"\nTotal Nodes: {len(nodes)}")
    print(f"Total Relationships: {len(rels)}")
    
    # 1. Relationship Types
    rel_types = Counter(r['type'] for r in rels)
    print("\nRelationship Types:")
    for r_type, count in rel_types.most_common():
        print(f"  {r_type}: {count}")
        
    # 2. Extract some examples of meaningful relationships
    print("\n--- Sample 'SVO_TRIPLE' (Business Actions) ---")
    svo_rels = [r for r in rels if r['type'] == 'SVO_TRIPLE'][:10]
    for r in svo_rels:
        start = nodes[r['start_node_id']]
        end = nodes[r['end_node_id']]
        verb = r['properties'].get('verb', 'unknown')
        print(f"  {start['properties'].get('canonical_name')} -[{verb}]-> {end['properties'].get('canonical_name')}")

    print("\n--- Sample 'SAID' (Speaker Statements) ---")
    said_rels = [r for r in rels if r['type'] == 'SAID'][:5]
    for r in said_rels:
        start = nodes[r['start_node_id']]
        end = nodes[r['end_node_id']]
        # Statement text might be long, truncate it
        stmt_text = end['properties'].get('text', '')[:100] + "..."
        print(f"  {start['properties'].get('canonical_name')} SAID: \"{stmt_text}\"")

if __name__ == "__main__":
    analyze_export()
