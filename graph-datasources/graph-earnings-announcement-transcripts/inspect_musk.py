import json

def inspect_musk():
    print("Loading export...")
    with open("memgraph_export.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Find Elon Node(s)
    musk_ids = []
    print("\n--- Nodes Matching 'Elon' ---")
    nodes_map = {}
    for n in data['nodes']:
        nodes_map[n['id']] = n
        props = n.get('properties', {})
        name = props.get('canonical_name', '')
        if 'Elon' in name:
            print(f"Found Node: ID={n['id']}, Labels={n['labels']}, Props={props}")
            musk_ids.append(n['id'])

    if not musk_ids:
        print("❌ No 'Elon' node found!")
        return

    # 2. Find Relationships attached to these IDs
    print(f"\n--- Analysis of Relationships for IDs: {musk_ids} ---")
    for r in data['relationships']:
        start = r['start_node_id']
        end = r['end_node_id']
        
        if start in musk_ids or end in musk_ids:
            direction = "-->" if start in musk_ids else "<--"
            other_id = end if start in musk_ids else start
            other_node = nodes_map.get(other_id)
            other_name = other_node.get('properties', {}).get('canonical_name', 'UNKNOWN') if other_node else "UNKNOWN"
            other_label = other_node.get('labels', ['UNKNOWN'])[0] if other_node else "UNKNOWN"
            
            print(f"ID {start if start in musk_ids else end} {direction} [{r['type']}] {direction} ({other_label}: {other_name})")
            print(f"   Props: {r.get('properties')}")

if __name__ == "__main__":
    inspect_musk()
