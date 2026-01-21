#!/usr/bin/env python3
"""
Memgraph Explorer v3 - Enhanced Graph Analysis & Test Query Generator
"""

import sys
import os

# Add parent directory to path for config import (now in scripts/ subdirectory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from neo4j import GraphDatabase
from tabulate import tabulate
import json
from collections import Counter
from datetime import datetime
import os

def explore_memgraph():
    """Connect to Memgraph and provide deep insights into data structure and quality."""
    
    print("\n=== Memgraph Database Explorer v3 ===\n")
    
    driver = GraphDatabase.driver(
        config.MEMGRAPH_URI,
        auth=(config.MEMGRAPH_USER, config.MEMGRAPH_PASSWORD)
    )
    
    test_queries = []  # Collect test queries for export
    
    with driver.session() as session:
        print(">> Connected to Memgraph\n")
        
        # --- 1. High-Level Stats (Enhanced) ---
        print("== Database Overview ==")
        print("-" * 60)
        
        counts = session.run("""
            MATCH (n) 
            WITH COUNT(n) as node_count
            MATCH ()-[r]->() 
            RETURN node_count, COUNT(r) as rel_count
        """).single()
        
        print(f"Total Nodes:         {counts['node_count']:,}")
        print(f"Total Relationships: {counts['rel_count']:,}")
        
        # Calculate average degree
        avg_degree = (counts['rel_count'] * 2) / counts['node_count'] if counts['node_count'] > 0 else 0
        print(f"Avg Node Degree:     {avg_degree:.2f}")
        
        # Index Check
        indexes = session.run("SHOW INDEX INFO").data()
        print(f"Active Indexes:      {len(indexes)}")
        if indexes:
            print("\nIndex Details:")
            for idx in indexes[:5]:  # Show first 5
                print(f"  - {idx.get('index name', 'N/A')}")
        print("-" * 60 + "\n")

        # --- 2. Detailed Node Analysis (Enhanced with Samples) ---
        print("== Node Labels & Properties (with Samples) ==")
        print("-" * 80)
        
        node_stats = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as Label, 
                   COUNT(*) as Count, 
                   keys(n)[0..5] as Top_Keys
            ORDER BY Count DESC
        """)
        
        node_table = []
        sample_data = {}  # Store sample nodes for each label
        
        for r in node_stats:
            label = r["Label"]
            count = r['Count']
            keys = r["Top_Keys"] if r["Top_Keys"] else []
            
            node_table.append([
                label, 
                f"{count:,}", 
                ", ".join(keys[:5]) if keys else "(no props)"
            ])
            
            # Get sample nodes for this label
            if count > 0:
                samples = session.run(f"""
                    MATCH (n:{label})
                    RETURN n
                    LIMIT 3
                """).data()
                
                sample_data[label] = []
                for sample in samples:
                    node = sample['n']
                    props = dict(node)
                    # Show key properties
                    sample_props = {}
                    for key in ['canonical_name', 'name', 'title', 'text']:
                        if key in props:
                            val = props[key]
                            if isinstance(val, str) and len(val) > 50:
                                val = val[:47] + "..."
                            sample_props[key] = val
                    if not sample_props and props:
                        # Show first 2 properties
                        for k, v in list(props.items())[:2]:
                            if isinstance(v, str) and len(v) > 50:
                                v = v[:47] + "..."
                            sample_props[k] = v
                    if sample_props:
                        sample_data[label] = sample_props
                        break  # Just one sample per label
        
        print(tabulate(node_table, headers=["Label", "Count", "Key Properties"], tablefmt="simple"))
        
        # Show sample data
        if sample_data:
            print("\n>> Sample Data:")
            for label, samples in sample_data.items():
                if samples:
                    print(f"\n  {label}:")
                    for key, value in samples.items():
                        print(f"    {key}: {value}")
        
        print("-" * 80 + "\n")

        # --- 3. Property Value Analysis ---
        print("== Property Value Analysis ==")
        print("-" * 80)
        
        # Analyze canonical_name values for each node type
        for label in ['PERSON', 'ORG', 'CONCEPT', 'STATEMENT']:
            result = session.run(f"""
                MATCH (n:{label})
                WHERE n.canonical_name IS NOT NULL
                RETURN n.canonical_name as name
                LIMIT 10
            """).data()
            
            if result:
                names = [r['name'] for r in result]
                print(f"\n{label} canonical_name samples:")
                for i, name in enumerate(names[:5], 1):
                    print(f"  {i}. {name}")
                if len(names) > 5:
                    print(f"  ... and {len(names) - 5} more")
                
                # Generate test query
                test_queries.append({
                    "query": f"Find all {label} nodes",
                    "cypher": f"MATCH (n:{label}) RETURN n LIMIT 10",
                    "type": "Simple",
                    "difficulty": "Easy"
                })
        
        print("-" * 80 + "\n")

        # --- 4. Relationship Matrix (Enhanced) ---
        print("== Connectivity Analysis ==")
        print("-" * 80)
        
        rel_stats = session.run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] as Source, 
                   type(r) as Relationship, 
                   labels(b)[0] as Target, 
                   COUNT(*) as Strength
            ORDER BY Strength DESC
            LIMIT 20
        """)
        
        rel_table = []
        for r in rel_stats:
            rel_table.append([
                f"(:{r['Source']})", 
                f"-[:{r['Relationship']}]->", 
                f"(:{r['Target']})", 
                f"{r['Strength']:,}"
            ])
            
        print(tabulate(rel_table, headers=["Source Node", "Relationship", "Target Node", "Count"], tablefmt="simple"))
        print("-" * 80 + "\n")

        # --- 5. Multi-hop Path Discovery ---
        print("== Multi-hop Path Discovery ==")
        print("-" * 80)
        
        # Find interesting 2-hop paths
        paths_2hop = session.run("""
            MATCH path = (a)-[*2]-(b)
            WHERE a <> b
            RETURN labels(a)[0] as StartLabel,
                   labels(b)[0] as EndLabel,
                   [r in relationships(path) | type(r)] as RelTypes,
                   COUNT(*) as PathCount
            ORDER BY PathCount DESC
            LIMIT 10
        """).data()
        
        if paths_2hop:
            print("\nTop 2-hop Path Patterns:")
            for i, path in enumerate(paths_2hop[:5], 1):
                rels = " -> ".join(path['RelTypes'])
                print(f"  {i}. (:{path['StartLabel']}) -[{rels}]-> (:{path['EndLabel']}) [{path['PathCount']:,} paths]")
        
        # Find interesting 3-hop paths
        paths_3hop = session.run("""
            MATCH path = (a)-[*3]-(b)
            WHERE a <> b
            RETURN labels(a)[0] as StartLabel,
                   labels(b)[0] as EndLabel,
                   [r in relationships(path) | type(r)] as RelTypes,
                   COUNT(*) as PathCount
            ORDER BY PathCount DESC
            LIMIT 5
        """).data()
        
        if paths_3hop:
            print("\nTop 3-hop Path Patterns:")
            for i, path in enumerate(paths_3hop[:3], 1):
                rels = " -> ".join(path['RelTypes'])
                print(f"  {i}. (:{path['StartLabel']}) -[{rels}]-> (:{path['EndLabel']}) [{path['PathCount']:,} paths]")
        
        print("-" * 80 + "\n")

        # --- 6. Content Analysis (Enhanced) ---
        print("== Content Analysis (Statements & Transcripts) ==")
        print("-" * 80)
        
        stmt_count = session.run("MATCH (s:STATEMENT) RETURN count(s) as c").single()["c"]
        if stmt_count > 0:
            print(f"Found {stmt_count:,} STATEMENTS. Analyzing content...")
            
            # Find most mentioned entities
            mentions = session.run("""
                MATCH (s:STATEMENT)-[:MENTIONS]->(e)
                RETURN labels(e)[0] as Type, e.canonical_name as Name, COUNT(*) as Mentions
                ORDER BY Mentions DESC
                LIMIT 10
            """)
            
            print("\n>> Top Mentioned Entities:")
            mention_list = []
            for m in mentions:
                mention_list.append(f"  - {m['Name']} ({m['Type']}): {m['Mentions']:,} mentions")
                print(mention_list[-1])
            
            # Find key speakers with their statement counts
            speakers = session.run("""
                MATCH (p:PERSON)-[:SAID]->(s:STATEMENT)
                RETURN p.canonical_name as Speaker, COUNT(*) as Lines
                ORDER BY Lines DESC
                LIMIT 10
            """)
            
            print("\n>> Top Speakers:")
            speaker_list = []
            for s in speakers:
                speaker_list.append(f"  - {s['Speaker']}: {s['Lines']:,} statements")
                print(speaker_list[-1])
                
                # Generate test queries
                test_queries.append({
                    "query": f"What did {s['Speaker']} say?",
                    "cypher": f"MATCH (p:PERSON {{canonical_name: '{s['Speaker']}'}})-[:SAID]->(s:STATEMENT) RETURN s.text LIMIT 10",
                    "type": "Person -> Statement",
                    "difficulty": "Medium"
                })
            
            # Analyze statement text for keywords
            keywords = session.run("""
                MATCH (s:STATEMENT)
                WHERE s.text IS NOT NULL
                RETURN s.text as text
                LIMIT 100
            """).data()
            
            if keywords:
                # Extract common words (simple approach)
                all_words = []
                for k in keywords:
                    words = k['text'].lower().split()
                    all_words.extend([w for w in words if len(w) > 4])
                
                word_freq = Counter(all_words)
                top_words = word_freq.most_common(10)
                
                print("\n>> Common Keywords in Statements:")
                for word, count in top_words:
                    print(f"  - '{word}': appears {count} times")
                    
                    # Generate test query
                    test_queries.append({
                        "query": f"Find statements mentioning '{word}'",
                        "cypher": f"MATCH (s:STATEMENT) WHERE s.text CONTAINS '{word}' RETURN s.text LIMIT 10",
                        "type": "Text Search",
                        "difficulty": "Easy"
                    })
        else:
            print("No STATEMENT nodes found.")
            
        print("-" * 80 + "\n")

        # --- 7. Data Quality Checks ---
        print("== Data Quality Analysis ==")
        print("-" * 80)
        
        # Check for missing canonical_name
        missing_names = session.run("""
            MATCH (n)
            WHERE n.canonical_name IS NULL AND (n:PERSON OR n:ORG OR n:CONCEPT)
            RETURN labels(n)[0] as Label, COUNT(*) as Count
        """).data()
        
        if missing_names:
            print("\n>> Nodes Missing canonical_name:")
            for m in missing_names:
                print(f"  - {m['Label']}: {m['Count']:,} nodes")
        
        # Check for nodes with no relationships
        isolated = session.run("""
            MATCH (n)
            WHERE NOT (n)--()
            RETURN labels(n)[0] as Label, COUNT(*) as Count
            ORDER BY Count DESC
        """).data()
        
        if isolated:
            print("\n>> Isolated Nodes (no relationships):")
            for iso in isolated[:5]:
                print(f"  - {iso['Label']}: {iso['Count']:,} nodes")
        
        # Check property completeness
        print("\n>> Property Completeness:")
        for label in ['PERSON', 'ORG', 'CONCEPT']:
            total = session.run(f"MATCH (n:{label}) RETURN COUNT(n) as c").single()["c"]
            with_name = session.run(f"MATCH (n:{label}) WHERE n.canonical_name IS NOT NULL RETURN COUNT(n) as c").single()["c"]
            if total > 0:
                pct = (with_name / total) * 100
                print(f"  - {label}: {pct:.1f}% have canonical_name ({with_name:,}/{total:,})")
        
        print("-" * 80 + "\n")

        # --- 8. Enhanced Query Suggestions ---
        print("== Test Query Suggestions (Validated) ==")
        print("-" * 80)
        
        suggestions = []
        
        # Get actual person names for queries
        persons = session.run("""
            MATCH (p:PERSON)
            WHERE p.canonical_name IS NOT NULL
            RETURN p.canonical_name as name
            LIMIT 5
        """).data()
        
        for person in persons:
            name = person['name']
            # Escape single quotes in names for Cypher queries
            name_escaped = name.replace("'", "\\'")
            
            suggestions.append({
                "query": f"What is {name} connected to?",
                "cypher": f"MATCH (p:PERSON {{canonical_name: '{name_escaped}'}})-[r]-(connected) RETURN type(r), labels(connected)[0], connected.canonical_name LIMIT 20",
                "type": "Person Connections",
                "difficulty": "Easy"
            })
            
            # Check if person has role
            has_role = session.run(f"""
                MATCH (p:PERSON {{canonical_name: $name}})-[:HAS_ROLE]->(r:ROLE)
                RETURN r.title as role
                LIMIT 1
            """, name=name).single()
            
            if has_role:
                role_title = has_role['role'].replace("'", "\\'")
                suggestions.append({
                    "query": f"What role does {name} have?",
                    "cypher": f"MATCH (p:PERSON {{canonical_name: '{name_escaped}'}})-[:HAS_ROLE]->(r:ROLE) RETURN r.title",
                    "type": "Person Role",
                    "difficulty": "Easy"
                })
                
                # Check if person made statements
                has_statements = session.run(f"""
                    MATCH (p:PERSON {{canonical_name: $name}})-[:SAID]->(s:STATEMENT)
                    RETURN COUNT(s) as count
                """, name=name).single()
                
                if has_statements and has_statements['count'] > 0:
                    suggestions.append({
                        "query": f"What did {name} say?",
                        "cypher": f"MATCH (p:PERSON {{canonical_name: '{name_escaped}'}})-[:SAID]->(s:STATEMENT) RETURN s.text LIMIT 10",
                        "type": "Person Statements",
                        "difficulty": "Medium"
                    })
        
        # Organization queries
        orgs = session.run("""
            MATCH (o:ORG)
            WHERE o.canonical_name IS NOT NULL
            RETURN o.canonical_name as name
            LIMIT 5
        """).data()
        
        for org in orgs:
            name = org['name']
            name_escaped = name.replace("'", "\\'")
            suggestions.append({
                "query": f"What is {name} connected to?",
                "cypher": f"MATCH (o:ORG {{canonical_name: '{name_escaped}'}})-[r]-(connected) RETURN type(r), labels(connected)[0], connected.canonical_name LIMIT 20",
                "type": "Organization Connections",
                "difficulty": "Easy"
            })
        
        # Role-based queries
        roles = session.run("""
            MATCH (p:PERSON)-[:HAS_ROLE]->(r:ROLE)
            RETURN r.title as role, COUNT(DISTINCT p) as person_count
            ORDER BY person_count DESC
            LIMIT 5
        """).data()
        
        for role_data in roles:
            role = role_data['role']
            role_escaped = role.replace("'", "\\'")
            suggestions.append({
                "query": f"Who has the role of {role}?",
                "cypher": f"MATCH (p:PERSON)-[:HAS_ROLE]->(r:ROLE {{title: '{role_escaped}'}}) RETURN p.canonical_name",
                "type": "Role Query",
                "difficulty": "Medium"
            })
        
        # Print suggestions
        for i, item in enumerate(suggestions[:20], 1):  # Limit to 20
            print(f"{i}. [{item['difficulty']}] {item['query']}")
            print(f"   Type: {item['type']}")
            print(f"   Cypher: {item['cypher']}")
            print()
        
        test_queries.extend(suggestions)
        
        print("-" * 80 + "\n")

        # --- 9. Export Test Queries ---
        if test_queries:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_queries_{timestamp}.json"
            
            export_data = {
                "generated_at": datetime.now().isoformat(),
                "database_stats": {
                    "total_nodes": counts['node_count'],
                    "total_relationships": counts['rel_count']
                },
                "queries": test_queries
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"== Test Queries Exported ==")
            print(f"Saved {len(test_queries)} queries to: {filename}")
            print("-" * 80 + "\n")
    
    driver.close()

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate for pretty printing...")
        import subprocess
        subprocess.check_call(["python", "-m", "pip", "install", "tabulate"])
        
    try:
        explore_memgraph()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")
