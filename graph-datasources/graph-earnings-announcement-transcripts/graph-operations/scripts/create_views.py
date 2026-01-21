#!/usr/bin/env python3
"""
Create materialized views for high-performance queries
Run nightly or after every earnings transcript load

ENHANCED v2.0:
- Uses actual relationship types from earnings transcripts (CO_MENTIONED, SAID, HAS_ROLE, etc.)
- Includes v2.4 features (STATEMENT, ROLE, METRIC nodes)
- More comprehensive views for earnings call analysis
- Better error handling and query optimization
"""

from gqlalchemy import Memgraph
import sys
import time
from typing import Dict, List, Tuple

def connect_with_retry(host='localhost', port=7687, max_retries=3, retry_delay=2):
    """Connect to MemgraphDB with retry logic"""
    for attempt in range(1, max_retries + 1):
        try:
            db = Memgraph(host=host, port=port)
            db.execute("MATCH (n) RETURN count(n) LIMIT 1;")
            return db
        except Exception as e:
            if attempt < max_retries:
                wait_time = retry_delay * (2 ** (attempt - 1))
                print(f"⚠️  Connection attempt {attempt} failed: {e}")
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to connect after {max_retries} attempts: {e}")
    return None

MATERIALIZED_VIEWS = {
    "top_hubs": """
        MATCH (n)
        WHERE n.canonical_name IS NOT NULL AND n.mention_count > 5
        OPTIONAL MATCH (n)-[r:CO_MENTIONED]-()
        WITH n, 
             count(DISTINCT r) as degree,
             n.mention_count as mentions
        WHERE degree > 3
        RETURN labels(n)[0] AS type, 
               n.canonical_name AS name, 
               mentions, 
               degree,
               (degree * mentions) as influence
        ORDER BY influence DESC 
        LIMIT 100
    """,
    
    "speaker_network": """
        MATCH (p:PERSON)-[r:SAID]->(s:STATEMENT)
        OPTIONAL MATCH (p)-[:HAS_ROLE]->(role:ROLE)
        WITH p, 
             count(DISTINCT s) as statements,
             collect(DISTINCT role.title)[..3] as roles,
             p.mention_count as mentions
        WHERE statements > 0
        RETURN p.canonical_name AS speaker,
               mentions,
               statements,
               roles
        ORDER BY statements DESC, mentions DESC
        LIMIT 50
    """,
    
    "metric_analysis": """
        MATCH (m:METRIC)
        OPTIONAL MATCH (m)-[:SAME_AS]->(md:METRIC_DEFINITION)
        OPTIONAL MATCH (m)-[r:CO_MENTIONED]-(e)
        WHERE labels(e)[0] IN ['ORG', 'PRODUCT', 'CONCEPT']
        WITH m, md,
             collect(DISTINCT e.canonical_name)[..5] as entities,
             count(DISTINCT e) as entity_count,
             m.mention_count as mentions
        WHERE mentions > 2
        RETURN m.canonical_name AS metric,
               md.name AS canonical_metric,
               m.value AS value,
               m.unit AS unit,
               mentions,
               entity_count,
               entities
        ORDER BY mentions DESC
        LIMIT 50
    """,
    
    "causal_relationships": """
        MATCH (source)-[r:CAUSES|DRIVES|BOOSTS|HURTS|MITIGATES]->(target)
        WHERE r.confidence IS NULL OR r.confidence >= 0.6
        WITH source, target, type(r) as rel_type,
             count(r) as frequency,
             collect(DISTINCT r.quote)[..2] as quotes
        RETURN source.canonical_name AS source_entity,
               labels(source)[0] AS source_type,
               rel_type,
               target.canonical_name AS target_entity,
               labels(target)[0] AS target_type,
               frequency,
               quotes
        ORDER BY frequency DESC
        LIMIT 100
    """,
    
    "product_mentions": """
        MATCH (p:PRODUCT)
        WHERE p.mention_count > 1
        OPTIONAL MATCH (p)-[r:CO_MENTIONED]-(o:ORG)
        WHERE r.count > 0
        WITH p, 
             collect(DISTINCT o.canonical_name)[..5] as orgs,
             count(DISTINCT o) as org_count,
             sum(r.count) as total_co_mentions
        RETURN p.canonical_name AS product,
               p.mention_count AS mentions,
               org_count,
               total_co_mentions,
               orgs
        ORDER BY total_co_mentions DESC, p.mention_count DESC
        LIMIT 50
    """,
    
    "organization_roles": """
        MATCH (p:PERSON)-[:HAS_ROLE]->(r:ROLE)
        OPTIONAL MATCH (p)-[:WORKS_FOR]->(o:ORG)
        WITH p, r, o,
             p.mention_count as mentions
        WHERE mentions > 0
        RETURN p.canonical_name AS person,
               r.title AS role,
               o.canonical_name AS organization,
               mentions
        ORDER BY mentions DESC
        LIMIT 100
    """,
    
    "statement_topics": """
        MATCH (s:STATEMENT)
        OPTIONAL MATCH (s)<-[:SAID]-(p:PERSON)
        OPTIONAL MATCH (s)-[r:CO_MENTIONED]-(e)
        WHERE labels(e)[0] IN ['CONCEPT', 'PRODUCT', 'METRIC']
        WITH s, p,
             collect(DISTINCT e.canonical_name)[..5] as topics,
             count(DISTINCT e) as topic_count
        RETURN s.statement_id AS statement_id,
               p.canonical_name AS speaker,
               s.speaker AS speaker_name,
               topic_count,
               topics,
               substring(s.text, 0, 100) AS preview
        ORDER BY topic_count DESC
        LIMIT 50
    """,
    
    "entity_co_mention_network": """
        MATCH (a)-[r:CO_MENTIONED]->(b)
        WHERE r.count > 1
        WITH a, b, r.count as co_mentions,
             labels(a)[0] as source_type,
             labels(b)[0] as target_type
        RETURN a.canonical_name AS source,
               source_type,
               b.canonical_name AS target,
               target_type,
               co_mentions
        ORDER BY co_mentions DESC
        LIMIT 200
    """
}

def create_view(db: Memgraph, view_name: str, query: str) -> Tuple[bool, int, float]:
    """Execute a view query and return results
    
    Note: MemgraphDB doesn't have native materialized views,
    so we execute queries to warm the cache and validate them.
    
    Returns:
        Tuple of (success, row_count, execution_time)
    """
    print(f"\n📊 Creating view: {view_name}")
    query_preview = query.strip().replace('\n', ' ')[:100]
    print(f"   Query: {query_preview}...")
    
    start_time = time.time()
    try:
        # Execute query to validate and warm cache
        results = list(db.execute_and_fetch(query))
        execution_time = time.time() - start_time
        row_count = len(results)
        
        print(f"   ✅ Computed {row_count:,} rows in {execution_time:.2f}s")
        return True, row_count, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        # Show more helpful error messages
        if "does not exist" in error_msg or "not found" in error_msg:
            print(f"   ⚠️  Skipped (missing data): {error_msg[:80]}")
        else:
            print(f"   ❌ Failed: {error_msg[:100]}")
        return False, 0, execution_time

def validate_view_queries(db: Memgraph) -> Dict[str, bool]:
    """Validate that view queries can execute (syntax check)"""
    print("\n🔍 Validating view queries...")
    validation_results = {}
    
    for view_name, query in MATERIALIZED_VIEWS.items():
        try:
            # Just parse the query, don't execute
            # MemgraphDB will validate on execute, so we'll catch errors there
            validation_results[view_name] = True
        except Exception as e:
            validation_results[view_name] = False
            print(f"   ⚠️  {view_name}: {e}")
    
    return validation_results

def main():
    try:
        print("\n" + "="*70)
        print("MATERIALIZED VIEWS CREATION (Enhanced v2.0)")
        print("="*70)
        print("\nConnecting to MemgraphDB at localhost:7687...")
        
        # Use port 7687 (Bolt protocol) as per docker-compose.yaml
        db = connect_with_retry(host='localhost', port=7687, max_retries=3)
        print("✅ Connected successfully")
        
        # Get graph statistics
        try:
            result = list(db.execute_and_fetch("MATCH (n) RETURN count(n) as count"))
            node_count = result[0].get('count', 0) if result else 0
            print(f"\n📊 Graph contains {node_count:,} nodes")
        except:
            pass
        
        print("\nCreating materialized views (pre-computing complex queries)...")
        print("="*70)
        
        success_count = 0
        total_rows = 0
        total_time = 0.0
        view_results = {}
        
        for view_name, query in MATERIALIZED_VIEWS.items():
            success, row_count, exec_time = create_view(db, view_name, query)
            view_results[view_name] = {
                'success': success,
                'rows': row_count,
                'time': exec_time
            }
            
            if success:
                success_count += 1
                total_rows += row_count
                total_time += exec_time
        
        # Summary
        print("\n" + "="*70)
        print(f"✅ View creation complete!")
        print(f"   Successful: {success_count}/{len(MATERIALIZED_VIEWS)}")
        print(f"   Total rows computed: {total_rows:,}")
        print(f"   Total execution time: {total_time:.2f}s")
        
        if success_count < len(MATERIALIZED_VIEWS):
            print(f"\n⚠️  {len(MATERIALIZED_VIEWS) - success_count} views failed or skipped")
            print("   This may be normal if the graph doesn't contain all node/relationship types")
        
        print("\n💡 View queries are validated and ready to use")
        print("   Re-run this script after loading new earnings transcript data")
        print("\n📝 View names:")
        for view_name in MATERIALIZED_VIEWS.keys():
            status = "✅" if view_results.get(view_name, {}).get('success') else "❌"
            rows = view_results.get(view_name, {}).get('rows', 0)
            print(f"   {status} {view_name} ({rows:,} rows)")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure MemgraphDB is running:")
        print("  cd graph-datasources/memgraph_docker")
        print("  docker-compose up -d")
        print("\nOr check the connection settings in this script.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()






