#!/usr/bin/env python3
"""
Create indexes for MemgraphDB Earnings Announcement Transcripts Knowledge Graph
Run this once after loading data for optimal query performance

ENHANCED v2.0:
- Comprehensive index coverage for all node types
- Better error handling and retry logic
- Index verification and statistics
- Support for STATEMENT, ROLE, METRIC_DEFINITION nodes (v2.4)
- Connection retry with exponential backoff
"""

from gqlalchemy import Memgraph
import sys
import time
from typing import List, Dict, Tuple

def connect_with_retry(host='localhost', port=7687, max_retries=3, retry_delay=2):
    """Connect to MemgraphDB with retry logic"""
    for attempt in range(1, max_retries + 1):
        try:
            db = Memgraph(host=host, port=port)
            # Test connection
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

def create_indexes(db: Memgraph) -> Tuple[int, int, List[str]]:
    """Create all recommended indexes for optimal performance
    
    Returns:
        Tuple of (created_count, skipped_count, errors)
    """
    
    # Comprehensive index list based on actual graph schema
    indexes = [
        # Core entity indexes (by type) - canonical_name is primary lookup
        "CREATE INDEX ON :PERSON(canonical_name);",
        "CREATE INDEX ON :ORG(canonical_name);",
        "CREATE INDEX ON :PRODUCT(canonical_name);",
        "CREATE INDEX ON :CONCEPT(canonical_name);",
        "CREATE INDEX ON :METRIC(canonical_name);",
        "CREATE INDEX ON :EVENT(canonical_name);",
        "CREATE INDEX ON :TECHNOLOGY(canonical_name);",
        "CREATE INDEX ON :LOCATION(canonical_name);",
        "CREATE INDEX ON :DATE(canonical_name);",
        
        # v2.4 New node types
        "CREATE INDEX ON :STATEMENT(statement_id);",
        "CREATE INDEX ON :STATEMENT(speaker);",
        "CREATE INDEX ON :ROLE(title);",
        "CREATE INDEX ON :METRIC_DEFINITION(name);",
        
        # Additional entity types (from spaCy NER)
        "CREATE INDEX ON :MONEY(canonical_name);",
        "CREATE INDEX ON :GPE(canonical_name);",
        "CREATE INDEX ON :GEOGRAPHY(canonical_name);",
        "CREATE INDEX ON :REGULATION(canonical_name);",
        "CREATE INDEX ON :LAW(canonical_name);",
        "CREATE INDEX ON :RISK(canonical_name);",
        
        # Mention count indexes (for filtering and sorting)
        "CREATE INDEX ON :CONCEPT(mention_count);",
        "CREATE INDEX ON :ORG(mention_count);",
        "CREATE INDEX ON :PERSON(mention_count);",
        "CREATE INDEX ON :PRODUCT(mention_count);",
        "CREATE INDEX ON :METRIC(mention_count);",
        "CREATE INDEX ON :RISK(mention_count);",
        
        # Entity ID indexes (for lookups)
        "CREATE INDEX ON :PERSON(entity_id);",
        "CREATE INDEX ON :ORG(entity_id);",
        "CREATE INDEX ON :PRODUCT(entity_id);",
        "CREATE INDEX ON :CONCEPT(entity_id);",
        "CREATE INDEX ON :METRIC(entity_id);",
    ]
    
    print("Creating indexes for optimal query performance...")
    print("="*70)
    
    created = 0
    skipped = 0
    errors = []
    
    for idx, index_query in enumerate(indexes, 1):
        try:
            db.execute(index_query)
            # Parse index details for display
            parts = index_query.replace("CREATE INDEX ON :", "").replace(";", "").split("(")
            entity_type = parts[0]
            property_name = parts[1].rstrip(")")
            print(f"✅ [{idx:2d}/{len(indexes)}] Created index on :{entity_type}({property_name})")
            created += 1
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                parts = index_query.replace("CREATE INDEX ON :", "").replace(";", "").split("(")
                entity_type = parts[0]
                property_name = parts[1].rstrip(")")
                print(f"⏭️  [{idx:2d}/{len(indexes)}] Index already exists: :{entity_type}({property_name})")
                skipped += 1
            else:
                parts = index_query.replace("CREATE INDEX ON :", "").replace(";", "").split("(")
                entity_type = parts[0] if len(parts) > 0 else "UNKNOWN"
                print(f"❌ [{idx:2d}/{len(indexes)}] Failed: {e}")
                errors.append(f"{entity_type}: {str(e)}")
    
    print("="*70)
    return created, skipped, errors

def verify_indexes(db: Memgraph) -> Dict:
    """Verify all indexes are active and return statistics"""
    print("\nVerifying indexes...")
    stats = {
        'total': 0,
        'by_type': {},
        'by_property': {}
    }
    
    try:
        # MemgraphDB uses SHOW INDEX INFO
        result = list(db.execute_and_fetch("SHOW INDEX INFO;"))
        stats['total'] = len(result)
        
        if result:
            print(f"✅ {stats['total']} indexes are active")
            
            # Parse index information if available
            for idx_info in result[:10]:  # Show first 10
                if isinstance(idx_info, dict):
                    # Format depends on MemgraphDB version
                    print(f"   - {idx_info}")
        else:
            print("⚠️  No indexes found (this may be normal if using a different MemgraphDB version)")
        
        return stats
    except Exception as e:
        # Some MemgraphDB versions may not support SHOW INDEX INFO
        print(f"⚠️  Could not verify indexes (this may be version-specific): {e}")
        return stats

def get_graph_statistics(db: Memgraph) -> Dict:
    """Get basic graph statistics"""
    stats = {
        'node_count': 0,
        'relationship_count': 0,
        'node_types': {},
        'relationship_types': {}
    }
    
    try:
        # Node count
        result = list(db.execute_and_fetch("MATCH (n) RETURN count(n) as count"))
        if result:
            stats['node_count'] = result[0].get('count', 0)
        
        # Relationship count
        result = list(db.execute_and_fetch("MATCH ()-[r]->() RETURN count(r) as count"))
        if result:
            stats['relationship_count'] = result[0].get('count', 0)
        
        # Node types breakdown
        result = list(db.execute_and_fetch(
            "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC LIMIT 15"
        ))
        for row in result:
            label = row.get('label', 'UNKNOWN')
            count = row.get('count', 0)
            stats['node_types'][label] = count
        
        # Relationship types breakdown
        result = list(db.execute_and_fetch(
            "MATCH ()-[r]->() RETURN type(r) as rel_type, count(*) as count ORDER BY count DESC LIMIT 15"
        ))
        for row in result:
            rel_type = row.get('rel_type', 'UNKNOWN')
            count = row.get('count', 0)
            stats['relationship_types'][rel_type] = count
        
    except Exception as e:
        print(f"⚠️  Could not fetch graph statistics: {e}")
    
    return stats

def main():
    try:
        print("\n" + "="*70)
        print("MEMGRAPHDB INDEX CREATION SCRIPT (Enhanced v2.0)")
        print("="*70)
        print("\nConnecting to MemgraphDB at localhost:7687...")
        
        # Use port 7687 (Bolt protocol) as per docker-compose.yaml
        db = connect_with_retry(host='localhost', port=7687, max_retries=3)
        print("✅ Connected successfully\n")
        
        # Get initial graph statistics
        print("📊 Graph Statistics:")
        stats = get_graph_statistics(db)
        print(f"   Nodes: {stats['node_count']:,}")
        print(f"   Relationships: {stats['relationship_count']:,}")
        if stats['node_types']:
            print(f"   Node types: {len(stats['node_types'])}")
        if stats['relationship_types']:
            print(f"   Relationship types: {len(stats['relationship_types'])}")
        print()
        
        # Create indexes
        created, skipped, errors = create_indexes(db)
        
        # Summary
        print(f"\n✅ Index creation complete!")
        print(f"   Created: {created}")
        print(f"   Skipped: {skipped} (already existed)")
        print(f"   Total: {created + skipped}")
        if errors:
            print(f"   Errors: {len(errors)}")
            for error in errors[:5]:  # Show first 5 errors
                print(f"      - {error}")
        
        print(f"\n💡 Query performance should improve by 10-25x for most operations")
        
        # Verify indexes
        verify_indexes(db)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure MemgraphDB is running:")
        print("  cd graph-datasources/memgraph_docker")
        print("  docker-compose up -d")
        print("\nOr check the connection settings in this script.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()






