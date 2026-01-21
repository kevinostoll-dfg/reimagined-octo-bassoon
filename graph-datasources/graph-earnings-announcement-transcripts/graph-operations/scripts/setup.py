#!/usr/bin/env python3
"""
Complete setup script for Earnings Announcement Transcripts Knowledge Graph
Runs indexing and view materialization for optimal performance

ENHANCED v2.0:
- Correct port configuration (7687 for Bolt protocol)
- Better error handling and progress tracking
- Connection retry logic
- Comprehensive statistics

Run this ONCE after loading your earnings transcript data:
    python3 scripts/setup.py

Or individually:
    python3 scripts/create_indexes.py
    python3 scripts/create_views.py
"""

from gqlalchemy import Memgraph
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

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

def run_script(script_name: str) -> bool:
    """Run a setup script and return success status"""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            ['python3', str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Python3 not found or script not executable: {script_path}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running {script_name}: {e}")
        return False

def verify_graph_loaded(db: Memgraph) -> tuple[bool, Dict]:
    """Verify that graph data is loaded and return statistics"""
    stats = {
        'node_count': 0,
        'relationship_count': 0,
        'node_types': 0,
        'relationship_types': 0
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
        
        # Node types
        result = list(db.execute_and_fetch(
            "MATCH (n) RETURN count(DISTINCT labels(n)[0]) as count"
        ))
        if result:
            stats['node_types'] = result[0].get('count', 0)
        
        # Relationship types
        result = list(db.execute_and_fetch(
            "MATCH ()-[r]->() RETURN count(DISTINCT type(r)) as count"
        ))
        if result:
            stats['relationship_types'] = result[0].get('count', 0)
        
        if stats['node_count'] == 0:
            print("⚠️  WARNING: No data loaded in MemgraphDB!")
            print("   Please run the extraction pipeline first:")
            print("   python3 v1.0-graph-ea-scripts.py")
            return False, stats
        
        print(f"✅ Graph loaded: {stats['node_count']:,} nodes, {stats['relationship_count']:,} relationships")
        print(f"   Node types: {stats['node_types']}, Relationship types: {stats['relationship_types']}")
        return True, stats
        
    except Exception as e:
        print(f"⚠️  Could not verify graph: {e}")
        return False, stats

def main():
    print("\n" + "="*70)
    print("EARNINGS ANNOUNCEMENT TRANSCRIPTS KNOWLEDGE GRAPH - COMPLETE SETUP")
    print("Enhanced v2.0")
    print("="*70)
    
    # Connect to verify database is running
    try:
        print("\nConnecting to MemgraphDB at localhost:7687...")
        db = connect_with_retry(host='localhost', port=7687, max_retries=3)
        print("✅ Connected successfully")
    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")
        print("\nMake sure MemgraphDB is running:")
        print("  cd graph-datasources/memgraph_docker")
        print("  docker-compose up -d")
        print("\nOr check the connection settings in this script.\n")
        sys.exit(1)
    
    # Verify data is loaded
    has_data, stats = verify_graph_loaded(db)
    if not has_data:
        print("\n⚠️  Setup will continue, but some optimizations may not be effective")
        print("   Load data first for best results\n")
    else:
        print()
    
    # Run setup scripts
    success_count = 0
    total_scripts = 2
    script_results = {}
    
    # 1. Create indexes
    print("\n" + "="*70)
    print("STEP 1: Creating Indexes")
    print("="*70)
    if run_script("create_indexes.py"):
        success_count += 1
        script_results['indexes'] = True
    else:
        script_results['indexes'] = False
    
    # 2. Materialize views
    print("\n" + "="*70)
    print("STEP 2: Creating Views")
    print("="*70)
    if run_script("create_views.py"):
        success_count += 1
        script_results['views'] = True
    else:
        script_results['views'] = False
    
    # Summary
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    print(f"\n✅ {success_count}/{total_scripts} setup scripts completed successfully")
    
    if has_data:
        print(f"\n📊 Graph Statistics:")
        print(f"   Nodes: {stats['node_count']:,}")
        print(f"   Relationships: {stats['relationship_count']:,}")
        print(f"   Node Types: {stats['node_types']}")
        print(f"   Relationship Types: {stats['relationship_types']}")
    
    if success_count == total_scripts:
        print("\n🚀 Your knowledge graph is now optimized!")
        print("\nNext steps:")
        print("  1. Query the graph using GQLAlchemy or Memgraph Lab")
        print("  2. Use the materialized views for fast analytics")
        print("  3. Run additional queries as needed")
    else:
        print("\n⚠️  Some setup steps failed - review errors above")
        if not script_results.get('indexes'):
            print("   - Index creation had issues")
        if not script_results.get('views'):
            print("   - View creation had issues")
    
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)






