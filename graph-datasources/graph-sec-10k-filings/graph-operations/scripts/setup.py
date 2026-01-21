#!/usr/bin/env python3
"""
Complete setup script for SEC 10-K Knowledge Graph
Runs indexing and view materialization for optimal performance

Run this ONCE after loading your 10-K data:
    python3 scripts/setup.py

Or individually:
    python3 scripts/create_indexes.py
    python3 scripts/create_views.py
"""

from gqlalchemy import Memgraph
import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import Dict

# Default connection settings
DEFAULT_HOST = os.getenv("MEMGRAPH_HOST", "localhost")
DEFAULT_PORT = int(os.getenv("MEMGRAPH_PORT", "7687"))

def run_script(script_name: str, extra_args: list = None):
    """Run a setup script with optional extra arguments"""
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    if extra_args:
        print(f"Arguments: {' '.join(extra_args)}")
    print(f"{'='*70}\n")
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        cmd = ['python3', str(script_path)]
        if extra_args:
            cmd.extend(extra_args)
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Python3 not found. Please ensure Python 3 is installed.")
        return False

def verify_graph_loaded(db: Memgraph) -> Dict:
    """Verify that graph data is loaded and return statistics"""
    try:
        # Count nodes
        result = list(db.execute_and_fetch("MATCH (n) RETURN count(n) as count"))
        node_count = result[0]['count'] if result else 0
        
        # Count relationships
        result = list(db.execute_and_fetch("MATCH ()-[r]->() RETURN count(r) as count"))
        rel_count = result[0]['count'] if result else 0
        
        # Count by label (top 5)
        result = list(db.execute_and_fetch("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
            LIMIT 5;
        """))
        labels = {r['label']: r['count'] for r in result} if result else {}
        
        stats = {
            "loaded": node_count > 0,
            "node_count": node_count,
            "relationship_count": rel_count,
            "top_labels": labels
        }
        
        if node_count == 0:
            print("⚠️  WARNING: No data loaded in MemgraphDB!")
            print("   Please run the extraction pipeline first:")
            print("   python3 v3.0-prototype.py")
        else:
            print(f"✅ Graph loaded: {node_count:,} nodes, {rel_count:,} relationships")
            if labels:
                print("   Top entity types:")
                for label, count in labels.items():
                    print(f"   - {label}: {count:,}")
        
        return stats
        
    except Exception as e:
        print(f"❌ Could not verify graph: {e}")
        return {"loaded": False, "error": str(e)}

def test_connection(host: str, port: int) -> bool:
    """Test connection to MemgraphDB"""
    try:
        db = Memgraph(host=host, port=port)
        db.execute("MATCH (n) RETURN count(n) LIMIT 1;")
        return True
    except Exception as e:
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete setup for SEC 10-K Knowledge Graph")
    parser.add_argument("--host", default=DEFAULT_HOST, help="MemgraphDB host (default: localhost)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MemgraphDB Bolt port (default: 7687)")
    parser.add_argument("--skip-indexes", action="store_true", help="Skip index creation")
    parser.add_argument("--skip-views", action="store_true", help="Skip view materialization")
    parser.add_argument("--refresh-views", action="store_true", help="Force refresh views (ignore cache)")
    parser.add_argument("--stats-only", action="store_true", help="Show graph statistics only (no setup)")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SEC 10-K KNOWLEDGE GRAPH - COMPLETE SETUP")
    print("="*70)
    
    # Connect to verify database is running
    try:
        print(f"\nConnecting to MemgraphDB at {args.host}:{args.port}...")
        db = Memgraph(host=args.host, port=args.port)
        db.execute("MATCH (n) RETURN count(n) LIMIT 1;")
        print("✅ Connected successfully")
    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")
        print("\nMake sure MemgraphDB is running:")
        print("  cd graph-datasources/memgraph_docker")
        print("  docker-compose up -d\n")
        sys.exit(1)
    
    # Verify data is loaded
    stats = verify_graph_loaded(db)
    
    if args.stats_only:
        print("\n" + "="*70)
        print("STATISTICS ONLY MODE")
        print("="*70)
        sys.exit(0)
    
    if not stats.get("loaded"):
        print("\n⚠️  Setup will continue, but some optimizations may not be effective")
        print("   Load data first for best results\n")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            sys.exit(0)
    
    # Build command arguments for scripts
    script_args = ["--host", args.host, "--port", str(args.port)]
    if args.refresh_views:
        script_args.append("--refresh")
    
    # Run setup scripts
    success_count = 0
    total_scripts = 0
    scripts_run = []
    
    # 1. Create indexes
    if not args.skip_indexes:
        total_scripts += 1
        index_args = script_args + ["--stats"]
        if run_script("create_indexes.py", index_args):
            success_count += 1
            scripts_run.append("indexes")
    
    # 2. Materialize views
    if not args.skip_views:
        total_scripts += 1
        view_args = script_args.copy()
        if args.refresh_views:
            view_args.append("--refresh")
        if run_script("create_views.py", view_args):
            success_count += 1
            scripts_run.append("views")
    
    # Summary
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    print(f"\n✅ {success_count}/{total_scripts} setup scripts completed successfully")
    
    if scripts_run:
        print(f"   Completed: {', '.join(scripts_run)}")
    
    if success_count == total_scripts:
        print("\n🚀 Your knowledge graph is now optimized!")
        print("\nGraph Statistics:")
        print(f"   Nodes: {stats.get('node_count', 0):,}")
        print(f"   Relationships: {stats.get('relationship_count', 0):,}")
        
        print("\nNext steps:")
        print("  1. Query the graph using GQLAlchemy or Cypher")
        print("  2. Use Memgraph Lab at http://localhost:3000 for visualization")
        print("  3. Re-run setup after loading new 10-K data")
    else:
        print("\n⚠️  Some setup steps failed - review errors above")
    
    print()

if __name__ == "__main__":
    main()






