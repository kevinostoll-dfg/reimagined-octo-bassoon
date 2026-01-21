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
import subprocess
from pathlib import Path

def run_script(script_name: str):
    """Run a setup script"""
    script_path = Path(__file__).parent / script_name
    
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
        print(f"❌ Script not found: {script_path}")
        return False

def verify_graph_loaded(db: Memgraph) -> bool:
    """Verify that graph data is loaded"""
    try:
        result = list(db.execute_and_fetch("MATCH (n) RETURN count(n) as count"))
        count = result[0]['count']
        
        if count == 0:
            print("⚠️  WARNING: No data loaded in MemgraphDB!")
            print("   Please run the extraction pipeline first:")
            print("   python3 v3.0-prototype.py")
            return False
        
        print(f"✅ Graph loaded: {count:,} nodes")
        return True
        
    except Exception as e:
        print(f"❌ Could not verify graph: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("SEC 10-K KNOWLEDGE GRAPH - COMPLETE SETUP")
    print("="*70)
    
    # Connect to verify database is running
    try:
        print("\nConnecting to MemgraphDB at localhost:7688...")
        db = Memgraph(host='localhost', port=7688)
        db.execute("MATCH (n) RETURN count(n) LIMIT 1;")
        print("✅ Connected successfully")
    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")
        print("\nMake sure MemgraphDB is running:")
        print("  docker-compose up -d\n")
        sys.exit(1)
    
    # Verify data is loaded
    if not verify_graph_loaded(db):
        print("\n⚠️  Setup will continue, but some optimizations may not be effective")
        print("   Load data first for best results\n")
    
    # Run setup scripts
    success_count = 0
    total_scripts = 2
    
    # 1. Create indexes
    if run_script("create_indexes.py"):
        success_count += 1
    
    # 2. Materialize views
    if run_script("create_views.py"):
        success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    print(f"\n✅ {success_count}/{total_scripts} setup scripts completed successfully")
    
    if success_count == total_scripts:
        print("\n🚀 Your knowledge graph is now optimized!")
        print("\nNext steps:")
        print("  1. Run queries: python3 main.py")
        print("  2. Module-specific: python3 main.py -m risk")
        print("  3. With timing: python3 main.py -t")
        print("  4. Graph algorithms: python3 main.py -m algorithms")
    else:
        print("\n⚠️  Some setup steps failed - review errors above")
    
    print()

if __name__ == "__main__":
    main()






