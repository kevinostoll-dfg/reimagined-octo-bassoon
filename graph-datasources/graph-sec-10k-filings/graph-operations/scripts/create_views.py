#!/usr/bin/env python3
"""
Create materialized views for high-performance queries
Run nightly or after every 10-K load

Pre-computes expensive aggregations for instant access
Enhanced with performance metrics and better caching strategy
"""

from gqlalchemy import Memgraph
import sys
import os
import time
import json
from typing import Dict, List, Tuple
from pathlib import Path

# Default connection settings
DEFAULT_HOST = os.getenv("MEMGRAPH_HOST", "localhost")
DEFAULT_PORT = int(os.getenv("MEMGRAPH_PORT", "7687"))

# Cache directory for view results
CACHE_DIR = Path(__file__).parent.parent / "view_cache"
CACHE_DIR.mkdir(exist_ok=True)

MATERIALIZED_VIEWS = {
    "top_hubs": {
        "query": """
            MATCH (n)
            WHERE n.canonical_name IS NOT NULL AND n.mention_count > 10
            OPTIONAL MATCH (n)-[r]-()
            WITH n, 
                 count(DISTINCT r) as degree,
                 n.mention_count as mentions
            WHERE degree > 5
            RETURN labels(n)[0] AS type, 
                   n.canonical_name AS name, 
                   mentions, 
                   degree,
                   (degree * mentions) as influence
            ORDER BY influence DESC 
            LIMIT 100
        """,
        "description": "Top 100 entities by network influence (degree × mentions)"
    },
    
    "risk_dashboard": {
        "query": """
            MATCH (risk:RISK)
            OPTIONAL MATCH (risk)<-[:MITIGATES_RISK]-(strategy)
            OPTIONAL MATCH (risk)<-[:EXPOSES_TO_RISK]-(exposer)
            WITH risk,
                 count(DISTINCT strategy) AS mitigations,
                 count(DISTINCT exposer) AS exposures,
                 risk.mention_count AS mentions
            WHERE mentions > 1
            RETURN risk.canonical_name AS risk_name,
                   mentions,
                   exposures,
                   mitigations,
                   CASE WHEN exposures > 0 
                        THEN toFloat(mitigations)/(exposures + 1) 
                        ELSE 0.0 END AS mitigation_ratio
            ORDER BY mentions DESC 
            LIMIT 50
        """,
        "description": "Risk factors with mitigation strategies and exposure analysis"
    },
    
    "financial_hotspots": {
        "query": """
            MATCH (m:MONEY)
            WHERE m.mention_count > 5
            OPTIONAL MATCH (geo)-[r:CO_MENTIONED]-(m)
            WHERE labels(geo)[0] IN ['GEOGRAPHY', 'GPE', 'LOC']
            WITH m, 
                 collect(DISTINCT geo.canonical_name)[..3] as geographies,
                 count(DISTINCT geo) as geo_count
            RETURN m.canonical_name AS amount,
                   m.mention_count AS mentions,
                   geo_count,
                   geographies
            ORDER BY m.mention_count DESC
            LIMIT 50
        """,
        "description": "Financial amounts with geographic associations"
    },
    
    "product_ecosystem": {
        "query": """
            MATCH (p:PRODUCT)
            WHERE p.mention_count > 2
            OPTIONAL MATCH (p)-[r:CO_MENTIONED]-(o:ORG)
            WHERE r.count > 1
            WITH p, 
                 collect(DISTINCT o.canonical_name)[..5] as orgs,
                 count(DISTINCT o) as org_count
            RETURN p.canonical_name AS product,
                   p.mention_count AS mentions,
                   org_count,
                   orgs
            ORDER BY org_count DESC, p.mention_count DESC
            LIMIT 30
        """,
        "description": "Products and their associated organizations"
    },
    
    "regulatory_network": {
        "query": """
            MATCH (reg:REGULATION)
            WHERE reg.mention_count > 2
            OPTIONAL MATCH (reg)-[r:CO_MENTIONED]-(e)
            WHERE labels(e)[0] IN ['ORG', 'PERSON', 'CONCEPT'] AND r.count > 1
            WITH reg,
                 count(DISTINCT CASE WHEN labels(e)[0] = 'ORG' THEN e END) as orgs,
                 count(DISTINCT CASE WHEN labels(e)[0] = 'PERSON' THEN e END) as people,
                 count(DISTINCT CASE WHEN labels(e)[0] = 'CONCEPT' THEN e END) as concepts
            RETURN reg.canonical_name AS regulation,
                   reg.mention_count AS mentions,
                   orgs, people, concepts,
                   orgs + people + concepts AS total_impact
            ORDER BY total_impact DESC
            LIMIT 30
        """,
        "description": "Regulations and their impact on entities"
    },
    
    "competitive_landscape": {
        "query": """
            MATCH (org1:ORG)-[r:COMPETES_WITH]->(org2:ORG)
            OPTIONAL MATCH (org1)-[:CO_MENTIONED]-(m:MONEY)
            WITH org1, org2, r,
                 collect(DISTINCT m.canonical_name)[..5] as financial_mentions,
                 count(DISTINCT m) as money_count
            WHERE r.count > 1
            RETURN org1.canonical_name AS company1,
                   org2.canonical_name AS company2,
                   r.count AS co_mentions,
                   money_count,
                   financial_mentions
            ORDER BY r.count DESC
            LIMIT 50
        """,
        "description": "Competitive relationships between organizations"
    },
    
    "risk_exposure_map": {
        "query": """
            MATCH (org:ORG)-[r:EXPOSES_TO_RISK]->(risk:RISK)
            OPTIONAL MATCH (risk)<-[:MITIGATES_RISK]-(mitigation)
            WITH org, risk, r,
                 collect(DISTINCT mitigation.canonical_name)[..3] as mitigations,
                 count(DISTINCT mitigation) as mitigation_count
            WHERE r.count > 1
            RETURN org.canonical_name AS organization,
                   risk.canonical_name AS risk,
                   risk.mention_count AS risk_mentions,
                   r.count AS exposure_count,
                   mitigation_count,
                   mitigations
            ORDER BY exposure_count DESC, risk.mention_count DESC
            LIMIT 100
        """,
        "description": "Organizations exposed to risks with mitigation strategies"
    },
    
    "temporal_insights": {
        "query": """
            MATCH (e)-[r:CO_MENTIONED]-(d:DATE)
            WHERE r.count > 2
            WITH e, d,
                 collect(DISTINCT labels(e)[0])[0] as entity_type,
                 e.canonical_name as entity_name
            WHERE entity_name IS NOT NULL
            RETURN entity_type,
                   entity_name,
                   d.canonical_name AS date,
                   count(*) AS temporal_connections
            ORDER BY temporal_connections DESC
            LIMIT 50
        """,
        "description": "Entities with temporal context associations"
    }
}

def save_view_cache(view_name: str, results: List[Dict], execution_time: float):
    """Save view results to cache file"""
    cache_file = CACHE_DIR / f"{view_name}.json"
    cache_data = {
        "view_name": view_name,
        "cached_at": time.time(),
        "execution_time": execution_time,
        "row_count": len(results),
        "results": results
    }
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"   ⚠️  Could not save cache: {e}")
        return False

def load_view_cache(view_name: str, max_age_seconds: int = 3600) -> Tuple[bool, List[Dict], float]:
    """Load view results from cache if available and fresh"""
    cache_file = CACHE_DIR / f"{view_name}.json"
    if not cache_file.exists():
        return False, [], 0.0
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        cached_at = cache_data.get("cached_at", 0)
        age = time.time() - cached_at
        
        if age > max_age_seconds:
            return False, [], 0.0
        
        return True, cache_data.get("results", []), cache_data.get("execution_time", 0.0)
    except Exception as e:
        return False, [], 0.0

def create_view(db: Memgraph, view_name: str, view_config: Dict, use_cache: bool = True, force_refresh: bool = False) -> Dict:
    """
    Create a materialized view with performance tracking
    Returns: dict with success, row_count, execution_time, cached status
    """
    query = view_config["query"]
    description = view_config.get("description", "")
    
    print(f"\n📊 Creating view: {view_name}")
    if description:
        print(f"   {description}")
    print(f"   Query preview: {query.strip()[:80]}...")
    
    # Check cache first (unless force refresh)
    if use_cache and not force_refresh:
        cached, cached_results, cached_time = load_view_cache(view_name)
        if cached:
            print(f"   ✅ Using cached results ({len(cached_results)} rows, {cached_time:.3f}s)")
            return {
                "success": True,
                "view_name": view_name,
                "row_count": len(cached_results),
                "execution_time": cached_time,
                "cached": True
            }
    
    # Execute query
    start_time = time.time()
    try:
        results = list(db.execute_and_fetch(query))
        execution_time = time.time() - start_time
        
        # Save to cache
        if use_cache:
            save_view_cache(view_name, results, execution_time)
        
        print(f"   ✅ Computed {len(results)} rows in {execution_time:.3f}s")
        
        # Show sample results for debugging
        if results and len(results) > 0:
            print(f"   📋 Sample result keys: {list(results[0].keys())}")
        
        return {
            "success": True,
            "view_name": view_name,
            "row_count": len(results),
            "execution_time": execution_time,
            "cached": False
        }
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"   ❌ Failed after {execution_time:.3f}s: {e}")
        return {
            "success": False,
            "view_name": view_name,
            "error": str(e),
            "execution_time": execution_time
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create materialized views for MemgraphDB 10-K Knowledge Graph")
    parser.add_argument("--host", default=DEFAULT_HOST, help="MemgraphDB host (default: localhost)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MemgraphDB Bolt port (default: 7687)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh all views (ignore cache)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--view", help="Create only a specific view (by name)")
    args = parser.parse_args()
    
    try:
        print("\n" + "="*70)
        print("MATERIALIZED VIEWS CREATION")
        print("="*70)
        print(f"\nConnecting to MemgraphDB at {args.host}:{args.port}...")
        
        db = Memgraph(host=args.host, port=args.port)
        db.execute("MATCH (n) RETURN count(n) LIMIT 1;")
        print("✅ Connected successfully")
        
        # Select views to create
        views_to_create = MATERIALIZED_VIEWS
        if args.view:
            if args.view not in MATERIALIZED_VIEWS:
                print(f"\n❌ View '{args.view}' not found. Available views:")
                for name in MATERIALIZED_VIEWS.keys():
                    print(f"   - {name}")
                sys.exit(1)
            views_to_create = {args.view: MATERIALIZED_VIEWS[args.view]}
        
        print("\nCreating materialized views (pre-computing complex queries)...")
        if args.refresh:
            print("   🔄 Force refresh enabled - ignoring cache")
        if args.no_cache:
            print("   🚫 Caching disabled")
        print("="*70)
        
        results = []
        total_time = 0.0
        
        for view_name, view_config in views_to_create.items():
            result = create_view(
                db, 
                view_name, 
                view_config,
                use_cache=not args.no_cache,
                force_refresh=args.refresh
            )
            results.append(result)
            if result.get("success"):
                total_time += result.get("execution_time", 0.0)
        
        # Summary
        success_count = sum(1 for r in results if r.get("success"))
        cached_count = sum(1 for r in results if r.get("cached", False))
        total_rows = sum(r.get("row_count", 0) for r in results if r.get("success"))
        
        print("\n" + "="*70)
        print("VIEW CREATION SUMMARY")
        print("="*70)
        print(f"✅ Successfully created: {success_count}/{len(results)} views")
        print(f"📦 From cache: {cached_count}")
        print(f"📊 Total rows computed: {total_rows:,}")
        print(f"⏱️  Total execution time: {total_time:.3f}s")
        
        if success_count < len(results):
            failed = [r["view_name"] for r in results if not r.get("success")]
            print(f"\n⚠️  Failed views: {', '.join(failed)}")
        
        print("\n💡 These queries are now cached for instant access")
        print("   Re-run this script after loading new 10-K data")
        print(f"   Cache location: {CACHE_DIR}\n")
        
        # Performance insights
        if results:
            avg_time = sum(r.get("execution_time", 0) for r in results if r.get("success")) / max(success_count, 1)
            slowest = max([r for r in results if r.get("success")], 
                         key=lambda x: x.get("execution_time", 0), 
                         default=None)
            if slowest:
                print("📈 Performance insights:")
                print(f"   Average query time: {avg_time:.3f}s")
                print(f"   Slowest view: {slowest['view_name']} ({slowest.get('execution_time', 0):.3f}s)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure MemgraphDB is running:")
        print("  cd graph-datasources/memgraph_docker")
        print("  docker-compose up -d\n")
        sys.exit(1)

if __name__ == "__main__":
    main()






