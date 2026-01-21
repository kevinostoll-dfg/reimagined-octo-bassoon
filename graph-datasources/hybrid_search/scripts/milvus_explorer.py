#!/usr/bin/env python3
"""
Milvus Collection Explorer

This script explores Milvus collections, their schemas, aliases, and data.
Useful for understanding collection structure before configuring hybrid search.
"""

import sys
import os
import json
import logging
from typing import List, Dict, Optional, Any

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from pymilvus import MilvusClient, connections, utility
except ImportError:
    logger.error("pymilvus not installed. Install with: pip install pymilvus")
    sys.exit(1)


def get_milvus_client() -> Optional[MilvusClient]:
    """Get Milvus client connection."""
    try:
        milvus_uri = os.getenv("MILVUS_URI", config.MILVUS_URI)
        milvus_token = os.getenv("MILVUS_TOKEN", config.MILVUS_TOKEN)
        
        logger.info(f"Connecting to Milvus at {milvus_uri}")
        client = MilvusClient(
            uri=milvus_uri,
            token=milvus_token
        )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return None


def list_all_collections(client: MilvusClient) -> List[str]:
    """List all collections in Milvus."""
    try:
        collections = client.list_collections()
        return collections
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return []


def get_collection_info(client: MilvusClient, collection_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a collection."""
    try:
        # Check if collection exists
        if not client.has_collection(collection_name=collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return None
        
        # Get collection description
        collection_info = client.describe_collection(collection_name=collection_name)
        
        # Get collection stats
        try:
            stats = client.get_collection_stats(collection_name=collection_name)
        except Exception as e:
            logger.warning(f"Could not get collection stats: {e}")
            stats = None
        
        # Get aliases (if any)
        try:
            # Try to get aliases using utility
            aliases = utility.list_aliases(collection_name)
        except Exception as e:
            logger.debug(f"Could not get aliases: {e}")
            aliases = []
        
        return {
            "name": collection_name,
            "description": collection_info,
            "stats": stats,
            "aliases": aliases
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_collection_schema(client: MilvusClient, collection_name: str) -> Optional[Dict[str, Any]]:
    """Get collection schema with field details."""
    try:
        collection_info = client.describe_collection(collection_name=collection_name)
        
        schema_info = {
            "collection_name": collection_name,
            "fields": []
        }
        
        # Extract fields from collection info
        if hasattr(collection_info, 'fields'):
            fields = collection_info.fields
        elif isinstance(collection_info, dict) and 'fields' in collection_info:
            fields = collection_info['fields']
        else:
            logger.warning("Could not extract fields from collection info")
            return schema_info
        
        for field in fields:
            field_info = {
                "name": field.get('name', 'unknown'),
                "type": field.get('type', 'unknown'),
                "description": field.get('description', ''),
                "is_primary": field.get('is_primary', False),
                "auto_id": field.get('auto_id', False),
            }
            
            # Add vector-specific info
            if 'params' in field:
                params = field['params']
                if 'dim' in params:
                    field_info["dimension"] = params['dim']
                if 'metric_type' in params:
                    field_info["metric_type"] = params['metric_type']
            
            # Check if it's a sparse vector (BM25)
            if field.get('type') == 'SPARSE_FLOAT_VECTOR' or 'sparse' in field.get('name', '').lower():
                field_info["is_sparse"] = True
            elif field.get('type') == 'FLOAT_VECTOR' or 'dense' in field.get('name', '').lower():
                field_info["is_dense"] = True
            
            schema_info["fields"].append(field_info)
        
        return schema_info
    except Exception as e:
        logger.error(f"Failed to get collection schema: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def sample_collection_data(client: MilvusClient, collection_name: str, limit: int = 5) -> Optional[List[Dict]]:
    """Get sample data from collection."""
    try:
        # Get collection schema first to know what fields exist
        schema = get_collection_schema(client, collection_name)
        if not schema or not schema.get("fields"):
            logger.warning("Could not get schema, skipping data sampling")
            return None
        
        # Get non-vector field names
        field_names = []
        for field in schema["fields"]:
            field_name = field.get("name")
            if field_name and field_name not in ["id"]:
                # Skip vector fields for sampling (they're too large)
                if not field.get("is_dense") and not field.get("is_sparse"):
                    field_names.append(field_name)
        
        if not field_names:
            logger.info("No non-vector fields found for sampling")
            return None
        
        # Query sample data
        try:
            results = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=field_names,
                limit=limit
            )
            return results
        except Exception as e:
            logger.warning(f"Could not query sample data: {e}")
            return None
    except Exception as e:
        logger.error(f"Failed to sample collection data: {e}")
        return None


def explore_collection(client: MilvusClient, collection_name: str, include_samples: bool = True):
    """Explore a single collection in detail."""
    print(f"\n{'='*80}")
    print(f"EXPLORING COLLECTION: {collection_name}")
    print(f"{'='*80}\n")
    
    # Get collection info
    info = get_collection_info(client, collection_name)
    if not info:
        print(f"❌ Collection '{collection_name}' not found or error occurred")
        return
    
    # Display basic info
    print("📋 COLLECTION INFORMATION:")
    print("-" * 80)
    print(f"Name: {info['name']}")
    
    if info.get('aliases'):
        print(f"Aliases: {', '.join(info['aliases'])}")
    else:
        print("Aliases: None")
    
    if info.get('stats'):
        stats = info['stats']
        if isinstance(stats, dict):
            row_count = stats.get('row_count', 'Unknown')
            print(f"Row Count: {row_count}")
    
    # Get and display schema
    print(f"\n📊 COLLECTION SCHEMA:")
    print("-" * 80)
    schema = get_collection_schema(client, collection_name)
    if schema:
        for field in schema.get("fields", []):
            field_name = field.get("name", "unknown")
            field_type = field.get("type", "unknown")
            is_primary = "✓ PRIMARY" if field.get("is_primary") else ""
            is_dense = " [DENSE VECTOR]" if field.get("is_dense") else ""
            is_sparse = " [SPARSE VECTOR/BM25]" if field.get("is_sparse") else ""
            
            print(f"  • {field_name}: {field_type}{is_primary}{is_dense}{is_sparse}")
            
            if field.get("dimension"):
                print(f"    Dimension: {field['dimension']}")
            if field.get("metric_type"):
                print(f"    Metric: {field['metric_type']}")
    else:
        print("  Could not retrieve schema")
    
    # Sample data
    if include_samples:
        print(f"\n📝 SAMPLE DATA (first 5 rows):")
        print("-" * 80)
        samples = sample_collection_data(client, collection_name, limit=5)
        if samples:
            for i, sample in enumerate(samples, 1):
                print(f"\n  Row {i}:")
                for key, value in sample.items():
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"    {key}: {value}")
        else:
            print("  No sample data available or query failed")
    
    print(f"\n{'='*80}\n")


def explore_all_collections(client: MilvusClient):
    """Explore all collections."""
    print("\n" + "="*80)
    print("MILVUS COLLECTION EXPLORER")
    print("="*80)
    
    collections = list_all_collections(client)
    
    if not collections:
        print("\n❌ No collections found in Milvus")
        return
    
    print(f"\n📚 Found {len(collections)} collection(s):")
    for i, coll_name in enumerate(collections, 1):
        print(f"  {i}. {coll_name}")
    
    print("\n" + "="*80)
    print("DETAILED EXPLORATION")
    print("="*80)
    
    for collection_name in collections:
        explore_collection(client, collection_name, include_samples=True)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore Milvus collections")
    parser.add_argument(
        "--collection",
        type=str,
        help="Specific collection name to explore (if not provided, explores all)"
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip sampling data from collections"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export collection info to JSON file"
    )
    
    args = parser.parse_args()
    
    # Connect to Milvus
    client = get_milvus_client()
    if not client:
        print("❌ Failed to connect to Milvus. Check your MILVUS_URI and MILVUS_TOKEN.")
        sys.exit(1)
    
    try:
        if args.collection:
            # Explore specific collection
            explore_collection(client, args.collection, include_samples=not args.no_samples)
            
            # Export if requested
            if args.export:
                schema = get_collection_schema(client, args.collection)
                info = get_collection_info(client, args.collection)
                export_data = {
                    "collection_name": args.collection,
                    "schema": schema,
                    "info": info
                }
                with open(args.export, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                print(f"\n✅ Exported collection info to {args.export}")
        else:
            # Explore all collections
            explore_all_collections(client)
    finally:
        client.close()
        print("\n✅ Exploration complete")


if __name__ == "__main__":
    main()

