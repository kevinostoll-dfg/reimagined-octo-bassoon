#!/usr/bin/env python3
"""
DragonFly Config Search

This script searches and explores configuration stored in DragonFly (Redis-compatible).
Useful for finding Milvus configs, embedding configs, and other stored configurations.
"""

import sys
import os
import json
import logging
import redis
from typing import List, Dict, Optional, Any

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dragonfly_client():
    """Get a DragonFly (Redis-compatible) client."""
    return redis.Redis(
        host=config.DRAGONFLY_HOST,
        port=config.DRAGONFLY_PORT,
        password=config.DRAGONFLY_PASSWORD,
        db=config.DRAGONFLY_DB,
        ssl=config.DRAGONFLY_SSL,
        decode_responses=True,
        socket_timeout=10
    )


def search_keys(client: redis.Redis, pattern: str = "*") -> List[str]:
    """Search for keys matching a pattern."""
    try:
        keys = client.keys(pattern)
        return sorted(keys)
    except Exception as e:
        logger.error(f"Failed to search keys: {e}")
        return []


def get_config(client: redis.Redis, key: str) -> Optional[Dict[str, Any]]:
    """Get and parse config from DragonFly."""
    try:
        data = client.get(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # Return as string if not JSON
                return {"_raw": data}
        return None
    except Exception as e:
        logger.error(f"Failed to get config for key '{key}': {e}")
        return None


def display_config(key: str, config_data: Dict[str, Any], show_full: bool = False):
    """Display configuration in a readable format."""
    print(f"\n{'='*80}")
    print(f"CONFIG KEY: {key}")
    print(f"{'='*80}\n")
    
    if config_data is None:
        print("❌ Config not found or could not be retrieved")
        return
    
    # If it's raw string data
    if "_raw" in config_data and len(config_data) == 1:
        print("📄 RAW DATA (not JSON):")
        print("-" * 80)
        raw_data = config_data["_raw"]
        if len(raw_data) > 500 and not show_full:
            print(raw_data[:500] + "...")
            print(f"\n... (truncated, use --full to see all)")
        else:
            print(raw_data)
        return
    
    # Display as formatted JSON
    print("📋 CONFIGURATION:")
    print("-" * 80)
    
    if show_full:
        print(json.dumps(config_data, indent=2, default=str))
    else:
        # Display summary
        display_config_summary(config_data)


def display_config_summary(config_data: Dict[str, Any], indent: int = 0):
    """Display a summary of config data."""
    prefix = "  " * indent
    
    for key, value in config_data.items():
        if isinstance(value, dict):
            print(f"{prefix}• {key}:")
            display_config_summary(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}• {key}: [{len(value)} items]")
            if len(value) > 0 and isinstance(value[0], dict):
                # Show first item as example
                print(f"{prefix}  Example:")
                display_config_summary(value[0], indent + 2)
        elif isinstance(value, str) and len(value) > 100:
            print(f"{prefix}• {key}: {value[:100]}...")
        else:
            print(f"{prefix}• {key}: {value}")


def search_milvus_configs(client: redis.Redis) -> List[str]:
    """Search for Milvus-related config keys."""
    patterns = [
        "*milvus*",
        "*blacksmith*embeddings*",
        "*hybrid*config*",
        "*collection*",
    ]
    
    all_keys = set()
    for pattern in patterns:
        keys = search_keys(client, pattern)
        all_keys.update(keys)
    
    return sorted(all_keys)


def search_embedding_configs(client: redis.Redis) -> List[str]:
    """Search for embedding-related config keys."""
    patterns = [
        "*embedding*",
        "*qwen*",
        "*llama*",
    ]
    
    all_keys = set()
    for pattern in patterns:
        keys = search_keys(client, pattern)
        all_keys.update(keys)
    
    return sorted(all_keys)


def list_all_configs(client: redis.Redis, pattern: str = "*") -> List[str]:
    """List all config keys matching pattern."""
    return search_keys(client, pattern)


def export_config(client: redis.Redis, key: str, output_file: str):
    """Export config to JSON file."""
    config_data = get_config(client, key)
    if config_data:
        with open(output_file, 'w') as f:
            json.dump({key: config_data}, f, indent=2, default=str)
        print(f"✅ Exported config to {output_file}")
    else:
        print(f"❌ Could not retrieve config for key '{key}'")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search and explore DragonFly configurations")
    parser.add_argument(
        "--search",
        type=str,
        help="Search for keys matching pattern (e.g., '*milvus*', '*blacksmith*')"
    )
    parser.add_argument(
        "--key",
        type=str,
        help="Get specific config key"
    )
    parser.add_argument(
        "--milvus",
        action="store_true",
        help="Search for Milvus-related configs"
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Search for embedding-related configs"
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all config keys (use with caution, may be slow)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full config (not just summary)"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export config to JSON file (requires --key)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Pattern for key search (default: '*')"
    )
    
    args = parser.parse_args()
    
    # Connect to DragonFly
    try:
        client = get_dragonfly_client()
        client.ping()
        print("✅ Connected to DragonFly")
    except Exception as e:
        print(f"❌ Failed to connect to DragonFly: {e}")
        sys.exit(1)
    
    try:
        if args.key:
            # Get specific config
            config_data = get_config(client, args.key)
            display_config(args.key, config_data, show_full=args.full)
            
            if args.export:
                export_config(client, args.key, args.export)
        
        elif args.milvus:
            # Search Milvus configs
            print("\n" + "="*80)
            print("SEARCHING FOR MILVUS CONFIGS")
            print("="*80)
            keys = search_milvus_configs(client)
            
            if not keys:
                print("\n❌ No Milvus-related configs found")
            else:
                print(f"\n📚 Found {len(keys)} Milvus-related config(s):\n")
                for i, key in enumerate(keys, 1):
                    print(f"  {i}. {key}")
                
                print("\n" + "="*80)
                print("CONFIG DETAILS")
                print("="*80)
                
                for key in keys:
                    config_data = get_config(client, key)
                    display_config(key, config_data, show_full=args.full)
        
        elif args.embeddings:
            # Search embedding configs
            print("\n" + "="*80)
            print("SEARCHING FOR EMBEDDING CONFIGS")
            print("="*80)
            keys = search_embedding_configs(client)
            
            if not keys:
                print("\n❌ No embedding-related configs found")
            else:
                print(f"\n📚 Found {len(keys)} embedding-related config(s):\n")
                for i, key in enumerate(keys, 1):
                    print(f"  {i}. {key}")
                
                print("\n" + "="*80)
                print("CONFIG DETAILS")
                print("="*80)
                
                for key in keys:
                    config_data = get_config(client, key)
                    display_config(key, config_data, show_full=args.full)
        
        elif args.search:
            # Search with custom pattern
            print(f"\n🔍 Searching for keys matching: {args.search}")
            keys = search_keys(client, args.search)
            
            if not keys:
                print(f"\n❌ No keys found matching pattern '{args.search}'")
            else:
                print(f"\n📚 Found {len(keys)} key(s):\n")
                for i, key in enumerate(keys, 1):
                    print(f"  {i}. {key}")
                
                if len(keys) == 1:
                    # Auto-display if only one result
                    print("\n" + "="*80)
                    config_data = get_config(client, keys[0])
                    display_config(keys[0], config_data, show_full=args.full)
        
        elif args.list_all:
            # List all keys (with pattern)
            print(f"\n📋 Listing all keys matching pattern: {args.pattern}")
            keys = list_all_configs(client, args.pattern)
            
            if not keys:
                print(f"\n❌ No keys found matching pattern '{args.pattern}'")
            else:
                print(f"\n📚 Found {len(keys)} key(s):\n")
                for i, key in enumerate(keys, 1):
                    print(f"  {i}. {key}")
        
        else:
            # Default: show common configs
            print("\n" + "="*80)
            print("DRAGONFLY CONFIG SEARCH")
            print("="*80)
            print("\nCommon config keys:")
            
            common_keys = [
                "llama:milvus:blacksmith_embeddings:hybrid_config",
                "llama:milvus:longterm_memory:hybrid_config",
                "memgraph:cipher:full",
            ]
            
            for key in common_keys:
                exists = client.exists(key)
                status = "✅" if exists else "❌"
                print(f"  {status} {key}")
            
            print("\nUse --milvus to search Milvus configs")
            print("Use --embeddings to search embedding configs")
            print("Use --search '*pattern*' to search custom patterns")
            print("Use --key 'key_name' to get specific config")
    
    finally:
        client.close()
        print("\n✅ Search complete")


if __name__ == "__main__":
    main()

