#!/usr/bin/env python3
"""
Standalone Schema Fetcher for Memgraph
Fetches and displays the graph schema from Dragonfly.
"""

import redis
import json
import logging
import sys
import os

# Add parent directory to path for config import (now in scripts/ subdirectory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import config
except ImportError:
    print("Error: Could not import config.py. Please ensure config.py exists in the parent directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_schema_from_dragonfly(key="memgraph:cipher:full"):
    """
    Connects to Dragonfly and retrieves the stored schema.
    
    Args:
        key: Redis key to fetch schema from (default: "memgraph:cipher:full")
        
    Returns:
        Dictionary containing the schema or None if not found
    """
    try:
        logger.info(f"Connecting to Dragonfly at {config.DRAGONFLY_HOST}:{config.DRAGONFLY_PORT}...")
        client = redis.Redis(
            host=config.DRAGONFLY_HOST,
            port=config.DRAGONFLY_PORT,
            password=config.DRAGONFLY_PASSWORD,
            db=config.DRAGONFLY_DB,
            ssl=config.DRAGONFLY_SSL,
            decode_responses=True,
            socket_timeout=10
        )
        
        # Test connection
        client.ping()
        logger.info("✅ Successfully connected to Dragonfly")
        
        logger.info(f"Fetching schema for key: {key}")
        data = client.get(key)
        
        if data:
            try:
                schema = json.loads(data)
                logger.info("✅ Schema retrieved and parsed successfully")
                return schema
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse schema JSON: {e}")
                return None
        else:
            logger.warning(f"⚠️  Key '{key}' not found in Dragonfly")
            return None
            
    except redis.ConnectionError as e:
        logger.error(f"❌ Connection error: {e}")
        logger.error("Please check your Dragonfly connection settings in config.py")
        return None
    except Exception as e:
        logger.error(f"❌ Error fetching from Dragonfly: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_schema_for_display(schema):
    """
    Formats the JSON schema into a human-readable display.
    
    Args:
        schema: Dictionary containing the schema
        
    Returns:
        Formatted string representation
    """
    if not schema:
        return "No schema data available."
    
    output = []
    output.append("\n" + "="*80)
    output.append("MEMGRAPH SCHEMA")
    output.append("="*80 + "\n")
    
    # Node Types
    if "node_types" in schema and schema["node_types"]:
        output.append("NODE TYPES:")
        output.append("-" * 80)
        for node in schema["node_types"]:
            label = node.get("label", "Unknown")
            props = node.get("properties", {})
            
            output.append(f"\n  Label: {label}")
            if props:
                output.append("  Properties:")
                for prop_name, prop_info in props.items():
                    prop_type = prop_info.get("type", "unknown") if isinstance(prop_info, dict) else str(type(prop_info))
                    output.append(f"    - {prop_name}: {prop_type}")
            else:
                output.append("  Properties: (none)")
        output.append("")
    
    # Relationship Types
    if "relationship_types" in schema and schema["relationship_types"]:
        output.append("RELATIONSHIP TYPES:")
        output.append("-" * 80)
        for rel in schema["relationship_types"]:
            rtype = rel.get("type", "Unknown")
            src_labels = rel.get("source_labels", [])
            tgt_labels = rel.get("target_labels", [])
            props = rel.get("properties", {})
            
            src_str = ", ".join(src_labels) if src_labels else "*"
            tgt_str = ", ".join(tgt_labels) if tgt_labels else "*"
            
            output.append(f"\n  Type: {rtype}")
            output.append(f"  Pattern: (:{src_str})-[:{rtype}]->(:{tgt_str})")
            
            if props:
                output.append("  Properties:")
                for prop_name, prop_info in props.items():
                    prop_type = prop_info.get("type", "unknown") if isinstance(prop_info, dict) else str(type(prop_info))
                    output.append(f"    - {prop_name}: {prop_type}")
            else:
                output.append("  Properties: (none)")
        output.append("")
    
    # Additional metadata
    if "metadata" in schema:
        output.append("METADATA:")
        output.append("-" * 80)
        for key, value in schema["metadata"].items():
            output.append(f"  {key}: {value}")
        output.append("")
    
    output.append("="*80)
    
    return "\n".join(output)


def save_schema_to_file(schema, filename="schema.json"):
    """
    Saves the schema to a JSON file.
    
    Args:
        schema: Dictionary containing the schema
        filename: Output filename (default: "schema.json")
    """
    try:
        with open(filename, 'w') as f:
            json.dump(schema, f, indent=2)
        logger.info(f"✅ Schema saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save schema to file: {e}")
        return False


def main():
    """Main function to fetch and display schema."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch and display Memgraph schema from Dragonfly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_schema_standalone.py
  python fetch_schema_standalone.py --key "memgraph:cipher:full"
  python fetch_schema_standalone.py --save schema_output.json
  python fetch_schema_standalone.py --key "memgraph:cipher:full" --save schema.json
        """
    )
    
    parser.add_argument(
        "--key",
        default="memgraph:cipher:full",
        help="Redis key to fetch schema from (default: memgraph:cipher:full)"
    )
    
    parser.add_argument(
        "--save",
        metavar="FILENAME",
        help="Save schema to JSON file"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress logging output (only show schema)"
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    print("\n" + "="*80)
    print("MEMGRAPH SCHEMA FETCHER")
    print("="*80)
    print(f"Fetching schema from key: {args.key}\n")
    
    # Fetch schema
    schema = get_schema_from_dragonfly(key=args.key)
    
    if schema:
        # Display schema
        formatted = format_schema_for_display(schema)
        print(formatted)
        
        # Save to file if requested
        if args.save:
            save_schema_to_file(schema, args.save)
        
        # Also save to default filename if not specified
        if not args.save:
            save_schema_to_file(schema, "schema.json")
            print(f"\n💾 Schema also saved to: schema.json")
        
        print("\n✅ Schema fetch completed successfully!")
        return 0
    else:
        print("\n❌ Failed to fetch schema. Please check:")
        print("  1. Dragonfly connection settings in config.py")
        print("  2. The schema key exists in Dragonfly")
        print("  3. Network connectivity to Dragonfly")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

