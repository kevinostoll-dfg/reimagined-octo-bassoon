#!/usr/bin/env python3
"""
Store Memgraph CIPHER schemas from actual Memgraph database in DragonFly.

This script:
1. Queries Memgraph to get the full schema (all services)
2. Tags each element with which service(s) it belongs to
3. Stores both full and filtered service-specific schemas in DragonFly

Keys stored:
- memgraph:cipher:full - Complete schema from all services
- memgraph:cipher:ea - Filtered to earnings announcement elements only
- memgraph:cipher:fomc - Filtered to FOMC transcript elements only
- memgraph:cipher:sec-10k - Filtered to SEC 10-K filing elements only
- memgraph:cipher:sec-f4 - Filtered to SEC Form 4 filing elements only
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Optional
from dotenv import load_dotenv

try:
    import redis
except ImportError:
    print("❌ redis package not installed. Install with: pip install redis")
    exit(1)

try:
    from gqlalchemy import Memgraph
except ImportError:
    print("❌ GQLAlchemy package not installed. Install with: pip install GQLAlchemy")
    exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Service definitions - which node labels and relationship types belong to which services
SERVICE_DEFINITIONS = {
    "ea": {
        "name": "Earnings Announcement Transcripts",
        "node_labels": {
            "PERSON", "ORG", "PRODUCT", "CONCEPT", "METRIC", "METRIC_DEFINITION",
            "STATEMENT", "ROLE", "TECHNOLOGY", "DATE", "TIME", "MONEY", "PERCENT",
            "CARDINAL", "QUANTITY"
        },
        "relationship_types": {
            "SAID", "HAS_ROLE", "SVO_TRIPLE", "TEMPORAL_CONTEXT", "QUANTITY_OF",
            "QUANTITY_IN_ACTION", "CO_MENTIONED", "SAME_AS", "CAUSES", "DRIVES",
            "BOOSTS", "HURTS", "MITIGATES", "RESULTS_IN", "LEADS_TO", "TARGETS",
            "OUTPERFORMED", "POSITIVE_ABOUT", "NEGATIVE_ABOUT", "EVENT_INVOLVES"
        }
    },
    "fomc": {
        "name": "FOMC Transcripts",
        "node_labels": {
            "PERSON", "ORG", "PRODUCT", "CONCEPT", "METRIC", "METRIC_DEFINITION",
            "SECTION", "TECHNOLOGY", "DATE", "TIME", "MONEY", "PERCENT",
            "CARDINAL", "QUANTITY"
        },
        "relationship_types": {
            "SVO_TRIPLE", "TEMPORAL_CONTEXT", "QUANTITY_OF", "QUANTITY_IN_ACTION",
            "CO_MENTIONED", "SAME_AS", "CAUSES", "DRIVES", "BOOSTS", "HURTS",
            "MITIGATES", "RESULTS_IN", "LEADS_TO", "TARGETS", "OUTPERFORMED",
            "POSITIVE_ABOUT", "NEGATIVE_ABOUT", "EVENT_INVOLVES"
        }
    },
    "sec-10k": {
        "name": "SEC 10-K Filings",
        "node_labels": {
            "PERSON", "ORG", "PRODUCT", "CONCEPT", "METRIC", "METRIC_DEFINITION",
            "SECTION", "TECHNOLOGY", "DATE", "TIME", "MONEY", "PERCENT",
            "CARDINAL", "QUANTITY"
        },
        "relationship_types": {
            "SVO_TRIPLE", "TEMPORAL_CONTEXT", "QUANTITY_OF", "QUANTITY_IN_ACTION",
            "CO_MENTIONED", "SAME_AS", "CAUSES", "DRIVES", "BOOSTS", "HURTS",
            "MITIGATES", "RESULTS_IN", "LEADS_TO", "TARGETS", "OUTPERFORMED",
            "POSITIVE_ABOUT", "NEGATIVE_ABOUT", "EVENT_INVOLVES"
        }
    },
    "sec-f4": {
        "name": "SEC Form 4 Filings",
        "node_labels": {
            "Insider", "Company", "Transaction", "Security"
        },
        "relationship_types": {
            "FILED", "INVOLVES", "TRADES", "HOLDS_POSITION"
        }
    }
}


def connect_to_memgraph(host: str = None, port: int = None) -> Optional[Memgraph]:
    """
    Connect to Memgraph database.
    
    Uses environment variables:
    - MEMGRAPH_HOST (default: localhost)
    - MEMGRAPH_PORT (default: 7687)
    - MEMGRAPH_USER (optional)
    - MEMGRAPH_PASSWORD (optional)
    """
    host = host or os.getenv("MEMGRAPH_HOST", "localhost")
    port = port or int(os.getenv("MEMGRAPH_PORT", "7687"))
    username = os.getenv("MEMGRAPH_USER", None)
    password = os.getenv("MEMGRAPH_PASSWORD", None)
    
    try:
        logger.info(f"Connecting to Memgraph at {host}:{port}...")
        if username:
            logger.info(f"   Using authentication (user: {username})")
        
        # GQLAlchemy Memgraph constructor supports username/password if provided
        if username and password:
            db = Memgraph(host=host, port=port, username=username, password=password)
        else:
            db = Memgraph(host=host, port=port)
        
        # Test connection
        db.execute("MATCH (n) RETURN count(n) LIMIT 1;")
        logger.info("✅ Successfully connected to Memgraph")
        return db
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Memgraph: {e}")
        logger.error(f"   Host: {host}, Port: {port}, User: {username or 'None'}")
        return None


def query_node_labels(db: Memgraph) -> List[str]:
    """Query Memgraph for all node labels."""
    try:
        result = db.execute_and_fetch("""
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN DISTINCT label
            ORDER BY label;
        """)
        return [row["label"] for row in result]
    except Exception as e:
        logger.error(f"❌ Could not query node labels: {e}")
        return []


def query_relationship_types(db: Memgraph) -> List[str]:
    """Query Memgraph for all relationship types."""
    try:
        result = db.execute_and_fetch("""
            MATCH ()-[r]->()
            RETURN DISTINCT type(r) AS relationshipType
            ORDER BY relationshipType;
        """)
        return [row["relationshipType"] for row in result]
    except Exception as e:
        logger.error(f"❌ Could not query relationship types: {e}")
        return []


def query_property_keys(db: Memgraph) -> List[str]:
    """Query Memgraph for all property keys used across nodes and relationships."""
    try:
        node_keys = set()
        rel_keys = set()
        
        for row in db.execute_and_fetch("""
            MATCH (n)
            UNWIND keys(n) AS propertyKey
            RETURN DISTINCT propertyKey;
        """):
            if row.get("propertyKey") is not None:
                node_keys.add(row["propertyKey"])
        
        for row in db.execute_and_fetch("""
            MATCH ()-[r]->()
            UNWIND keys(r) AS propertyKey
            RETURN DISTINCT propertyKey;
        """):
            if row.get("propertyKey") is not None:
                rel_keys.add(row["propertyKey"])
        
        return sorted(node_keys.union(rel_keys))
    except Exception as e:
        logger.error(f"❌ Could not query property keys: {e}")
        return []


def get_node_properties(db: Memgraph, label: str) -> Dict[str, str]:
    """
    Get properties for a specific node label by sampling nodes.
    Returns dict of property_name -> property_type_description
    """
    try:
        result = db.execute_and_fetch(f"""
            MATCH (n:{label})
            RETURN keys(n) as props
            LIMIT 10
        """)
        
        all_props = set()
        for row in result:
            all_props.update(row.get("props", []))
        
        # Try to infer types from sample values
        props_info = {}
        for prop in all_props:
            # Sample a value to infer type
            sample_result = db.execute_and_fetch(f"""
                MATCH (n:{label})
                WHERE n.{prop} IS NOT NULL
                RETURN n.{prop} as value
                LIMIT 1
            """)
            for row in sample_result:
                value = row.get("value")
                if isinstance(value, bool):
                    props_info[prop] = "boolean"
                elif isinstance(value, int):
                    props_info[prop] = "integer"
                elif isinstance(value, float):
                    props_info[prop] = "float"
                elif isinstance(value, list):
                    props_info[prop] = "list"
                else:
                    props_info[prop] = "string"
                break
            else:
                props_info[prop] = "unknown"
        
        return props_info
    except Exception as e:
        logger.warning(f"⚠️  Could not get properties for {label}: {e}")
        return {}


def get_relationship_properties(db: Memgraph, rel_type: str) -> Dict[str, str]:
    """
    Get properties for a specific relationship type by sampling relationships.
    Returns dict of property_name -> property_type_description
    """
    try:
        result = db.execute_and_fetch(f"""
            MATCH ()-[r:{rel_type}]->()
            RETURN keys(r) as props
            LIMIT 10
        """)
        
        all_props = set()
        for row in result:
            all_props.update(row.get("props", []))
        
        # Try to infer types from sample values
        props_info = {}
        for prop in all_props:
            # Sample a value to infer type
            sample_result = db.execute_and_fetch(f"""
                MATCH ()-[r:{rel_type}]->()
                WHERE r.{prop} IS NOT NULL
                RETURN r.{prop} as value
                LIMIT 1
            """)
            for row in sample_result:
                value = row.get("value")
                if isinstance(value, bool):
                    props_info[prop] = "boolean"
                elif isinstance(value, int):
                    props_info[prop] = "integer"
                elif isinstance(value, float):
                    props_info[prop] = "float"
                elif isinstance(value, list):
                    props_info[prop] = "list"
                else:
                    props_info[prop] = "string"
                break
            else:
                props_info[prop] = "unknown"
        
        return props_info
    except Exception as e:
        logger.warning(f"⚠️  Could not get properties for {rel_type}: {e}")
        return {}


def query_indexes(db: Memgraph) -> List[Dict[str, str]]:
    """Query Memgraph for indexes."""
    try:
        # Memgraph uses SHOW INDEX INFO
        result = db.execute_and_fetch("SHOW INDEX INFO")
        indexes = []
        for row in result:
            indexes.append({
                "label": row.get("label", ""),
                "property": row.get("property", ""),
                "type": row.get("type", "unknown")
            })
        return indexes
    except Exception as e:
        logger.warning(f"⚠️  Could not query indexes: {e}")
        return []


def get_services_for_label(label: str) -> List[str]:
    """Determine which service(s) a node label belongs to."""
    services = []
    for service_id, service_def in SERVICE_DEFINITIONS.items():
        if label in service_def["node_labels"]:
            services.append(service_id)
    return services if services else ["unknown"]


def get_services_for_relationship(rel_type: str) -> List[str]:
    """Determine which service(s) a relationship type belongs to."""
    services = []
    for service_id, service_def in SERVICE_DEFINITIONS.items():
        if rel_type in service_def["relationship_types"]:
            services.append(service_id)
    return services if services else ["unknown"]


def build_full_schema(db: Memgraph) -> Dict[str, Any]:
    """
    Query Memgraph and build the full schema with service tags.
    """
    logger.info("Querying Memgraph for schema information...")
    
    # Query all schema elements
    node_labels = query_node_labels(db)
    relationship_types = query_relationship_types(db)
    property_keys = query_property_keys(db)
    indexes = query_indexes(db)
    
    logger.info(f"   Found {len(node_labels)} node labels")
    logger.info(f"   Found {len(relationship_types)} relationship types")
    logger.info(f"   Found {len(property_keys)} property keys")
    logger.info(f"   Found {len(indexes)} indexes")
    
    # Build node types with service tags
    node_types = []
    for label in node_labels:
        services = get_services_for_label(label)
        properties = get_node_properties(db, label)
        
        node_types.append({
            "label": label,
            "services": services,
            "properties": {k: f"{v} (inferred)" for k, v in properties.items()},
            "description": f"Node label: {label}"
        })
    
    # Build relationship types with service tags
    relationship_types_list = []
    for rel_type in relationship_types:
        services = get_services_for_relationship(rel_type)
        properties = get_relationship_properties(db, rel_type)
        
        # Try to infer source/target labels from actual relationships
        source_labels = set()
        target_labels = set()
        try:
            result = db.execute_and_fetch(f"""
                MATCH (a)-[r:{rel_type}]->(b)
                RETURN DISTINCT labels(a)[0] as source_label, labels(b)[0] as target_label
                LIMIT 20
            """)
            for row in result:
                if row.get("source_label"):
                    source_labels.add(row["source_label"])
                if row.get("target_label"):
                    target_labels.add(row["target_label"])
        except:
            pass
        
        relationship_types_list.append({
            "type": rel_type,
            "services": services,
            "source_labels": sorted(list(source_labels)) if source_labels else [],
            "target_labels": sorted(list(target_labels)) if target_labels else [],
            "properties": {k: f"{v} (inferred)" for k, v in properties.items()},
            "description": f"Relationship type: {rel_type}"
        })
    
    # Build schema
    schema = {
        "version": "1.0",
        "generated_from": "memgraph_introspection",
        "description": "Complete Memgraph CIPHER schema from all services",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "node_types": node_types,
        "relationship_types": relationship_types_list,
        "indexes": indexes,
        "property_keys": property_keys,
        "services": {
            service_id: service_def["name"]
            for service_id, service_def in SERVICE_DEFINITIONS.items()
        }
    }
    
    return schema


def filter_schema_by_service(full_schema: Dict[str, Any], service_id: str) -> Dict[str, Any]:
    """
    Filter the full schema to only include elements for a specific service.
    """
    if service_id not in SERVICE_DEFINITIONS:
        logger.warning(f"⚠️  Unknown service: {service_id}")
        return {}
    
    service_def = SERVICE_DEFINITIONS[service_id]
    
    # Filter node types
    filtered_nodes = [
        node for node in full_schema["node_types"]
        if service_id in node.get("services", [])
    ]
    
    # Filter relationship types
    filtered_rels = [
        rel for rel in full_schema["relationship_types"]
        if service_id in rel.get("services", [])
    ]
    
    # Filter indexes (only for nodes in this service)
    service_labels = {node["label"] for node in filtered_nodes}
    filtered_indexes = [
        idx for idx in full_schema["indexes"]
        if idx.get("label") in service_labels
    ]
    
    filtered_schema = {
        "version": full_schema["version"],
        "service": service_id,
        "service_name": service_def["name"],
        "description": f"Memgraph CIPHER schema filtered for {service_def['name']}",
        "generated_at": full_schema.get("generated_at", ""),
        "node_types": filtered_nodes,
        "relationship_types": filtered_rels,
        "indexes": filtered_indexes,
        "total_node_types": len(filtered_nodes),
        "total_relationship_types": len(filtered_rels),
        "total_indexes": len(filtered_indexes)
    }
    
    return filtered_schema


def connect_to_dragonfly() -> redis.Redis:
    """
    Connect to DragonFly (Redis-compatible) database.
    """
    host = os.getenv("DRAGONFLY_HOST", "localhost")
    port = int(os.getenv("DRAGONFLY_PORT", "6379"))
    password = os.getenv("DRAGONFLY_PASSWORD", None)
    db = int(os.getenv("DRAGONFLY_DB", "0"))
    ssl = os.getenv("DRAGONFLY_SSL", "false").lower() == "true"
    
    try:
        logger.info(f"Connecting to DragonFly at {host}:{port} (db={db}, ssl={ssl})")
        
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            ssl=ssl,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=30
        )
        
        # Test connection
        client.ping()
        logger.info("✅ Successfully connected to DragonFly")
        return client
        
    except redis.ConnectionError as e:
        logger.error(f"❌ Failed to connect to DragonFly: {e}")
        raise
    except redis.AuthenticationError as e:
        logger.error(f"❌ Authentication failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error connecting to DragonFly: {e}")
        raise


def store_schema_in_dragonfly(client: redis.Redis, schema: Dict[str, Any], key: str) -> bool:
    """
    Store the schema in DragonFly as JSON.
    """
    try:
        # Convert schema to JSON string
        schema_json = json.dumps(schema, indent=2)
        
        # Store in DragonFly (no TTL - persistent storage)
        client.set(key, schema_json)
        
        logger.info(f"✅ Stored schema in DragonFly at key: {key}")
        logger.info(f"   Schema size: {len(schema_json)} bytes")
        
        # Verify it was stored correctly
        stored_value = client.get(key)
        if stored_value:
            logger.info("✅ Verified schema was stored correctly")
            return True
        else:
            logger.error("❌ Schema was not stored correctly")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to store schema in DragonFly: {e}")
        return False


def main():
    """Main execution: query Memgraph, build schemas, and store in DragonFly."""
    print("="*80)
    print("MEMGRAPH CIPHER SCHEMA STORAGE (FROM MEMGRAPH INTROSPECTION)")
    print("="*80)
    print()
    
    # Connect to Memgraph
    db = connect_to_memgraph()
    if not db:
        logger.error("❌ Failed to connect to Memgraph. Exiting.")
        exit(1)
    
    # Build full schema
    logger.info("Building full schema from Memgraph...")
    full_schema = build_full_schema(db)
    logger.info(f"✅ Built full schema")
    logger.info(f"   Node types: {len(full_schema['node_types'])}")
    logger.info(f"   Relationship types: {len(full_schema['relationship_types'])}")
    logger.info(f"   Indexes: {len(full_schema['indexes'])}")
    print()
    
    # Connect to DragonFly
    try:
        dragonfly_client = connect_to_dragonfly()
    except Exception as e:
        logger.error(f"❌ Failed to connect to DragonFly. Exiting.")
        exit(1)
    
    # Store full schema
    logger.info("Storing full schema in DragonFly...")
    success = store_schema_in_dragonfly(dragonfly_client, full_schema, "memgraph:cipher:full")
    if not success:
        logger.error("❌ Failed to store full schema")
        exit(1)
    print()
    
    # Store filtered schemas for each service
    stored_keys = ["memgraph:cipher:full"]
    for service_id in SERVICE_DEFINITIONS.keys():
        logger.info(f"Filtering and storing schema for service: {service_id}")
        filtered_schema = filter_schema_by_service(full_schema, service_id)
        
        if filtered_schema:
            key = f"memgraph:cipher:{service_id}"
            success = store_schema_in_dragonfly(dragonfly_client, filtered_schema, key)
            if success:
                stored_keys.append(key)
                logger.info(f"   ✅ Stored {len(filtered_schema['node_types'])} node types, {len(filtered_schema['relationship_types'])} relationship types")
            else:
                logger.warning(f"   ⚠️  Failed to store schema for {service_id}")
        else:
            logger.warning(f"   ⚠️  No schema elements found for {service_id}")
        print()
    
    # Summary
    print("="*80)
    print("✅ SUCCESS")
    print("="*80)
    print(f"Stored {len(stored_keys)} schema(s) in DragonFly:")
    for key in stored_keys:
        print(f"  - {key}")
    print()
    print("You can retrieve them with:")
    print(f"  redis-cli GET memgraph:cipher:full")
    for service_id in SERVICE_DEFINITIONS.keys():
        print(f"  redis-cli GET memgraph:cipher:{service_id}")
    print()


if __name__ == "__main__":
    main()

