
import sys
import os

# Add parent directory to path for config import (now in scripts/ subdirectory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import redis
import json
import logging
import config

logger = logging.getLogger(__name__)

def get_schema_from_dragonfly(key="memgraph:cipher:full"):
    """
    Connects to Dragonfly and retrieves the stored schema.
    Returns the schema dict or None.
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
            socket_timeout=5
        )
        
        # client.ping() 
        
        logger.info(f"Fetching schema for key: {key}")
        data = client.get(key)
        
        if data:
            try:
                schema = json.loads(data)
                logger.info("✅ Schema retrieved and parsed.")
                return schema
            except json.JSONDecodeError:
                logger.error("❌ Failed to parse schema JSON.")
                return None
        else:
            logger.warning(f"⚠️ Key {key} not found in Dragonfly.")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error fetching from Dragonfly: {e}")
        return None

def format_schema_for_prompt(schema) -> str:
    """
    Formats the JSON schema into a concise string for LLM prompts.
    """
    if not schema:
        return "Schema information unavailable."

    summary = ["The graph contains the following structure based on the stored schema:"]
    
    # Nodes
    if "node_types" in schema:
        summary.append("\nNode Labels:")
        for node in schema["node_types"]:
            label = node.get("label", "Unknown")
            props = list(node.get("properties", {}).keys())
            # Limit props to keep prompt short
            props_str = ", ".join(props[:5])
            if len(props) > 5:
                props_str += ", ..."
            summary.append(f"- {label} (Properties: {props_str})")

    # Relationships
    if "relationship_types" in schema:
        summary.append("\nRelationship Types:")
        for rel in schema["relationship_types"]:
            rtype = rel.get("type", "Unknown")
            src = rel.get("source_labels", ["*"])
            tgt = rel.get("target_labels", ["*"])
            
            src_str = "/".join(src[:3]) if src else "*"
            tgt_str = "/".join(tgt[:3]) if tgt else "*"
            
            summary.append(f"- (:{src_str})-[:{rtype}]->(:{tgt_str})")

    return "\n".join(summary)
