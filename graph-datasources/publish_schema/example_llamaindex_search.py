#!/usr/bin/env python3
"""
Example MemgraphDB Search with DragonFly Schema Lookup

This example demonstrates:
1. Retrieving Memgraph CIPHER schema from DragonFly
2. Using schema information to inform MemgraphDB queries
3. Performing intelligent graph searches based on the schema

Note: This uses GQLAlchemy (same as the graph datasource services) rather than
LlamaIndex's Memgraph integration, as it's more aligned with the existing codebase.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
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


class MemgraphSchemaLoader:
    """Load and parse Memgraph schemas from DragonFly."""
    
    def __init__(self, dragonfly_client: redis.Redis):
        self.client = dragonfly_client
    
    def load_full_schema(self) -> Optional[Dict[str, Any]]:
        """Load the complete schema from DragonFly."""
        try:
            schema_json = self.client.get("memgraph:cipher:full")
            if schema_json:
                return json.loads(schema_json)
            return None
        except Exception as e:
            logger.error(f"Failed to load full schema: {e}")
            return None
    
    def load_service_schema(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Load a service-specific schema from DragonFly."""
        try:
            key = f"memgraph:cipher:{service_id}"
            schema_json = self.client.get(key)
            if schema_json:
                return json.loads(schema_json)
            return None
        except Exception as e:
            logger.error(f"Failed to load schema for {service_id}: {e}")
            return None
    
    def get_node_labels(self, schema: Dict[str, Any], service_id: Optional[str] = None) -> List[str]:
        """Get list of node labels from schema, optionally filtered by service."""
        if not schema or "node_types" not in schema:
            return []
        
        node_types = schema["node_types"]
        if service_id:
            # Filter by service
            node_types = [
                nt for nt in node_types
                if service_id in nt.get("services", [])
            ]
        
        return [nt["label"] for nt in node_types]
    
    def get_relationship_types(self, schema: Dict[str, Any], service_id: Optional[str] = None) -> List[str]:
        """Get list of relationship types from schema, optionally filtered by service."""
        if not schema or "relationship_types" not in schema:
            return []
        
        rel_types = schema["relationship_types"]
        if service_id:
            # Filter by service
            rel_types = [
                rt for rt in rel_types
                if service_id in rt.get("services", [])
            ]
        
        return [rt["type"] for rt in rel_types]
    
    def get_schema_summary(self, schema: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the schema."""
        if not schema:
            return "No schema available"
        
        node_count = len(schema.get("node_types", []))
        rel_count = len(schema.get("relationship_types", []))
        index_count = len(schema.get("indexes", []))
        
        summary = f"Schema Summary:\n"
        summary += f"  - Node Types: {node_count}\n"
        summary += f"  - Relationship Types: {rel_count}\n"
        summary += f"  - Indexes: {index_count}\n"
        
        if "services" in schema:
            summary += f"\nServices:\n"
            for service_id, service_name in schema["services"].items():
                summary += f"  - {service_id}: {service_name}\n"
        
        return summary


class MemgraphSearchEngine:
    """MemgraphDB search engine with schema awareness using GQLAlchemy."""
    
    def __init__(
        self,
        memgraph_host: str = "localhost",
        memgraph_port: int = 7687,
        memgraph_user: Optional[str] = None,
        memgraph_password: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ):
        self.memgraph_host = memgraph_host
        self.memgraph_port = memgraph_port
        self.memgraph_user = memgraph_user
        self.memgraph_password = memgraph_password
        self.schema = schema
        
        # Initialize Memgraph connection using GQLAlchemy
        if memgraph_user and memgraph_password:
            self.db = Memgraph(
                host=memgraph_host,
                port=memgraph_port,
                username=memgraph_user,
                password=memgraph_password
            )
        else:
            self.db = Memgraph(host=memgraph_host, port=memgraph_port)
        
        logger.info(f"Initialized Memgraph connection at {memgraph_host}:{memgraph_port}")
    
    def build_schema_aware_query(self, query: str, service_id: Optional[str] = None) -> str:
        """
        Enhance a natural language query with schema awareness.
        Uses schema information to suggest node labels and relationship types.
        """
        if not self.schema:
            return query
        
        # Get relevant node labels and relationship types
        schema_loader = MemgraphSchemaLoader(None)  # We already have the schema
        node_labels = schema_loader.get_node_labels(self.schema, service_id)
        rel_types = schema_loader.get_relationship_types(self.schema, service_id)
        
        # Build schema context
        schema_context = f"""
Available Node Labels: {', '.join(node_labels[:20])}  # Showing first 20
Available Relationship Types: {', '.join(rel_types[:20])}  # Showing first 20

Use these labels and relationship types when constructing Cypher queries.
"""
        
        enhanced_query = f"{query}\n\n{schema_context}"
        return enhanced_query
    
    def search_by_node_label(
        self,
        label: str,
        limit: int = 10,
        properties: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for nodes by label, optionally filtering by properties.
        
        Args:
            label: Node label to search for
            limit: Maximum number of results
            properties: Optional property filters (e.g., {"canonical_name": "Apple"})
        """
        # Build Cypher query
        if properties:
            props_str = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"MATCH (n:{label}) WHERE {props_str} RETURN n LIMIT {limit}"
            params = properties
        else:
            query = f"MATCH (n:{label}) RETURN n LIMIT {limit}"
            params = {}
        
        try:
            results = self.db.execute_and_fetch(query, params)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def search_relationships(
        self,
        source_label: Optional[str] = None,
        rel_type: Optional[str] = None,
        target_label: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for relationships matching criteria.
        
        Args:
            source_label: Optional source node label
            rel_type: Optional relationship type
            target_label: Optional target node label
            limit: Maximum number of results
        """
        # Build Cypher query
        source_match = f"(a:{source_label})" if source_label else "(a)"
        target_match = f"(b:{target_label})" if target_label else "(b)"
        rel_match = f"-[r:{rel_type}]->" if rel_type else "-[r]->"
        
        query = f"MATCH {source_match}{rel_match}{target_match} RETURN a, r, b LIMIT {limit}"
        
        try:
            results = self.db.execute_and_fetch(query)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def search_by_concept(
        self,
        concept: str,
        service_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Intelligent search that uses schema to find relevant nodes and relationships.
        
        Args:
            concept: Concept to search for (e.g., "revenue growth", "Apple")
            service_id: Optional service filter (e.g., "ea", "fomc")
            limit: Maximum number of results
        """
        # Get relevant node labels from schema
        schema_loader = MemgraphSchemaLoader(None)
        node_labels = schema_loader.get_node_labels(self.schema, service_id)
        
        # Build a query that searches across multiple node types
        # This is a simple example - you could make it more sophisticated
        queries = []
        
        # Search in PERSON nodes
        if "PERSON" in node_labels:
            queries.append(f"""
                MATCH (p:PERSON)
                WHERE toLower(p.canonical_name) CONTAINS toLower('{concept}')
                RETURN p, 'PERSON' as type
                LIMIT {limit}
            """)
        
        # Search in ORG nodes
        if "ORG" in node_labels:
            queries.append(f"""
                MATCH (o:ORG)
                WHERE toLower(o.canonical_name) CONTAINS toLower('{concept}')
                RETURN o, 'ORG' as type
                LIMIT {limit}
            """)
        
        # Search in CONCEPT nodes
        if "CONCEPT" in node_labels:
            queries.append(f"""
                MATCH (c:CONCEPT)
                WHERE toLower(c.canonical_name) CONTAINS toLower('{concept}')
                RETURN c, 'CONCEPT' as type
                LIMIT {limit}
            """)
        
        # Execute all queries and combine results
        all_results = []
        for query in queries:
            try:
                results = self.db.execute_and_fetch(query)
                all_results.extend([dict(row) for row in results])
            except Exception as e:
                logger.warning(f"Query failed: {e}")
        
        return all_results[:limit]
    
    def get_node_relationships(
        self,
        node_label: str,
        node_property: str,
        node_value: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get all relationships for a specific node.
        
        Args:
            node_label: Label of the node
            node_property: Property to match on
            node_value: Value to match
            limit: Maximum relationships per direction
        """
        query = f"""
        MATCH (n:{node_label} {{{node_property}: $value}})
        OPTIONAL MATCH (n)-[r1]->(target)
        OPTIONAL MATCH (source)-[r2]->(n)
        RETURN 
            n,
            collect(DISTINCT {{rel: r1, target: target}})[0..{limit}] as outgoing,
            collect(DISTINCT {{rel: r2, source: source}})[0..{limit}] as incoming
        LIMIT 1
        """
        
        try:
            results = self.db.execute_and_fetch(query, {"value": node_value})
            result_list = list(results)
            return dict(result_list[0]) if result_list else {}
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {}


def connect_to_dragonfly() -> redis.Redis:
    """Connect to DragonFly (Redis-compatible) database."""
    host = os.getenv("DRAGONFLY_HOST", "localhost")
    port = int(os.getenv("DRAGONFLY_PORT", "6379"))
    password = os.getenv("DRAGONFLY_PASSWORD", None)
    db = int(os.getenv("DRAGONFLY_DB", "0"))
    ssl = os.getenv("DRAGONFLY_SSL", "false").lower() == "true"
    
    try:
        logger.info(f"Connecting to DragonFly at {host}:{port}...")
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
        client.ping()
        logger.info("✅ Successfully connected to DragonFly")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to DragonFly: {e}")
        raise


def main():
    """Example usage of schema-aware Memgraph search."""
    print("="*80)
    print("MEMGRAPH SEARCH WITH DRAGONFLY SCHEMA LOOKUP")
    print("="*80)
    print()
    
    # Step 1: Connect to DragonFly and load schema
    logger.info("Step 1: Loading schema from DragonFly...")
    dragonfly_client = connect_to_dragonfly()
    
    schema_loader = MemgraphSchemaLoader(dragonfly_client)
    
    # Load full schema
    full_schema = schema_loader.load_full_schema()
    if not full_schema:
        logger.error("❌ Failed to load schema from DragonFly")
        return
    
    logger.info("✅ Loaded schema from DragonFly")
    print(schema_loader.get_schema_summary(full_schema))
    print()
    
    # Optionally load service-specific schema
    service_id = "ea"  # Earnings Announcements
    service_schema = schema_loader.load_service_schema(service_id)
    if service_schema:
        logger.info(f"✅ Loaded {service_id} service schema")
        print(f"Service Schema: {service_schema.get('service_name', service_id)}")
        print(f"  - Node Types: {service_schema.get('total_node_types', 0)}")
        print(f"  - Relationship Types: {service_schema.get('total_relationship_types', 0)}")
        print()
    
    # Step 2: Initialize Memgraph search engine with schema
    logger.info("Step 2: Initializing Memgraph search engine...")
    memgraph_host = os.getenv("MEMGRAPH_HOST", "localhost")
    memgraph_port = int(os.getenv("MEMGRAPH_PORT", "7687"))
    memgraph_user = os.getenv("MEMGRAPH_USER", None)
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", None)
    
    search_engine = MemgraphSearchEngine(
        memgraph_host=memgraph_host,
        memgraph_port=memgraph_port,
        memgraph_user=memgraph_user,
        memgraph_password=memgraph_password,
        schema=full_schema  # Pass schema for schema-aware queries
    )
    logger.info("✅ Memgraph search engine initialized")
    print()
    
    # Step 3: Example searches using schema information
    
    # Example 1: Search by node label (using schema to know available labels)
    print("="*80)
    print("Example 1: Search by Node Label (PERSON)")
    print("="*80)
    person_nodes = search_engine.search_by_node_label("PERSON", limit=5)
    print(f"Found {len(person_nodes)} PERSON nodes:")
    for i, node in enumerate(person_nodes[:3], 1):
        node_data = node.get('n', {})
        print(f"  {i}. {node_data.get('canonical_name', 'Unknown')} (mentions: {node_data.get('mention_count', 0)})")
    print()
    
    # Example 2: Search by concept (uses schema to determine which node types to search)
    print("="*80)
    print("Example 2: Search by Concept ('revenue')")
    print("="*80)
    concept_results = search_engine.search_by_concept("revenue", service_id="ea", limit=5)
    print(f"Found {len(concept_results)} results for 'revenue':")
    for i, result in enumerate(concept_results[:3], 1):
        node = result.get('c') or result.get('o') or result.get('p', {})
        node_type = result.get('type', 'Unknown')
        print(f"  {i}. [{node_type}] {node.get('canonical_name', 'Unknown')}")
    print()
    
    # Example 3: Search relationships (using schema to know available relationship types)
    print("="*80)
    print("Example 3: Search Relationships (SAID)")
    print("="*80)
    relationships = search_engine.search_relationships(
        source_label="PERSON",
        rel_type="SAID",
        limit=5
    )
    print(f"Found {len(relationships)} SAID relationships:")
    for i, rel in enumerate(relationships[:3], 1):
        source = rel.get('a', {})
        target = rel.get('b', {})
        print(f"  {i}. {source.get('canonical_name', 'Unknown')} SAID -> {target.get('text', 'Unknown')[:50]}...")
    print()
    
    # Example 4: Get node with all relationships
    print("="*80)
    print("Example 4: Get Node with All Relationships")
    print("="*80)
    if person_nodes:
        first_person = person_nodes[0].get('n', {})
        person_name = first_person.get('canonical_name')
        if person_name:
            node_rels = search_engine.get_node_relationships(
                "PERSON",
                "canonical_name",
                person_name,
                limit=5
            )
            print(f"Relationships for {person_name}:")
            outgoing = node_rels.get('outgoing', [])
            incoming = node_rels.get('incoming', [])
            print(f"  Outgoing: {len(outgoing)} relationships")
            print(f"  Incoming: {len(incoming)} relationships")
    print()
    
    # Example 5: Schema-aware query building
    print("="*80)
    print("Example 5: Schema-Aware Query Enhancement")
    print("="*80)
    natural_query = "Find all people who mentioned revenue growth"
    enhanced_query = search_engine.build_schema_aware_query(natural_query, service_id="ea")
    print(f"Original Query: {natural_query}")
    print(f"\nEnhanced Query (with schema context):")
    print(enhanced_query[:200] + "..." if len(enhanced_query) > 200 else enhanced_query)
    print()
    
    print("="*80)
    print("✅ Examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

