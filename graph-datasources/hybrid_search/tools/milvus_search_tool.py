"""
Milvus Hybrid Search Tool for LlamaIndex Agent.

This module provides a hybrid search function that combines dense vector search
(dense embeddings) with sparse BM25 search (raw text) using Milvus.
Configuration is loaded from DragonFly Redis-compatible store.
"""

import os
import sys
import json
import logging
import redis
from typing import List, Dict, Optional, Any

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config

logger = logging.getLogger(__name__)

# Config key for blacksmith_embeddings collection
BLACKSMITH_EMBEDDINGS_CONFIG_KEY = "llama:milvus:blacksmith_embeddings:hybrid_config"

# Cached config
_cached_milvus_config = None


def get_dragonfly_client():
    """Get a DragonFly (Redis-compatible) client."""
    return redis.Redis(
        host=config.DRAGONFLY_HOST,
        port=config.DRAGONFLY_PORT,
        password=config.DRAGONFLY_PASSWORD,
        db=config.DRAGONFLY_DB,
        ssl=config.DRAGONFLY_SSL,
        decode_responses=True,
        socket_timeout=5
    )


def load_milvus_config_from_dragonfly(config_key: str = BLACKSMITH_EMBEDDINGS_CONFIG_KEY) -> Optional[Dict[str, Any]]:
    """
    Load Milvus collection configuration from DragonFly.
    
    Args:
        config_key: Redis key to fetch config from (default: blacksmith_embeddings config key)
        
    Returns:
        Dictionary containing Milvus configuration or None if not found
    """
    try:
        logger.info(f"Connecting to Dragonfly at {config.DRAGONFLY_HOST}:{config.DRAGONFLY_PORT}...")
        client = get_dragonfly_client()
        
        logger.info(f"Fetching Milvus config for key: {config_key}")
        data = client.get(config_key)
        
        if data:
            try:
                config_data = json.loads(data)
                logger.info("✅ Milvus config retrieved and parsed from DragonFly.")
                return config_data
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse config JSON: {e}")
                return None
        else:
            logger.warning(f"⚠️ Key '{config_key}' not found in Dragonfly.")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error fetching Milvus config from Dragonfly: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_milvus_config_cached(config_key: str = BLACKSMITH_EMBEDDINGS_CONFIG_KEY) -> Optional[Dict[str, Any]]:
    """
    Get Milvus configuration with caching.
    
    Args:
        config_key: Redis key to fetch config from
        
    Returns:
        Cached configuration dictionary or None if not found
    """
    global _cached_milvus_config
    
    if _cached_milvus_config is None:
        logger.info("Loading Milvus configuration from DragonFly...")
        _cached_milvus_config = load_milvus_config_from_dragonfly(config_key)
    
    return _cached_milvus_config

try:
    from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker
except ImportError:
    logger.error("pymilvus not installed. Install with: pip install pymilvus")
    MilvusClient = None
    AnnSearchRequest = None
    RRFRanker = None
    WeightedRanker = None


def milvus_hybrid_search_func(
    query: str,
    collection_name: Optional[str] = None,
    dense_field: Optional[str] = None,
    sparse_field: Optional[str] = None,
    limit: Optional[int] = None,
    dense_candidates: Optional[int] = None,
    sparse_candidates: Optional[int] = None,
    use_weighted_ranker: bool = False,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: Optional[int] = None,
    filter_expr: Optional[str] = None,
    output_fields: Optional[List[str]] = None,
    config_key: Optional[str] = None
) -> str:
    """
    Perform hybrid search in Milvus combining dense vector search and sparse BM25 search.
    Configuration is loaded from DragonFly if not explicitly provided.
    
    Args:
        query: Natural language query string
        collection_name: Milvus collection name (defaults to config from DragonFly)
        dense_field: Dense vector field name (defaults to config from DragonFly)
        sparse_field: Sparse vector field name (defaults to config from DragonFly)
        limit: Final number of results to return (defaults to config from DragonFly)
        dense_candidates: Number of candidates from dense search (defaults to config from DragonFly)
        sparse_candidates: Number of candidates from sparse search (defaults to config from DragonFly)
        use_weighted_ranker: If True, use WeightedRanker instead of RRFRanker
        dense_weight: Weight for dense search results (only used with WeightedRanker)
        sparse_weight: Weight for sparse search results (only used with WeightedRanker)
        rrf_k: RRF constant (defaults to config from DragonFly)
        filter_expr: Optional filter expression (e.g., 'user_id == "user123"')
        output_fields: List of fields to return in results (defaults to config from DragonFly)
        config_key: Optional DragonFly config key (defaults to BLACKSMITH_EMBEDDINGS_CONFIG_KEY)
    
    Returns:
        Formatted string with search results
    """
    try:
        if MilvusClient is None:
            return "Error: pymilvus is not installed. Install with: pip install pymilvus"
        
        # Load configuration from DragonFly
        config_key_to_use = config_key or BLACKSMITH_EMBEDDINGS_CONFIG_KEY
        milvus_config = get_milvus_config_cached(config_key_to_use)
        
        # Initialize field names (will be overridden by config if available)
        text_field_name = "chunk_text"
        metadata_field_name = "metadata"
        
        if milvus_config:
            logger.info("Using Milvus configuration from DragonFly")
            # Extract settings from config
            collection_name = collection_name or milvus_config.get("collection_name")
            dense_field = dense_field or milvus_config.get("dense_vector_field") or milvus_config.get("dense_field")
            sparse_field = sparse_field or milvus_config.get("sparse_vector_field") or milvus_config.get("sparse_field")
            
            # Extract ranker settings
            # hybrid_ranker can be a string like "RRFRanker" or "WeightedRanker"
            hybrid_ranker = milvus_config.get("hybrid_ranker", "RRFRanker")
            hybrid_ranker_params = milvus_config.get("hybrid_ranker_params", {})
            
            # Determine ranker type from string or dict
            if isinstance(hybrid_ranker, str):
                ranker_name_lower = hybrid_ranker.lower()
                if "weighted" in ranker_name_lower:
                    use_weighted_ranker = True
                    # Get weights from params or use defaults
                    dense_weight = hybrid_ranker_params.get("dense_weight", dense_weight)
                    sparse_weight = hybrid_ranker_params.get("sparse_weight", sparse_weight)
                    logger.info(f"Using WeightedRanker from config: dense={dense_weight}, sparse={sparse_weight}")
                else:
                    # Default to RRF
                    use_weighted_ranker = False
                    rrf_k = rrf_k or hybrid_ranker_params.get("k", config.MILVUS_RRF_K)
                    logger.info(f"Using RRFRanker from config: k={rrf_k}")
            elif isinstance(hybrid_ranker, dict):
                # Legacy format: hybrid_ranker as dict
                ranker_type = hybrid_ranker.get("type", "rrf")
                if ranker_type == "weighted":
                    use_weighted_ranker = True
                    dense_weight = hybrid_ranker.get("dense_weight", dense_weight)
                    sparse_weight = hybrid_ranker.get("sparse_weight", sparse_weight)
                else:
                    rrf_k = rrf_k or hybrid_ranker.get("k", config.MILVUS_RRF_K)
            
            # Extract limits and candidates
            limit = limit or milvus_config.get("similarity_top_k", milvus_config.get("limit", config.MILVUS_HYBRID_LIMIT))
            dense_candidates = dense_candidates or milvus_config.get("dense_candidates", config.MILVUS_DENSE_CANDIDATES)
            sparse_candidates = sparse_candidates or milvus_config.get("sparse_candidates", config.MILVUS_SPARSE_CANDIDATES)
            
            # Extract field names from config
            text_field_name = milvus_config.get("text_field", "chunk_text")
            metadata_field_name = milvus_config.get("metadata_json_field", "metadata")
            primary_key_field = milvus_config.get("primary_key_field", "unique_id")
            
            # Extract output fields - use text_field from config
            if output_fields is None:
                # Default to text_field and metadata_field from config
                output_fields = [text_field_name, metadata_field_name]
            
            # Extract Milvus connection settings
            milvus_uri = milvus_config.get("milvus_uri") or config.MILVUS_URI
            milvus_token = milvus_config.get("milvus_token") or config.MILVUS_TOKEN
        else:
            logger.warning("⚠️ Could not load config from DragonFly, using defaults from config.py")
            # Fallback to config.py defaults
            collection_name = collection_name or config.MILVUS_COLLECTION_NAME
            dense_field = dense_field or config.MILVUS_DENSE_FIELD
            sparse_field = sparse_field or config.MILVUS_SPARSE_FIELD
            limit = limit or config.MILVUS_HYBRID_LIMIT
            dense_candidates = dense_candidates or config.MILVUS_DENSE_CANDIDATES
            sparse_candidates = sparse_candidates or config.MILVUS_SPARSE_CANDIDATES
            rrf_k = rrf_k or config.MILVUS_RRF_K
            output_fields = output_fields or ["text", "metadata"]
            # Use defaults for field names
            text_field_name = "text"
            metadata_field_name = "metadata"
            milvus_uri = config.MILVUS_URI
            milvus_token = config.MILVUS_TOKEN
        
        # Validate required fields
        if not collection_name:
            return "Error: Collection name is required. Please set MILVUS_COLLECTION_NAME or provide config in DragonFly."
        if not dense_field:
            return "Error: Dense field name is required. Please set MILVUS_DENSE_FIELD or provide config in DragonFly."
        if not sparse_field:
            return "Error: Sparse field name is required. Please set MILVUS_SPARSE_FIELD or provide config in DragonFly."
        
        # Initialize Milvus client
        logger.info(f"Connecting to Milvus at {milvus_uri}")
        client = MilvusClient(
            uri=milvus_uri,
            token=milvus_token
        )
        
        # Verify collection exists
        if not client.has_collection(collection_name=collection_name):
            client.close()
            return f"Error: Collection '{collection_name}' does not exist in Milvus."
        
        # Get collection schema to validate output_fields
        try:
            collection_info = client.describe_collection(collection_name=collection_name)
            # Extract field names from schema
            schema_fields = []
            if hasattr(collection_info, 'fields'):
                schema_fields = [field.get('name', '') for field in collection_info.fields if field.get('name')]
            elif isinstance(collection_info, dict) and 'fields' in collection_info:
                schema_fields = [field.get('name', '') for field in collection_info['fields'] if field.get('name')]
            
            # Filter output_fields to only include fields that exist in the collection
            if output_fields and schema_fields:
                original_output_fields = output_fields.copy()
                output_fields = [field for field in output_fields if field in schema_fields]
                if len(output_fields) < len(original_output_fields):
                    missing_fields = set(original_output_fields) - set(output_fields)
                    logger.warning(f"Removed non-existent fields from output_fields: {missing_fields}")
                    logger.info(f"Available fields in collection: {schema_fields}")
                    logger.info(f"Using output_fields: {output_fields}")
            
            # If no valid output_fields, use all available fields (except vector fields)
            if not output_fields and schema_fields:
                # Exclude vector fields and id field
                vector_field_names = [dense_field, sparse_field]
                output_fields = [field for field in schema_fields if field not in vector_field_names and field != 'id']
                logger.info(f"No output_fields specified, using available fields: {output_fields}")
        except Exception as e:
            logger.warning(f"Could not get collection schema: {e}. Proceeding with specified output_fields.")
            # If we can't get schema, use empty list to let Milvus return default fields
            if not output_fields:
                output_fields = []
        
        # Generate dense embedding for the query
        logger.info(f"Generating embedding for query: {query[:100]}...")
        try:
            # Try to use Qwen3 embedding model if available
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
            from qwen3_embedding_model import Qwen3EmbeddingModel
            embed_model = Qwen3EmbeddingModel()
            query_embedding = embed_model.get_query_embedding(query)
            logger.info(f"Generated embedding with dimension: {len(query_embedding)}")
        except Exception as e:
            logger.warning(f"Failed to use Qwen3EmbeddingModel: {e}")
            # Fallback: try to get embedding from environment or use a placeholder
            # In production, you should always have an embedding model
            return f"Error: Could not generate embedding for query. Embedding model error: {str(e)}"
        
        # Create dense vector search request
        logger.info("Creating dense vector search request...")
        dense_search_params = {
            "metric_type": "COSINE",
            "params": {}
        }
        
        # Build dense request kwargs with optional filter expression
        dense_req_kwargs = {
            "data": [query_embedding],  # List of query vectors
            "anns_field": dense_field,
            "param": dense_search_params,
            "limit": dense_candidates
        }
        if filter_expr:
            dense_req_kwargs["expr"] = filter_expr
            logger.info(f"Added filter to dense search: {filter_expr}")
        
        dense_req = AnnSearchRequest(**dense_req_kwargs)
        
        # Create sparse BM25 search request
        logger.info("Creating sparse BM25 search request...")
        sparse_search_params = {
            "metric_type": "BM25",  # CRITICAL: Must be BM25 for sparse vectors
            "params": {}
        }
        
        # Build sparse request kwargs with optional filter expression
        sparse_req_kwargs = {
            "data": [query],  # Raw query text (NOT a vector!)
            "anns_field": sparse_field,
            "param": sparse_search_params,
            "limit": sparse_candidates
        }
        if filter_expr:
            sparse_req_kwargs["expr"] = filter_expr
            logger.info(f"Added filter to sparse search: {filter_expr}")
        
        sparse_req = AnnSearchRequest(**sparse_req_kwargs)
        
        # Create ranker
        if use_weighted_ranker:
            logger.info(f"Using WeightedRanker (dense={dense_weight}, sparse={sparse_weight})")
            ranker = WeightedRanker(dense_weight, sparse_weight)
        else:
            logger.info(f"Using RRFRanker (k={rrf_k})")
            ranker = RRFRanker(k=rrf_k)
        
        # Execute hybrid search
        logger.info(f"Executing hybrid search on collection '{collection_name}'...")
        results = client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=limit,
            output_fields=output_fields
        )
        
        # Close client
        client.close()
        
        # Process and format results
        if not results or len(results) == 0:
            return f"No results found for query: '{query}'"
        
        # Results structure: List[List[Dict]]
        query_results = results[0]  # Get results for first (and usually only) query
        
        if not query_results:
            return f"No results found for query: '{query}'"
        
        # Format results
        formatted_results = []
        formatted_results.append(f"Found {len(query_results)} results for query: '{query}'\n")
        
        for i, hit in enumerate(query_results, 1):
            # Extract fields using config field names
            text = hit.get(text_field_name, hit.get("text", ""))
            distance = hit.get("distance", 0.0)
            similarity = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)  # Convert to similarity
            
            # Extract metadata using config field name
            metadata = hit.get(metadata_field_name, hit.get("metadata", {}))
            doc_id = hit.get("id", "N/A")
            
            # Build result string
            result_str = f"\n--- Result {i} ---\n"
            result_str += f"Similarity Score: {similarity:.4f}\n"
            result_str += f"Document ID: {doc_id}\n"
            
            if text:
                # Truncate long text
                display_text = text[:500] + "..." if len(text) > 500 else text
                result_str += f"Text: {display_text}\n"
            
            if metadata:
                result_str += f"Metadata: {metadata}\n"
            
            # Add other output fields (excluding already displayed fields and vector fields)
            excluded_fields = {text_field_name, "text", metadata_field_name, "metadata", "id", "distance", dense_field, sparse_field}
            for field in output_fields:
                if field not in excluded_fields and field in hit:
                    value = hit[field]
                    # Truncate long string values
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    result_str += f"{field}: {value}\n"
            
            formatted_results.append(result_str)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        import traceback
        logger.error(f"Error in Milvus hybrid search: {e}")
        logger.error(traceback.format_exc())
        return f"Error executing Milvus hybrid search: {repr(e)}"


# Create the Tool object
from llama_index.core.tools import FunctionTool

milvus_search_tool = FunctionTool.from_defaults(
    fn=milvus_hybrid_search_func,
    name="milvus_search",
    description="""Hybrid search tool that searches Milvus vector database using both dense vector embeddings and sparse BM25 text search.
    
    This tool combines semantic similarity (dense vectors) with keyword matching (BM25) to provide comprehensive search results.
    
    Use this tool when:
    - You need to search through document collections or long-term memory
    - You want semantic understanding combined with keyword matching
    - You're looking for documents, conversations, or text content
    - Graph search doesn't return relevant results
    
    The tool automatically:
    - Generates dense embeddings for semantic search
    - Uses BM25 for keyword-based search
    - Combines both using RRF (Reciprocal Rank Fusion) ranking
    - Returns top results with similarity scores
    
    Parameters:
    - query: Natural language query string (required)
    - filter_expr: Optional filter expression (e.g., 'user_id == "user123"')
    - limit: Number of results to return (default: 10)
    
    Example queries:
    - "What did we discuss about Tesla revenue?"
    - "Find conversations about machine learning"
    - "Search for documents mentioning financial metrics"
    """
)

if __name__ == "__main__":
    # Test the hybrid search
    import logging
    logging.basicConfig(level=logging.INFO)
    
    test_query = "machine learning algorithms"
    print(f"\n=== Testing Milvus Hybrid Search ===\n")
    print(f"Query: '{test_query}'\n")
    
    result = milvus_hybrid_search_func(test_query)
    print(result)
    print("\n=== Test Complete ===")

