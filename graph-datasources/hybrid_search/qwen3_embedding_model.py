"""
Qwen3 Embedding Model for LlamaIndex PropertyGraphIndex.

This module provides a custom embedding model that uses the Qwen3 Embedding API
to generate embeddings compatible with LlamaIndex.
"""

import time
import logging
import requests
from typing import List, Optional
from llama_index.core.embeddings import BaseEmbedding
import redis
import json
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logger = logging.getLogger(__name__)


def get_embedding_config_from_dragonfly(key: str = "llama:milvus:longterm_memory:hybrid_config") -> dict:
    """
    Load embedding API configuration from Dragonfly.
    
    Args:
        key: Redis key to fetch config from
        
    Returns:
        Dictionary containing embedding API configuration
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
        
        logger.info(f"Fetching embedding config for key: {key}")
        data = client.get(key)
        
        if data:
            try:
                config_data = json.loads(data)
                embedding_config = config_data.get("embedding_api_config", {})
                logger.info("✅ Embedding config retrieved and parsed.")
                return embedding_config
            except json.JSONDecodeError:
                logger.error("❌ Failed to parse config JSON.")
                return {}
        else:
            logger.warning(f"⚠️ Key {key} not found in Dragonfly. Using defaults.")
            return {}
            
    except Exception as e:
        logger.error(f"❌ Error fetching from Dragonfly: {e}")
        return {}


class Qwen3EmbeddingModel(BaseEmbedding):
    """
    Custom embedding model using Qwen3 Embedding API.
    
    Compatible with LlamaIndex BaseEmbedding interface.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint_path: str = "/v1/embeddings/qwen3/embeddings",
        model_id: str = "qwen/qwen3-embedding-8b",
        encoding_format: str = "float",
        dimension: int = 4096,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize Qwen3 Embedding Model.
        
        Args:
            base_url: Base URL for the embedding API (defaults to config or API)
            endpoint_path: API endpoint path
            model_id: Model identifier
            encoding_format: Encoding format ('float' or 'base64')
            dimension: Embedding dimension (4096 for Qwen3)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial retry delay in seconds
        """
        # Initialize BaseEmbedding with model_name (required Pydantic field)
        super().__init__(model_name=model_id, **kwargs)
        
        # Try to load config from Dragonfly first
        dragonfly_config = get_embedding_config_from_dragonfly()
        
        # Store config as PRIVATE attributes (not Pydantic fields) to avoid validation errors
        self._base_url = base_url or dragonfly_config.get(
            "base_url", 
            "https://api.blacksmith.deerfieldgreen.com"
        )
        self._endpoint_path = dragonfly_config.get("endpoint_path", endpoint_path)
        self._model_id = dragonfly_config.get("model_id", model_id)
        self._encoding_format = dragonfly_config.get("encoding_format", encoding_format)
        self._dimension_value = dragonfly_config.get("dim", dimension)
        
        # Request parameters
        request_params = dragonfly_config.get("request_params", {})
        self._timeout = request_params.get("timeout_seconds", timeout)
        self._max_retries = request_params.get("max_retries", max_retries)
        self._retry_delay = request_params.get("retry_delay_seconds", retry_delay)
        
        self._api_url = f"{self._base_url}{self._endpoint_path}"
        
        logger.info(f"Qwen3EmbeddingModel initialized:")
        logger.info(f"  API URL: {self._api_url}")
        logger.info(f"  Model: {self._model_id}")
        logger.info(f"  Dimension: {self._dimension_value}")
        logger.info(f"  Timeout: {self._timeout}s")
        logger.info(f"  Max Retries: {self._max_retries}")
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension_value
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self._get_text_embeddings([text])[0]
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []
        
        retry_delay = self._retry_delay
        
        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    self._api_url,  # Use private attribute
                    json={
                        "input": texts if len(texts) > 1 else texts[0],
                        "model": self._model_id,  # Use private attribute
                        "encoding_format": self._encoding_format  # Use private attribute
                    },
                    timeout=self._timeout,  # Use private attribute
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle both single and batch responses
                    if isinstance(result.get("data"), list):
                        embeddings = [item["embedding"] for item in result["data"]]
                    else:
                        # Single embedding response
                        embeddings = [result.get("data", {}).get("embedding", [])]
                    
                    logger.debug(f"Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0]) if embeddings else 0})")
                    return embeddings
                    
                else:
                    error_msg = f"API returned status {response.status_code}: {response.text}"
                    logger.warning(f"Attempt {attempt + 1}/{self._max_retries}: {error_msg}")
                    
                    if attempt < self._max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise Exception(f"Failed after {self._max_retries} attempts: {error_msg}")
                        
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self._max_retries}: Network error: {e}")
                if attempt < self._max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"Network error after {self._max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"Unexpected error generating embeddings: {e}")
                raise
        
        raise Exception(f"Failed to generate embeddings after {self._max_retries} attempts")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        return self._get_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of _get_text_embedding."""
        # For now, use sync version (can be enhanced with async requests later)
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of _get_text_embeddings."""
        # For now, use sync version (can be enhanced with async requests later)
        return self._get_text_embeddings(texts)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of _get_query_embedding."""
        return self._get_query_embedding(query)

