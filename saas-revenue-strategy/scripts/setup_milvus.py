#!/usr/bin/env python3
"""
Setup script for Milvus vector database.
Initializes the collection with proper schema for hybrid search.
"""

import os
import sys
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# Environment variables are read from the process environment.


def setup_milvus():
    """Initialize Milvus connection and create collection."""
    
    # Connection parameters
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    collection_name = os.getenv("MILVUS_COLLECTION_NAME", "saas_revenue_knowledge")
    embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "4096"))
    
    print(f"Connecting to Milvus at {host}:{port}...")
    
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        print("✓ Connected to Milvus successfully")
        
        # Check if collection already exists
        if utility.has_collection(collection_name):
            print(f"⚠ Collection '{collection_name}' already exists")
            user_input = input("Do you want to drop and recreate it? (yes/no): ")
            if user_input.lower() == "yes":
                utility.drop_collection(collection_name)
                print(f"✓ Dropped existing collection '{collection_name}'")
            else:
                print("Skipping collection creation")
                return
        
        # Define schema for hybrid search
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            # For sparse embeddings, we'll use a separate approach as Milvus handles them differently
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="SaaS Revenue Strategy Knowledge Base with Hybrid Search"
        )
        
        # Create collection
        print(f"Creating collection '{collection_name}'...")
        collection = Collection(
            name=collection_name,
            schema=schema
        )
        print(f"✓ Collection '{collection_name}' created successfully")
        
        # Create index for vector search
        print("Creating index for dense embeddings...")
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        
        collection.create_index(
            field_name="dense_embedding",
            index_params=index_params
        )
        print("✓ Index created successfully")
        
        # Load collection
        collection.load()
        print("✓ Collection loaded into memory")
        
        print("\n" + "="*60)
        print("Milvus setup completed successfully!")
        print("="*60)
        print(f"Collection name: {collection_name}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Status: Ready for use")
        
    except Exception as e:
        print(f"✗ Error during setup: {e}")
        sys.exit(1)
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    setup_milvus()
