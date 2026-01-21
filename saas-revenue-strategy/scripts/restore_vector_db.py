#!/usr/bin/env python3
"""
Restore script for Milvus vector database.
Restores a collection from a snapshot tar.gz file.
"""

import os
import sys
import tarfile
import json
from pathlib import Path
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# Environment variables are read from the process environment.


def restore_vector_db(input_path: str):
    """Restore Milvus collection from a tar.gz backup file."""
    
    if not os.path.exists(input_path):
        print(f"✗ Backup file not found: {input_path}")
        sys.exit(1)
    
    # Connection parameters
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    
    print(f"Restoring from: {input_path}")
    print(f"Connecting to Milvus at {host}:{port}...")
    
    # Extract backup files
    temp_dir = Path("./temp_restore")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Extract tar.gz
        print("Extracting backup archive...")
        with tarfile.open(input_path, "r:gz") as tar:
            tar.extractall(temp_dir)
        
        # Read metadata
        metadata_file = temp_dir / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        collection_name = metadata["collection_name"]
        num_entities = metadata["exported_entities"]
        
        print(f"Backup info:")
        print(f"  Collection: {collection_name}")
        print(f"  Entities: {num_entities}")
        print(f"  Timestamp: {metadata['backup_timestamp']}")
        
        # Read data
        data_file = temp_dir / "data.json"
        with open(data_file, "r") as f:
            data = json.load(f)
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        print("✓ Connected to Milvus")
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            print(f"⚠ Collection '{collection_name}' already exists")
            user_input = input("Do you want to drop and recreate it? (yes/no): ")
            if user_input.lower() == "yes":
                utility.drop_collection(collection_name)
                print(f"✓ Dropped existing collection '{collection_name}'")
            else:
                print("Restore cancelled")
                return
        
        # Recreate collection from schema
        print("Recreating collection...")
        
        # Build schema from metadata
        fields = []
        for field_info in metadata["schema"]["fields"]:
            dtype_str = field_info["type"].split(".")[-1]
            dtype = getattr(DataType, dtype_str)
            
            field_params = field_info["params"]
            
            if dtype == DataType.FLOAT_VECTOR:
                field = FieldSchema(
                    name=field_info["name"],
                    dtype=dtype,
                    dim=field_params.get("dim", 1536)
                )
            elif dtype == DataType.VARCHAR:
                field = FieldSchema(
                    name=field_info["name"],
                    dtype=dtype,
                    max_length=field_params.get("max_length", 65535)
                )
            elif field_info["name"] == "id":
                field = FieldSchema(
                    name=field_info["name"],
                    dtype=dtype,
                    is_primary=True,
                    auto_id=True
                )
            else:
                field = FieldSchema(
                    name=field_info["name"],
                    dtype=dtype
                )
            
            fields.append(field)
        
        schema = CollectionSchema(
            fields=fields,
            description="Restored from backup"
        )
        
        collection = Collection(
            name=collection_name,
            schema=schema
        )
        print(f"✓ Collection '{collection_name}' created")
        
        # Insert data if available
        if len(data) > 0:
            print(f"Inserting {len(data)} entities...")
            
            # Prepare data for insertion
            # Note: In production, you'd need to handle this more carefully
            # based on your actual schema
            collection.insert(data)
            collection.flush()
            print(f"✓ Inserted {len(data)} entities")
        
        # Create index
        print("Creating index...")
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        
        collection.create_index(
            field_name="dense_embedding",
            index_params=index_params
        )
        print("✓ Index created")
        
        # Load collection
        collection.load()
        print("✓ Collection loaded")
        
        print("\n" + "="*60)
        print("Restore completed successfully!")
        print("="*60)
        print(f"Collection: {collection_name}")
        print(f"Entities: {collection.num_entities}")
        
    except Exception as e:
        print(f"✗ Error during restore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup temp files
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        
        connections.disconnect("default")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Restore Milvus vector database")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input path for backup file"
    )
    
    args = parser.parse_args()
    restore_vector_db(args.input)
