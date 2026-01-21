#!/usr/bin/env python3
"""
Backup script for Milvus vector database.
Creates a snapshot of the collection data.
"""

import os
import sys
import tarfile
import json
import shutil
from datetime import datetime
from pathlib import Path
from pymilvus import connections, Collection, utility

# Environment variables are read from the process environment.


def _resolve_output_paths(output_spec: str | None) -> tuple[Path, Path | None]:
    if output_spec is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_path = Path(f"./backups/milvus_snapshot_{timestamp}.tar.gz")
        return local_path, None

    if "@" not in output_spec:
        return Path(output_spec), None

    local_spec, dest_spec = output_spec.rsplit("@", 1)
    if not local_spec or not dest_spec:
        raise ValueError("Output must be in the form <local_path>@<destination_path>")

    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent.parent
    local_path = Path(local_spec)
    dest_path = Path(dest_spec)
    if not dest_path.is_absolute():
        dest_path = workspace_root / dest_path

    return local_path, dest_path


def backup_vector_db(output_path: str = None):
    """Backup Milvus collection to a tar.gz file."""
    
    # Connection parameters
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    collection_name = os.getenv("MILVUS_COLLECTION_NAME", "saas_revenue_knowledge")
    
    try:
        output_file, destination_path = _resolve_output_paths(output_path)
    except ValueError as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to Milvus at {host}:{port}...")
    
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        print("✓ Connected to Milvus")
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            print(f"✗ Collection '{collection_name}' does not exist")
            sys.exit(1)
        
        # Get collection
        collection = Collection(collection_name)
        collection.load()
        
        # Get collection stats
        num_entities = collection.num_entities
        print(f"Collection '{collection_name}' has {num_entities} entities")
        
        if num_entities == 0:
            print("⚠ Warning: Collection is empty, creating empty backup")
        
        # Create metadata
        metadata = {
            "collection_name": collection_name,
            "num_entities": num_entities,
            "backup_timestamp": datetime.now().isoformat(),
            "schema": {
                "fields": [
                    {
                        "name": field.name,
                        "type": str(field.dtype),
                        "params": field.params
                    }
                    for field in collection.schema.fields
                ]
            }
        }
        
        # Query all data (in production, you'd want to batch this)
        print("Exporting collection data...")
        
        # For simplicity, we'll export a limited set
        # In production, implement pagination
        limit = min(num_entities, 10000)  # Limit for safety
        if num_entities > 0:
            results = collection.query(
                expr="id >= 0",
                output_fields=["*"],
                limit=limit
            )
            metadata["exported_entities"] = len(results)
        else:
            results = []
            metadata["exported_entities"] = 0
        
        # Create temporary directory for backup files
        temp_dir = Path("./temp_backup")
        temp_dir.mkdir(exist_ok=True)
        
        # Write metadata
        metadata_file = temp_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Write data
        data_file = temp_dir / "data.json"
        with open(data_file, "w") as f:
            json.dump(results, f)
        
        # Create tar.gz archive
        print(f"Creating backup archive: {output_file}")
        with tarfile.open(output_file, "w:gz") as tar:
            tar.add(metadata_file, arcname="metadata.json")
            tar.add(data_file, arcname="data.json")
        
        # Cleanup temp files
        metadata_file.unlink()
        data_file.unlink()
        temp_dir.rmdir()
        
        print("\n" + "="*60)
        print("Backup completed successfully!")
        print("="*60)
        print(f"Backup file: {output_file}")
        print(f"Entities backed up: {metadata['exported_entities']} / {num_entities}")

        if destination_path is not None:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_file, destination_path)
            print(f"Copied backup to: {destination_path}")
        
    except Exception as e:
        print(f"✗ Error during backup: {e}")
        sys.exit(1)
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup Milvus vector database")
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for backup file (default: ./backups/milvus_snapshot_<timestamp>.tar.gz)"
    )
    
    args = parser.parse_args()
    backup_vector_db(args.output)
