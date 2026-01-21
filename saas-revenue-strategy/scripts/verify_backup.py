#!/usr/bin/env python3
"""
Verify a Milvus backup archive by extracting and validating contents.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path


def _fail(message: str) -> None:
    print(f"✗ {message}")
    sys.exit(1)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        _fail(f"Failed to read {path.name}: {exc}")


def verify_backup(archive_path: Path, extract_dir: Path) -> None:
    if not archive_path.exists():
        _fail(f"Archive does not exist: {archive_path}")
    if not archive_path.is_file():
        _fail(f"Archive is not a file: {archive_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    archive_size = archive_path.stat().st_size
    print(f"Archive: {archive_path}")
    print(f"Archive size: {archive_size} bytes")
    print(f"Extract target: {extract_dir.resolve()}")
    print("Listing archive contents:")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            print(f"Archive members: {len(members)}")
            for member in members:
                print(f"- {member.name} ({member.size} bytes)")
            print("Extracting archive...")
            tar.extractall(path=extract_dir, filter="data")
    except Exception as exc:
        _fail(f"Failed to extract archive: {exc}")

    metadata_path = extract_dir / "metadata.json"
    data_path = extract_dir / "data.json"

    if not metadata_path.exists():
        _fail("metadata.json not found after extraction")
    if not data_path.exists():
        _fail("data.json not found after extraction")

    print(f"Reading metadata: {metadata_path}")
    metadata = _load_json(metadata_path)
    print(f"Reading data: {data_path}")
    data = _load_json(data_path)

    if not isinstance(metadata, dict):
        _fail("metadata.json is not an object")
    if not isinstance(data, list):
        _fail("data.json is not a list")

    exported = metadata.get("exported_entities")
    if not isinstance(exported, int):
        _fail("metadata.json missing integer exported_entities")

    if exported != len(data):
        _fail(
            f"exported_entities mismatch: metadata={exported}, data_rows={len(data)}"
        )

    collection_name = metadata.get("collection_name")
    num_entities = metadata.get("num_entities")
    backup_timestamp = metadata.get("backup_timestamp")
    schema_fields = metadata.get("schema", {}).get("fields", [])

    print("\n" + "=" * 60)
    print("Backup verification completed successfully!")
    print("=" * 60)
    print(f"Archive: {archive_path}")
    print(f"Extracted to: {extract_dir}")
    print(f"Collection name: {collection_name}")
    print(f"num_entities: {num_entities}")
    print(f"backup_timestamp: {backup_timestamp}")
    print(f"schema_fields: {len(schema_fields)}")
    print(f"exported_entities: {exported}")
    print(f"data_rows: {len(data)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a Milvus backup archive by extracting and validating it."
    )
    parser.add_argument(
        "--archive",
        "-a",
        required=True,
        help="Path to the backup .tar.gz archive",
    )
    parser.add_argument(
        "--extract-dir",
        "-x",
        required=True,
        help="Directory to extract and verify contents",
    )
    args = parser.parse_args()

    verify_backup(Path(args.archive), Path(args.extract_dir))


if __name__ == "__main__":
    main()
