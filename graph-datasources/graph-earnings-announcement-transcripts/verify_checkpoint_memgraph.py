#!/usr/bin/env python3
"""
Sample the processed checkpoint and query Memgraph graph statistics.

Behavior:
- Downloads the processed_transcripts checkpoint from GCS.
- Randomly samples N entries (default: 10) from the checkpoint.
- Connects to Memgraph and prints high-level graph statistics.

Requirements:
- Environment must provide access to GCS (Application Default Credentials or env).
- Memgraph must be reachable at MEMGRAPH_HOST:MEMGRAPH_PORT (optionally MEMGRAPH_USER/PASSWORD).
"""

import argparse
import json
import os
import pickle
import random
from typing import List, Tuple, Set, Dict, Any

from dotenv import load_dotenv
from google.cloud import storage
from gqlalchemy import Memgraph


def load_checkpoint_from_gcs(bucket_name: str, checkpoint_path: str) -> Set[Tuple[str, str, str]]:
    """Load the processed transcripts checkpoint from GCS (must exist)."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(checkpoint_path)
    if not blob.exists():
        raise RuntimeError(
            f"Checkpoint not found at gs://{bucket_name}/{checkpoint_path}"
        )
    data = blob.download_as_bytes()
    processed = pickle.loads(data)
    if not isinstance(processed, (set, list)):
        raise RuntimeError("Checkpoint contents are not a set or list of tuples")
    return set(processed)


def sample_items(items: Set[Tuple[str, str, str]], sample_size: int) -> List[Tuple[str, str, str]]:
    """Randomly sample up to sample_size items from the set."""
    items_list = list(items)
    if not items_list:
        return []
    if len(items_list) <= sample_size:
        random.shuffle(items_list)
        return items_list
    return random.sample(items_list, sample_size)


def connect_memgraph(host: str, port: int, user: str, password: str) -> Memgraph:
    """Establish a Memgraph connection. Raises if connection fails."""
    mg = Memgraph(host=host, port=port, username=user or None, password=password or None)
    # Simple sanity check
    mg.execute("RETURN 1;")
    return mg


def query_graph_stats(mg: Memgraph) -> Dict[str, Any]:
    """Collect high-level graph statistics."""
    stats: Dict[str, Any] = {}

    # Total nodes and relationships
    stats["counts"] = {
        "nodes": list(mg.execute_and_fetch("MATCH (n) RETURN count(n) AS c;"))[0]["c"],
        "relationships": list(mg.execute_and_fetch("MATCH ()-[r]->() RETURN count(r) AS c;"))[0]["c"],
    }

    # Nodes by label
    label_rows = mg.execute_and_fetch(
        """
        MATCH (n)
        WITH labels(n) AS lbls
        UNWIND lbls AS lbl
        RETURN lbl AS label, count(*) AS count
        ORDER BY count DESC
        LIMIT 20;
        """
    )
    stats["nodes_by_label"] = list(label_rows)

    # Relationship types
    rel_rows = mg.execute_and_fetch(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC
        LIMIT 20;
        """
    )
    stats["relationships_by_type"] = list(rel_rows)

    # Top entities by mention_count (if present)
    top_entities_rows = mg.execute_and_fetch(
        """
        MATCH (n)
        WHERE n.mention_count IS NOT NULL
        RETURN n.canonical_name AS canonical_name,
               n.mention_count AS mention_count,
               labels(n)[0] AS label
        ORDER BY mention_count DESC
        LIMIT 15;
        """
    )
    stats["top_entities_by_mentions"] = list(top_entities_rows)

    return stats


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sample checkpoint entries and query Memgraph graph statistics."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of checkpoint entries to sample (default: 10)",
    )
    args = parser.parse_args()

    # GCS configuration
    bucket_name = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
    checkpoint_path = os.getenv(
        "CHECKPOINT_PATH",
        "earnings-announcement-transcripts/.checkpoint/processed_transcripts.pkl",
    )

    # Memgraph configuration
    mg_host = os.getenv("MEMGRAPH_HOST", "localhost")
    mg_port = int(os.getenv("MEMGRAPH_PORT", "7687"))
    mg_user = os.getenv("MEMGRAPH_USER", "")
    mg_password = os.getenv("MEMGRAPH_PASSWORD", "")

    print(f"📥 Loading checkpoint from gs://{bucket_name}/{checkpoint_path} ...", flush=True)
    processed = load_checkpoint_from_gcs(bucket_name, checkpoint_path)
    print(f"✅ Loaded {len(processed)} processed transcript(s) from checkpoint", flush=True)

    sampled = sample_items(processed, args.sample_size)
    print(f"🎲 Sampled {len(sampled)} transcript(s):", flush=True)
    for symbol, year, quarter in sampled:
        print(f"   - {symbol} {year} Q{quarter}", flush=True)

    print(f"\n🗄️  Connecting to Memgraph at {mg_host}:{mg_port} ...", flush=True)
    mg = connect_memgraph(mg_host, mg_port, mg_user, mg_password)
    print("✅ Connected to Memgraph", flush=True)

    print("📊 Collecting graph statistics ...", flush=True)
    stats = query_graph_stats(mg)

    print("\n=== GRAPH STATISTICS ===", flush=True)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

