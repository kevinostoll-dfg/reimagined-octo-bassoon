#!/usr/bin/env python3
"""
Memgraph & GCS Stats Inspector
==============================
This script provides a comprehensive overview of the data ingestion status by comparing:
1. Source Data (GCS): Total transcript files available.
2. Processing State (Checkpoint): How many transcripts have been marked as processed.
3. Destination Data (Memgraph): Actual nodes and relationships in the database.

It helps verify if the ingestion pipeline is working as expected and identifying data discrepancies.
"""

import os
import sys
import pickle
import logging
from typing import Dict, Set, Tuple, List
from google.cloud import storage
from gqlalchemy import Memgraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging (simplified for this script)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
# Memgraph
MEMGRAPH_HOST = "test-memgraph.blacksmith.deerfieldgreen.com"
MEMGRAPH_PORT = 7687
MEMGRAPH_USER = "memgraphdb"
MEMGRAPH_PASSWORD = ""

# GCS
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "earnings-announcement-transcripts")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", f"{GCS_BASE_PATH}/.checkpoint/processed_transcripts.pkl")

# --------------------------------------------------------------------------
# STATS GATHERING FUNCTIONS
# --------------------------------------------------------------------------

def get_gcs_stats() -> Dict:
    """Count total transcript files in GCS."""
    print("☁️  Inspecting GCS Bucket...", end=" ", flush=True)
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        # Count JSON files in the transcripts directory
        prefix = f"{GCS_BASE_PATH}/"
        blobs = bucket.list_blobs(prefix=prefix)
        
        transcript_count = 0
        file_size_total = 0
        
        for blob in blobs:
            if blob.name.endswith('.json') and '.checkpoint' not in blob.name:
                transcript_count += 1
                file_size_total += blob.size
                
        print("✅")
        return {
            "total_transcripts": transcript_count,
            "total_size_bytes": file_size_total,
            "status": "success"
        }
    except Exception as e:
        print("❌")
        return {"status": "error", "error": str(e)}

def get_checkpoint_stats() -> Dict:
    """Read local/remote checkpoint to see processed count."""
    print("💾 Inspecting Checkpoint...", end=" ", flush=True)
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(CHECKPOINT_PATH)
        
        if not blob.exists():
            print("⚠️  (Not Found)")
            return {"count": 0, "status": "not_found"}
            
        data = blob.download_as_bytes()
        processed_set = pickle.loads(data)
        
        print("✅")
        return {
            "count": len(processed_set),
            "status": "success",
            "last_updated": blob.updated
        }
    except Exception as e:
        print("❌")
        return {"status": "error", "error": str(e)}

def get_memgraph_stats() -> Dict:
    """Query Memgraph for graph statistics."""
    print(f"🔌 Connecting to Memgraph ({MEMGRAPH_HOST})...", end=" ", flush=True)
    stats = {}
    try:
        memgraph = Memgraph(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT, username=MEMGRAPH_USER, password=MEMGRAPH_PASSWORD)
        
        # 1. Total Counts
        node_count = list(memgraph.execute_and_fetch("MATCH (n) RETURN count(n) as c"))[0]['c']
        rel_count = list(memgraph.execute_and_fetch("MATCH ()-[r]->() RETURN count(r) as c"))[0]['c']
        
        stats['total_nodes'] = node_count
        stats['total_relationships'] = rel_count
        
        # 2. Node By Label
        labels_query = "MATCH (n) RETURN labels(n)[0] as label, count(*) as c ORDER BY c DESC"
        stats['node_labels'] = {row['label']: row['c'] for row in memgraph.execute_and_fetch(labels_query)}
        
        # 3. Relationship By Type
        rels_query = "MATCH ()-[r]->() RETURN type(r) as t, count(*) as c ORDER BY c DESC"
        stats['rel_types'] = {row['t']: row['c'] for row in memgraph.execute_and_fetch(rels_query)}
        
        # 4. Transcript Approximation (based on STATEMENT or distinct metadata if available)
        # Since we don't have explicit 'Transcript' nodes, we can estimate active transcripts 
        # by counting distinct speakers or dates, but 'STATEMENT' count gives a sense of volume.
        
        print("✅")
        stats['status'] = "success"
        return stats
        
    except Exception as e:
        print("❌")
        return {"status": "error", "error": str(e)}

# --------------------------------------------------------------------------
# REPORTING
# --------------------------------------------------------------------------


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/(1024*1024):.2f} MB"

def main():
    # Only console output helper
    def print_console(msg=""):
        print(msg)

    print_console("\nGathering statistics... please wait.")
    
    # Gather Data
    gcs_stats = get_gcs_stats()
    checkpoint_stats = get_checkpoint_stats()
    graph_stats = get_memgraph_stats()
    
    # ----------------------------------------------------------------------
    # GENERATE MARKDOWN REPORT
    # ----------------------------------------------------------------------
    md_lines = []
    
    md_lines.append("# 📊 Memgraph Data Ingestion Report")
    md_lines.append(f"**Date:** {os.getenv('DATE', 'Today')}")  # Could add date if needed
    md_lines.append("")
    
    # SUMMARY SECTION
    md_lines.append("## 1. Pipeline Summary")
    md_lines.append("| Stage | Metric | Value | Status |")
    md_lines.append("|:------|:-------|:------|:-------|")
    
    # GCS
    if gcs_stats['status'] == 'success':
        status = "✅ Active"
        rows = [
            f"| **Source (GCS)** | Total Transcripts | **{gcs_stats['total_transcripts']:,}** | {status} |",
            f"| | Total Size | {format_size(gcs_stats['total_size_bytes'])} | |"
        ]
    else:
         rows = [f"| **Source (GCS)** | Status | ❌ Error | {gcs_stats.get('error','')} |"]
    md_lines.extend(rows)
    
    # Checkpoint
    if checkpoint_stats['status'] == 'success':
        count = checkpoint_stats['count']
        match_icon = "✅" if count >= gcs_stats.get('total_transcripts', 0) else "⏳"
        rows = [
            f"| **Processor** | Checkpoint Count | **{count:,}** | {match_icon} (vs GCS) |"
        ]
    else:
        rows = [f"| **Processor** | Status | ❌ Error | {checkpoint_stats.get('error','')} |"]
    md_lines.extend(rows)
    
    # Memgraph
    if graph_stats['status'] == 'success':
        rows = [
            f"| **Destination** | Total Nodes | **{graph_stats['total_nodes']:,}** | ✅ Online |",
            f"| | Relationships | **{graph_stats['total_relationships']:,}** | |"
        ]
    else:
        rows = [f"| **Destination** | Status | ❌ Error | {graph_stats.get('error','')} |"]
    md_lines.extend(rows)
    md_lines.append("")

    # GRAPH DETAILS SECTION
    if graph_stats.get('status') == 'success':
        md_lines.append("## 2. Graph Database Composition")
        
        # Side-by-side tables if possible, but standard markdown stacks them
        md_lines.append("### Node Distribution")
        md_lines.append("| Node Label | Count | % Distribution |")
        md_lines.append("|:-----------|------:|:---------------|")
        
        total_nodes = graph_stats['total_nodes']
        if total_nodes > 0:
            for label, count in graph_stats['node_labels'].items():
                pct = (count / total_nodes) * 100
                bar = "█" * int(pct / 5)
                md_lines.append(f"| `{label}` | {count:,} | {pct:6.1f}% {bar} |")
        else:
            md_lines.append("| *(No nodes)* | 0 | - |")
        md_lines.append("")
        
        md_lines.append("### Relationship Distribution")
        md_lines.append("| Relationship Type | Count | % Distribution |")
        md_lines.append("|:------------------|------:|:---------------|")
        
        total_rels = graph_stats['total_relationships']
        if total_rels > 0:
            for rel, count in graph_stats['rel_types'].items():
                pct = (count / total_rels) * 100
                bar = "█" * int(pct / 5)
                md_lines.append(f"| `{rel}` | {count:,} | {pct:6.1f}% {bar} |")
        else:
            md_lines.append("| *(No relationships)* | 0 | - |")
        md_lines.append("")
        
        # HEALTH CHECK SECTION
        md_lines.append("## 3. Health & Quality Check")
        
        processed_count = checkpoint_stats.get('count', 0)
        if processed_count > 0:
            avg_stmts = graph_stats['node_labels'].get('STATEMENT', 0) / processed_count
            
            md_lines.append("> **Ingestion Ratios**")
            md_lines.append(f"> - **Nodes per Transcript:** {graph_stats['total_nodes']/processed_count:,.0f}")
            md_lines.append(f"> - **Edges per Transcript:** {graph_stats['total_relationships']/processed_count:,.0f}")
            md_lines.append(f"> - **Statements per Transcript:** {avg_stmts:,.1f}")
            md_lines.append("")
            
            if avg_stmts < 10:
                md_lines.append("### ⚠️ Critical Warning")
                md_lines.append(f"**Extremely low statement count ({avg_stmts:.1f}/file).**")
                md_lines.append("This suggests that `STATEMENT` nodes (speaker attributions) are NOT being created for most transcripts.")
                md_lines.append("- Check `v1.0-graph-ea-scripts.py` speaker extraction patterns.")
                md_lines.append("- Verify transcript format in GCS.")
            else:
                 md_lines.append("### ✅ Status: Healthy")
                 md_lines.append("Data density appears within expected ranges.")
        else:
            md_lines.append("*(Skipping health check: 0 processed transcripts)*")

    # Save to file
    output_filename = "memgraph_report.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    
    print_console(f"\n✅ Generated polished report: {output_filename}")
    
    # Print a preview to console
    print_console("-" * 60)
    print_console("PREVIEW:")
    print_console("-" * 60)
    for line in md_lines[:20]: # Show first 20 lines
        print_console(line)
    print_console("...")

if __name__ == "__main__":
    main()
