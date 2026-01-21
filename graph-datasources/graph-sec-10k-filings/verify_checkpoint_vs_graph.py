#!/usr/bin/env python3
"""
Verification script to compare checkpoint file with actual graph database contents.
Checks for discrepancies between what the checkpoint says was processed vs what's actually in Memgraph.
"""

import os
import sys
import pickle
import logging
from typing import Set, Tuple, Dict, List
from collections import defaultdict
from google.cloud import storage
from gqlalchemy import Memgraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
CHECKPOINT_GCS_PATH = os.getenv("CHECKPOINT_GCS_PATH", "checkpoints/10k_processing_checkpoint.pkl")

# Memgraph Configuration
MEMGRAPH_HOST = os.getenv("MEMGRAPH_HOST", "localhost")
MEMGRAPH_PORT = int(os.getenv("MEMGRAPH_PORT", "7687"))


def load_checkpoint_from_gcs() -> Dict:
    """Load checkpoint from GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(CHECKPOINT_GCS_PATH)
        
        if not blob.exists():
            logger.warning(f"⚠️  Checkpoint does not exist at gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}")
            return {'processed': {}, 'last_updated': None}
        
        checkpoint_bytes = blob.download_as_bytes()
        checkpoint_data = pickle.loads(checkpoint_bytes)
        logger.info(f"✅ Loaded checkpoint from GCS ({len(checkpoint_data.get('processed', {}))} entries)")
        return checkpoint_data
        
    except Exception as e:
        logger.error(f"❌ Error loading checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'processed': {}, 'last_updated': None}


def get_filings_from_checkpoint(checkpoint_data: Dict) -> Set[Tuple[str, str]]:
    """Extract (symbol, year) pairs from checkpoint."""
    filings = set()
    processed = checkpoint_data.get('processed', {})
    
    for key, entry in processed.items():
        symbol = entry.get('symbol', '').upper()
        year = entry.get('year', '')
        if symbol and year:
            filings.add((symbol, year))
    
    return filings


def get_filings_from_graph(db: Memgraph) -> Dict[Tuple[str, str], Dict]:
    """
    Query Memgraph for all unique (ticker, filing_year) pairs from SECTION nodes.
    Returns dict mapping (ticker, year) -> metadata dict with counts.
    """
    try:
        # First, try to get all SECTION nodes to understand the structure
        sample_query = """
        MATCH (s:SECTION)
        RETURN s.ticker AS ticker, s.filing_year AS filing_year, s.section_id AS section_id
        LIMIT 5
        """
        sample_results = list(db.execute_and_fetch(sample_query))
        if sample_results:
            logger.info(f"   Sample SECTION node: ticker={sample_results[0].get('ticker')}, filing_year={sample_results[0].get('filing_year')}")
        
        # Query for all unique ticker/year combinations with counts
        query = """
        MATCH (s:SECTION)
        WHERE s.ticker IS NOT NULL AND s.filing_year IS NOT NULL
        WITH s.ticker AS ticker, s.filing_year AS filing_year
        RETURN ticker, 
               filing_year,
               count(*) AS section_count
        ORDER BY ticker, filing_year
        """
        
        results = db.execute_and_fetch(query)
        
        filings = {}
        for record in results:
            ticker = record['ticker'].upper() if record['ticker'] else None
            filing_year = str(record['filing_year']) if record['filing_year'] else None
            
            if ticker and filing_year:
                key = (ticker, filing_year)
                # Get section IDs for this filing
                section_query = """
                MATCH (s:SECTION {ticker: $ticker, filing_year: $year})
                RETURN collect(DISTINCT s.section_id) AS section_ids
                """
                section_result = list(db.execute_and_fetch(section_query, {'ticker': ticker, 'year': filing_year}))
                section_ids = section_result[0]['section_ids'] if section_result else []
                
                filings[key] = {
                    'section_count': record['section_count'],
                    'section_ids': section_ids
                }
        
        logger.info(f"✅ Found {len(filings)} unique filing(s) in graph database")
        return filings
        
    except Exception as e:
        logger.error(f"❌ Error querying graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fallback: try a simpler query without aggregation
        try:
            logger.info("   Trying simpler query without aggregation...")
            simple_query = """
            MATCH (s:SECTION)
            WHERE s.ticker IS NOT NULL AND s.filing_year IS NOT NULL
            RETURN DISTINCT s.ticker AS ticker, s.filing_year AS filing_year
            """
            results = db.execute_and_fetch(simple_query)
            filings = {}
            for record in results:
                ticker = record['ticker'].upper() if record['ticker'] else None
                filing_year = str(record['filing_year']) if record['filing_year'] else None
                if ticker and filing_year:
                    filings[(ticker, filing_year)] = {'section_count': 0, 'section_ids': []}
            logger.info(f"✅ Found {len(filings)} unique filing(s) using simple query")
            return filings
        except Exception as e2:
            logger.error(f"❌ Fallback query also failed: {e2}")
            return {}


def get_graph_statistics(db: Memgraph) -> Dict:
    """Get overall graph statistics."""
    stats = {}
    
    try:
        # Count total nodes
        query_nodes = "MATCH (n) RETURN count(n) AS node_count"
        result = db.execute_and_fetch(query_nodes)
        stats['total_nodes'] = next(result)['node_count']
        
        # Count total relationships
        query_rels = "MATCH ()-[r]->() RETURN count(r) AS rel_count"
        result = db.execute_and_fetch(query_rels)
        stats['total_relationships'] = next(result)['rel_count']
        
        # Count SECTION nodes
        query_sections = "MATCH (s:SECTION) RETURN count(s) AS section_count"
        result = db.execute_and_fetch(query_sections)
        stats['section_nodes'] = next(result)['section_count']
        
        # Count entity nodes by type
        query_entity_types = """
        MATCH (n)
        WHERE NOT n:SECTION
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
        LIMIT 20
        """
        result = db.execute_and_fetch(query_entity_types)
        stats['entity_types'] = {record['label']: record['count'] for record in result}
        
    except Exception as e:
        logger.warning(f"⚠️  Error getting graph statistics: {e}")
    
    return stats


def compare_checkpoint_vs_graph(checkpoint_filings: Set[Tuple[str, str]], 
                                 graph_filings: Dict[Tuple[str, str], Dict]) -> Dict:
    """Compare checkpoint and graph filings, return discrepancies."""
    checkpoint_set = set(checkpoint_filings)
    graph_set = set(graph_filings.keys())
    
    # Filings in checkpoint but not in graph
    in_checkpoint_not_graph = checkpoint_set - graph_set
    
    # Filings in graph but not in checkpoint
    in_graph_not_checkpoint = graph_set - checkpoint_set
    
    # Filings in both
    in_both = checkpoint_set & graph_set
    
    return {
        'in_checkpoint_not_graph': in_checkpoint_not_graph,
        'in_graph_not_checkpoint': in_graph_not_checkpoint,
        'in_both': in_both,
        'checkpoint_count': len(checkpoint_set),
        'graph_count': len(graph_set),
        'both_count': len(in_both)
    }


def main():
    """Main verification function."""
    print("="*80)
    print("CHECKPOINT vs GRAPH DATABASE VERIFICATION")
    print("="*80)
    print(f"📦 GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 Checkpoint Path: {CHECKPOINT_GCS_PATH}")
    print(f"🗄️  Memgraph: {MEMGRAPH_HOST}:{MEMGRAPH_PORT}")
    print()
    
    # Load checkpoint
    logger.info("Loading checkpoint from GCS...")
    checkpoint_data = load_checkpoint_from_gcs()
    checkpoint_filings = get_filings_from_checkpoint(checkpoint_data)
    
    print(f"\n📋 CHECKPOINT DATA:")
    print(f"   Total filings in checkpoint: {len(checkpoint_filings)}")
    if checkpoint_filings:
        print(f"   Filings:")
        for symbol, year in sorted(checkpoint_filings):
            entry = checkpoint_data.get('processed', {}).get(f"{symbol}_{year}", {})
            status = entry.get('status', 'unknown')
            processed_at = entry.get('processed_at', 'unknown')
            print(f"      {symbol}-{year}: {status} (processed: {processed_at})")
    print()
    
    # Connect to Memgraph
    logger.info(f"Connecting to Memgraph at {MEMGRAPH_HOST}:{MEMGRAPH_PORT}...")
    try:
        db = Memgraph(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT)
        # Test connection
        db.execute("RETURN 1")
        logger.info("✅ Connected to Memgraph")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Memgraph: {e}")
        print("\n⚠️  Cannot verify graph database - connection failed")
        print("   Please ensure Memgraph is running and accessible")
        return 1
    
    # Get graph statistics
    logger.info("Getting graph statistics...")
    stats = get_graph_statistics(db)
    
    print(f"\n📊 GRAPH DATABASE STATISTICS:")
    print(f"   Total nodes: {stats.get('total_nodes', 'N/A'):,}")
    print(f"   Total relationships: {stats.get('total_relationships', 'N/A'):,}")
    print(f"   SECTION nodes: {stats.get('section_nodes', 'N/A'):,}")
    if stats.get('entity_types'):
        print(f"   Top entity types:")
        for label, count in list(stats['entity_types'].items())[:10]:
            print(f"      {label}: {count:,}")
    print()
    
    # Get filings from graph
    logger.info("Querying graph for processed filings...")
    graph_filings = get_filings_from_graph(db)
    
    print(f"\n📋 GRAPH DATABASE DATA:")
    print(f"   Total filings in graph: {len(graph_filings)}")
    if graph_filings:
        print(f"   Filings (with section counts):")
        for (symbol, year), metadata in sorted(graph_filings.items()):
            section_count = metadata.get('section_count', 0)
            print(f"      {symbol}-{year}: {section_count} section(s)")
    print()
    
    # Compare
    logger.info("Comparing checkpoint vs graph...")
    comparison = compare_checkpoint_vs_graph(checkpoint_filings, graph_filings)
    
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    print(f"📊 SUMMARY:")
    print(f"   Checkpoint filings: {comparison['checkpoint_count']}")
    print(f"   Graph filings: {comparison['graph_count']}")
    print(f"   In both: {comparison['both_count']}")
    print()
    
    # Filings in checkpoint but not in graph
    if comparison['in_checkpoint_not_graph']:
        print(f"⚠️  FILINGS IN CHECKPOINT BUT NOT IN GRAPH ({len(comparison['in_checkpoint_not_graph'])}):")
        for symbol, year in sorted(comparison['in_checkpoint_not_graph']):
            print(f"      {symbol}-{year}")
        print()
    else:
        print("✅ All checkpoint filings are present in graph\n")
    
    # Filings in graph but not in checkpoint
    if comparison['in_graph_not_checkpoint']:
        print(f"❌ FILINGS IN GRAPH BUT NOT IN CHECKPOINT ({len(comparison['in_graph_not_checkpoint'])}):")
        print("   These filings were processed but not recorded in checkpoint!")
        for symbol, year in sorted(comparison['in_graph_not_checkpoint']):
            metadata = graph_filings[(symbol, year)]
            section_count = metadata.get('section_count', 0)
            section_ids = metadata.get('section_ids', [])
            print(f"      {symbol}-{year}: {section_count} section(s) - sections: {', '.join(map(str, section_ids[:5]))}")
            if len(section_ids) > 5:
                print(f"         ... and {len(section_ids) - 5} more")
        print()
    else:
        print("✅ All graph filings are recorded in checkpoint\n")
    
    # Filings in both
    if comparison['in_both']:
        print(f"✅ FILINGS IN BOTH ({len(comparison['in_both'])}):")
        for symbol, year in sorted(list(comparison['in_both'])[:10]):
            entry = checkpoint_data.get('processed', {}).get(f"{symbol}_{year}", {})
            status = entry.get('status', 'unknown')
            metadata = graph_filings.get((symbol, year), {})
            section_count = metadata.get('section_count', 0)
            print(f"      {symbol}-{year}: checkpoint={status}, graph={section_count} sections")
        if len(comparison['in_both']) > 10:
            print(f"      ... and {len(comparison['in_both']) - 10} more")
        print()
    
    # Final assessment
    print(f"{'='*80}")
    print("ASSESSMENT")
    print(f"{'='*80}\n")
    
    if comparison['in_graph_not_checkpoint']:
        print("❌ DISCREPANCY DETECTED!")
        print(f"   {len(comparison['in_graph_not_checkpoint'])} filing(s) exist in graph but are missing from checkpoint.")
        print("   This suggests the checkpoint was not properly updated during processing.")
        print("\n   RECOMMENDATION:")
        print("   1. Review the batch processing logs to see what happened")
        print("   2. Consider updating the checkpoint file to include missing filings")
        print("   3. Verify that checkpoint saving is working correctly in batch_process_10k.py")
        return 1
    elif comparison['in_checkpoint_not_graph']:
        print("⚠️  WARNING:")
        print(f"   {len(comparison['in_checkpoint_not_graph'])} filing(s) in checkpoint but not in graph.")
        print("   These may have failed during processing or were cleared from the graph.")
    else:
        print("✅ CHECKPOINT AND GRAPH ARE IN SYNC")
        print("   All checkpoint entries match graph database contents.")
    
    print(f"\n{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

