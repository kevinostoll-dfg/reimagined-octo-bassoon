#!/usr/bin/env python3
"""
Fix checkpoint file by adding missing filings that exist in the graph database.
This script will:
1. Load the current checkpoint
2. Query the graph for all processed filings
3. Add missing filings to the checkpoint with 'completed' status
4. Save the updated checkpoint back to GCS
"""

import os
import sys
import pickle
import logging
from typing import Set, Tuple, Dict
from datetime import datetime, timezone
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
            logger.warning(f"⚠️  Checkpoint does not exist, creating new one")
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


def save_checkpoint_to_gcs(checkpoint_data: Dict):
    """Save checkpoint to GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(CHECKPOINT_GCS_PATH)
        
        # Update timestamp
        checkpoint_data['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Serialize to pickle
        checkpoint_bytes = pickle.dumps(checkpoint_data)
        
        # Upload to GCS
        blob.upload_from_string(checkpoint_bytes, content_type='application/octet-stream')
        
        # Verify it was saved correctly
        blob.reload()
        if blob.size != len(checkpoint_bytes):
            raise ValueError(
                f"Checkpoint size mismatch: uploaded {len(checkpoint_bytes)} bytes, "
                f"but blob has {blob.size} bytes"
            )
        
        processed_count = len(checkpoint_data.get('processed', {}))
        logger.info(f"✅ Checkpoint saved to gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}")
        logger.info(f"   {processed_count} filing(s), {len(checkpoint_bytes)} bytes")
        
    except Exception as e:
        logger.error(f"❌ Error saving checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def get_filings_from_graph(db: Memgraph) -> Dict[Tuple[str, str], Dict]:
    """
    Query Memgraph for all unique (ticker, filing_year) pairs from SECTION nodes.
    Returns dict mapping (ticker, year) -> metadata dict with counts.
    """
    try:
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
                filings[key] = {
                    'section_count': record['section_count']
                }
        
        logger.info(f"✅ Found {len(filings)} unique filing(s) in graph database")
        return filings
        
    except Exception as e:
        logger.error(f"❌ Error querying graph: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def update_checkpoint_with_graph_filings(checkpoint_data: Dict, graph_filings: Dict[Tuple[str, str], Dict]) -> Dict:
    """
    Update checkpoint with missing filings from graph.
    Returns updated checkpoint data and summary of changes.
    """
    if 'processed' not in checkpoint_data:
        checkpoint_data['processed'] = {}
    
    added = []
    updated = []
    existing = []
    
    for (ticker, year), metadata in graph_filings.items():
        key = f"{ticker}_{year}"
        section_count = metadata.get('section_count', 0)
        
        if key in checkpoint_data['processed']:
            existing_entry = checkpoint_data['processed'][key]
            existing_status = existing_entry.get('status', 'unknown')
            
            # If existing entry is not 'completed', update it
            if existing_status != 'completed':
                checkpoint_data['processed'][key] = {
                    'symbol': ticker,
                    'year': year,
                    'status': 'completed',
                    'processed_at': datetime.now(timezone.utc).isoformat(),
                    'duration': existing_entry.get('duration'),
                    'error': None,
                    'section_count': section_count
                }
                updated.append((ticker, year))
            else:
                existing.append((ticker, year))
        else:
            # Add new entry
            checkpoint_data['processed'][key] = {
                'symbol': ticker,
                'year': year,
                'status': 'completed',
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'duration': None,
                'error': None,
                'section_count': section_count
            }
            added.append((ticker, year))
    
    return {
        'checkpoint_data': checkpoint_data,
        'added': added,
        'updated': updated,
        'existing': existing
    }


def main():
    """Main function to fix checkpoint."""
    print("="*80)
    print("FIX CHECKPOINT FROM GRAPH DATABASE")
    print("="*80)
    print(f"📦 GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 Checkpoint Path: {CHECKPOINT_GCS_PATH}")
    print(f"🗄️  Memgraph: {MEMGRAPH_HOST}:{MEMGRAPH_PORT}")
    print()
    
    # Load current checkpoint
    logger.info("Loading current checkpoint from GCS...")
    checkpoint_data = load_checkpoint_from_gcs()
    initial_count = len(checkpoint_data.get('processed', {}))
    print(f"   Current checkpoint has {initial_count} filing(s)\n")
    
    # Connect to Memgraph
    logger.info(f"Connecting to Memgraph at {MEMGRAPH_HOST}:{MEMGRAPH_PORT}...")
    try:
        db = Memgraph(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT)
        db.execute("RETURN 1")
        logger.info("✅ Connected to Memgraph")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Memgraph: {e}")
        print("\n⚠️  Cannot fix checkpoint - connection failed")
        print("   Please ensure Memgraph is running and accessible")
        return 1
    
    # Get filings from graph
    logger.info("Querying graph for processed filings...")
    graph_filings = get_filings_from_graph(db)
    print(f"   Found {len(graph_filings)} filing(s) in graph database\n")
    
    if not graph_filings:
        print("⚠️  No filings found in graph database. Nothing to update.")
        return 0
    
    # Update checkpoint
    logger.info("Updating checkpoint with graph filings...")
    result = update_checkpoint_with_graph_filings(checkpoint_data, graph_filings)
    updated_checkpoint = result['checkpoint_data']
    added = result['added']
    updated = result['updated']
    existing = result['existing']
    
    final_count = len(updated_checkpoint.get('processed', {}))
    
    print(f"\n📊 UPDATE SUMMARY:")
    print(f"   Initial checkpoint entries: {initial_count}")
    print(f"   Final checkpoint entries: {final_count}")
    print(f"   Added: {len(added)}")
    print(f"   Updated: {len(updated)}")
    print(f"   Existing (unchanged): {len(existing)}")
    print()
    
    if added:
        print(f"✅ ADDED TO CHECKPOINT ({len(added)}):")
        for ticker, year in sorted(added):
            metadata = graph_filings.get((ticker, year), {})
            section_count = metadata.get('section_count', 0)
            print(f"      {ticker}-{year} ({section_count} sections)")
        print()
    
    if updated:
        print(f"🔄 UPDATED IN CHECKPOINT ({len(updated)}):")
        for ticker, year in sorted(updated):
            print(f"      {ticker}-{year}")
        print()
    
    if not added and not updated:
        print("ℹ️  No changes needed - checkpoint is already up to date\n")
        return 0
    
    # Ask for confirmation
    print("="*80)
    response = input(f"⚠️  About to update checkpoint file in GCS. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled by user.")
        return 0
    
    # Save updated checkpoint
    logger.info("Saving updated checkpoint to GCS...")
    try:
        save_checkpoint_to_gcs(updated_checkpoint)
        print(f"\n✅ CHECKPOINT UPDATED SUCCESSFULLY!")
        print(f"   Added {len(added)} missing filing(s)")
        if updated:
            print(f"   Updated {len(updated)} existing filing(s)")
        print(f"   Total filings in checkpoint: {final_count}")
        print(f"\n{'='*80}\n")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

