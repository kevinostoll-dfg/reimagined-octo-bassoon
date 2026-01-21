#!/usr/bin/env python3
"""
Batch Processing Script for SEC Form 4 Filings
Discovers all Form 4 filings in GCS bucket and processes them using process_document.py
Runs up to 20 processes in parallel and monitors their progress.
"""

import os
import sys
import time
import logging
import pickle
from typing import List, Dict, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock, Thread
import urllib3
from google.cloud import storage
from dotenv import load_dotenv
from datetime import datetime, timezone

# Import core processing functions from process_document.py
# We need to ensure the module can be imported properly
sys.path.insert(0, os.path.dirname(__file__))
import process_document

# Load environment variables
load_dotenv('env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress urllib3 connection pool warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "form-4-filings")

# Script configuration
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_PROCESSES", "10"))

# Checkpoint configuration
CHECKPOINT_GCS_PATH = os.getenv("CHECKPOINT_GCS_PATH", "checkpoints/f4_processing_checkpoint.pkl")
CHECKPOINT_ENABLED = os.getenv("CHECKPOINT_ENABLED", "true").lower() == "true"

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Checkpoint lock for thread-safe updates
checkpoint_lock = Lock()


def load_checkpoint() -> Dict:
    """
    Load checkpoint from GCS PKL file.
    Returns dict with processed filings information.
    """
    if not CHECKPOINT_ENABLED:
        return {}
    
    try:
        blob = bucket.blob(CHECKPOINT_GCS_PATH)
        if blob.exists():
            # Download to memory
            checkpoint_bytes = blob.download_as_bytes()
            checkpoint_data = pickle.loads(checkpoint_bytes)
            logger.info(f"✅ Loaded checkpoint from gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}")
            logger.info(f"   Found {len(checkpoint_data.get('processed', {}))} processed filing(s)")
            return checkpoint_data
        else:
            logger.info(f"ℹ️  No checkpoint found, starting fresh")
            return {'processed': {}, 'last_updated': None}
    except Exception as e:
        logger.warning(f"⚠️  Error loading checkpoint: {e}, starting fresh")
        return {'processed': {}, 'last_updated': None}


def save_checkpoint(checkpoint_data: Dict):
    """
    Save checkpoint to GCS PKL file.
    Non-blocking operation - runs in background thread.
    """
    if not CHECKPOINT_ENABLED:
        return
    
    try:
        # Update timestamp
        checkpoint_data['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Serialize to pickle
        checkpoint_bytes = pickle.dumps(checkpoint_data)
        
        # Upload to GCS (this is the potentially slow operation)
        blob = bucket.blob(CHECKPOINT_GCS_PATH)
        blob.upload_from_string(checkpoint_bytes, content_type='application/octet-stream')
        
        # Skip verification to speed up (optional - can re-enable if needed)
        # blob.reload()
        # if blob.size != len(checkpoint_bytes):
        #     raise ValueError(
        #         f"Checkpoint size mismatch: uploaded {len(checkpoint_bytes)} bytes, "
        #         f"but blob has {blob.size} bytes"
        #     )
        
        processed_count = len(checkpoint_data.get('processed', {}))
        logger.debug(f"💾 Checkpoint saved to gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH} ({processed_count} filing(s), {len(checkpoint_bytes)} bytes)")
        
        # Mark save as complete
        with checkpoint_lock:
            checkpoint_data['save_in_progress'] = False
            
    except Exception as e:
        logger.error(f"❌ Error saving checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Mark save as complete even on error so we can retry
        with checkpoint_lock:
            checkpoint_data['save_in_progress'] = False


def update_checkpoint_entry(checkpoint_data: Dict, file_path: str, status: str, 
                            duration: float = None, error: str = None, relationships_count: int = 0,
                            save_immediately: bool = False):
    """
    Update checkpoint entry for a specific filing.
    Thread-safe operation. Checkpoint saves are non-blocking.
    
    Args:
        save_immediately: If True, save to GCS immediately. If False, batch saves.
    """
    if not CHECKPOINT_ENABLED:
        return
    
    # Update the checkpoint data (quick operation, just updating dict)
    with checkpoint_lock:
        if 'processed' not in checkpoint_data:
            checkpoint_data['processed'] = {}
        
        # Use file path as key (normalized)
        key = file_path
        
        checkpoint_data['processed'][key] = {
            'file_path': file_path,
            'status': status,  # 'completed', 'failed', 'error'
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'duration': duration,
            'relationships_count': relationships_count,
            'error': error[:200] if error else None  # Truncate long errors
        }
        
        # Track when checkpoint was last saved
        if 'last_saved_at' not in checkpoint_data:
            checkpoint_data['last_saved_at'] = time.time()
            checkpoint_data['last_saved_count'] = 0
            checkpoint_data['save_in_progress'] = False
        
        # Check if we should trigger a save (but don't block on it)
        should_save = save_immediately
        if not should_save:
            time_since_save = time.time() - checkpoint_data.get('last_saved_at', 0)
            current_count = len(checkpoint_data.get('processed', {}))
            filings_since_save = current_count - checkpoint_data.get('last_saved_count', 0)
            should_save = (time_since_save > 30 or filings_since_save >= 50) and not checkpoint_data.get('save_in_progress', False)
        
        if should_save:
            # Mark that save is in progress
            checkpoint_data['save_in_progress'] = True
            # Start save in background thread (non-blocking)
            save_thread = Thread(target=save_checkpoint, args=(checkpoint_data,), daemon=True)
            save_thread.start()
            # Update tracking (don't wait for save to complete)
            checkpoint_data['last_saved_at'] = time.time()
            checkpoint_data['last_saved_count'] = len(checkpoint_data.get('processed', {}))


def is_filing_processed(checkpoint_data: Dict, file_path: str) -> bool:
    """
    Check if a filing has already been successfully processed.
    """
    if not CHECKPOINT_ENABLED:
        return False
    
    entry = checkpoint_data.get('processed', {}).get(file_path)
    
    if entry and entry.get('status') == 'completed':
        return True
    
    return False


def filter_processed_filings(filings: List[str], checkpoint_data: Dict) -> tuple[List[str], List[str]]:
    """
    Filter out already processed filings.
    Returns (pending_filings, already_processed_filings)
    """
    if not CHECKPOINT_ENABLED:
        return filings, []
    
    pending = []
    processed = []
    
    for file_path in filings:
        if is_filing_processed(checkpoint_data, file_path):
            processed.append(file_path)
        else:
            pending.append(file_path)
    
    return pending, processed


def discover_f4_filings() -> List[str]:
    """
    Discover all Form 4 filings in GCS bucket.
    Returns list of file paths.
    
    Structure: {GCS_BASE_PATH}/{SYMBOL}/{YEAR}/{SYMBOL}_{DATE}_{HASH}.json
    """
    logger.info(f"🔍 Discovering Form 4 filings in GCS bucket: {GCS_BUCKET_NAME}")
    logger.info(f"   Base path: {GCS_BASE_PATH}/")
    
    prefix = f"{GCS_BASE_PATH}/"
    filings = []
    
    try:
        # List all blobs with the prefix
        all_blobs = list(bucket.list_blobs(prefix=prefix))
        
        # Filter for JSON files only
        for blob in all_blobs:
            if blob.name.endswith('.json'):
                filings.append(blob.name)
        
        # Sort by path
        filings.sort()
        
        logger.info(f"✅ Found {len(filings)} Form 4 filing(s)")
        return filings
        
    except Exception as e:
        logger.error(f"❌ Error discovering filings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def process_with_semaphore(file_path: str, semaphore: Semaphore, 
                            checkpoint_data: Dict, started_tasks: Dict, 
                            started_tasks_lock: Lock) -> Dict:
    """
    Process a filing with semaphore-based concurrency control.
    Updates checkpoint after processing.
    """
    task_id = id(file_path)
    with semaphore:
        # Log that we're starting
        logger.info(f"🔄 Starting: {file_path.split('/')[-1]}")
        sys.stdout.flush()
        with started_tasks_lock:
            started_tasks[task_id] = (file_path, time.time())
        
        try:
            # Process document (output disabled for batch processing)
            file_name = file_path.split('/')[-1]
            logger.debug(f"  📄 {file_name}: Fetching from GCS...")
            result = process_document.process_single_document(file_path, output_enabled=False)
            
            # Update checkpoint (batched saves to avoid blocking)
            if CHECKPOINT_ENABLED:
                status = result.get('status', 'error')
                duration = result.get('duration')
                error = result.get('error')
                relationships_count = result.get('relationships_count', 0)
                # Don't save immediately - batch saves every 50 filings or 30 seconds
                update_checkpoint_entry(checkpoint_data, file_path, status, duration, error, relationships_count, save_immediately=False)
            
            return result
        finally:
            # Remove from started tasks when done
            with started_tasks_lock:
                if task_id in started_tasks:
                    del started_tasks[task_id]


def main():
    """
    Main execution: discover filings and process them in parallel.
    """
    print("="*80)
    print("BATCH PROCESSING: SEC FORM 4 FILINGS")
    print("="*80)
    print(f"📦 GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 Base Path: {GCS_BASE_PATH}/")
    print(f"⚙️  Max Concurrent: {MAX_CONCURRENT}")
    if CHECKPOINT_ENABLED:
        print(f"💾 Checkpoint: gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}")
    else:
        print(f"💾 Checkpoint: Disabled")
    print()
    
    # Initialize spaCy model (shared across all workers)
    logger.info("Initializing spaCy model...")
    try:
        process_document.initialize_spacy_model()
        logger.info("✅ SpaCy model initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize spaCy model: {e}")
        sys.exit(1)
    
    # Initialize Memgraph connection (will be reused)
    logger.info("Initializing Memgraph connection...")
    try:
        process_document.connect_to_memgraph()
        logger.info("✅ Memgraph connection initialized")
    except Exception as e:
        logger.warning(f"⚠️  Could not initialize Memgraph connection: {e}")
        logger.warning("   Processing will continue but data won't be persisted")
    
    # Load checkpoint
    checkpoint_data = load_checkpoint()
    
    # Discover all filings
    filings = discover_f4_filings()
    
    if not filings:
        logger.warning("⚠️  No filings found. Exiting.")
        return
    
    # Filter out already processed filings
    pending_filings, processed_filings = filter_processed_filings(filings, checkpoint_data)
    
    # Display discovered filings
    print(f"\n📋 DISCOVERED FILINGS ({len(filings)} total):")
    if processed_filings:
        print(f"   ✅ Already processed ({len(processed_filings)}):")
        for file_path in processed_filings[:10]:  # Show first 10
            print(f"      {file_path}")
        if len(processed_filings) > 10:
            print(f"      ... and {len(processed_filings) - 10} more")
    
    if pending_filings:
        print(f"   ⏳ Pending processing ({len(pending_filings)}):")
        for file_path in pending_filings[:10]:  # Show first 10
            print(f"      {file_path}")
        if len(pending_filings) > 10:
            print(f"      ... and {len(pending_filings) - 10} more")
    print()
    
    if not pending_filings:
        logger.info("✅ All filings have already been processed. Exiting.")
        return
    
    # Ask for confirmation if processing many filings
    if len(pending_filings) > 10:
        response = input(f"⚠️  About to process {len(pending_filings)} filings. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Cancelled by user.")
            return
    
    # Process filings with concurrency control
    print(f"\n🚀 PROCESSING {len(pending_filings)} FILING(S) (max {MAX_CONCURRENT} concurrent)...")
    print("="*80)
    print()
    sys.stdout.flush()  # Ensure output is flushed
    
    start_time = time.time()
    results = []
    running_tasks = {}  # Track running tasks for progress display
    started_tasks = {}  # Track tasks that have started processing
    started_tasks_lock = Lock()
    completed_count = [0]  # Use list for shared mutable state
    
    # Status update thread to show periodic progress
    def status_updater():
        """Periodically print status of running tasks"""
        while True:
            time.sleep(5)  # Update every 5 seconds
            with started_tasks_lock:
                active_count = len(started_tasks)
                # Show which tasks are running and how long they've been running
                if active_count > 0:
                    active_tasks_info = []
                    for task_id, (file_path, start_time_task) in list(started_tasks.items())[:5]:  # Show first 5
                        duration = time.time() - start_time_task
                        file_name = file_path.split('/')[-1]
                        active_tasks_info.append(f"{file_name} ({duration:.0f}s)")
                    active_info = ", ".join(active_tasks_info)
                    if active_count > 5:
                        active_info += f" ... and {active_count - 5} more"
            completed = completed_count[0]
            if active_count > 0 or completed < len(pending_filings):
                elapsed = time.time() - start_time
                queued = len(pending_filings) - completed - active_count
                if active_count > 0 and active_count <= 10:
                    # Show active task details when there are few active tasks
                    logger.info(
                        f"📊 Status: {completed}/{len(pending_filings)} completed, "
                        f"{active_count} active, {queued} queued "
                        f"({elapsed:.0f}s elapsed) | Active: {active_info}"
                    )
                else:
                    logger.info(
                        f"📊 Status: {completed}/{len(pending_filings)} completed, "
                        f"{active_count} active, {queued} queued "
                        f"({elapsed:.0f}s elapsed)"
                    )
                sys.stdout.flush()
            if completed >= len(pending_filings):
                break
    
    # Use ThreadPoolExecutor with semaphore for concurrency control
    semaphore = Semaphore(MAX_CONCURRENT)
    
    # Start status updater thread
    status_thread = Thread(target=status_updater, daemon=True)
    status_thread.start()
    
    logger.info(f"📤 Submitting {len(pending_filings)} tasks to thread pool...")
    sys.stdout.flush()
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        # Submit all tasks (only pending filings)
        future_to_filing = {
            executor.submit(process_with_semaphore, file_path, semaphore, checkpoint_data, started_tasks, started_tasks_lock): file_path
            for file_path in pending_filings
        }
        
        logger.info(f"✅ All {len(future_to_filing)} tasks submitted. Processing started...")
        sys.stdout.flush()
        
        # Track running tasks
        for future, file_path in future_to_filing.items():
            running_tasks[future] = (file_path, time.time())
        
        # Process completed tasks
        for future in as_completed(future_to_filing):
            file_path = future_to_filing[future]
            start_task_time = running_tasks.get(future, (None, time.time()))[1]
            task_duration = time.time() - start_task_time
            
            try:
                result = future.result()
                results.append(result)
                completed_count[0] += 1
                completed = completed_count[0]
                
                # Remove from running tasks
                if future in running_tasks:
                    del running_tasks[future]
                
                # Progress update
                progress = (completed / len(pending_filings)) * 100
                remaining = len(pending_filings) - completed
                with started_tasks_lock:
                    running_count = len(started_tasks)
                
                status_emoji = "✅" if result.get('status') == 'completed' else "❌"
                relationships_count = result.get('relationships_count', 0)
                error_msg = result.get('error', '')
                
                if result.get('status') != 'completed':
                    # Log error details for failed filings
                    error_preview = error_msg[:200] if error_msg else 'Unknown error (no error message in result)'
                    logger.error(
                        f"{status_emoji} [{completed}/{len(pending_filings)}] {file_path.split('/')[-1]} "
                        f"({task_duration:.1f}s, {relationships_count} rels) | ERROR: {error_preview}"
                    )
                    # Also log the full result for debugging
                    logger.debug(f"Full result for {file_path}: {result}")
                else:
                    logger.info(
                        f"{status_emoji} [{completed}/{len(pending_filings)}] {file_path.split('/')[-1]} "
                        f"({task_duration:.1f}s, {relationships_count} rels) | Active: {running_count} | Remaining: {remaining}"
                    )
                sys.stdout.flush()
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                error_msg = str(e)
                logger.error(f"❌ Exception processing {file_path}: {error_msg}")
                logger.debug(f"Full traceback:\n{error_trace}")
                results.append({
                    'file_path': file_path,
                    'status': 'error',
                    'error': error_msg,
                    'duration': task_duration,
                    'relationships_count': 0
                })
                completed_count[0] += 1
                completed = completed_count[0]
                if future in running_tasks:
                    del running_tasks[future]
                sys.stdout.flush()
    
    total_time = time.time() - start_time
    
    # Summary statistics
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    status_counts = defaultdict(int)
    total_duration = 0
    total_relationships = 0
    
    for result in results:
        status = result.get('status', 'unknown')
        status_counts[status] += 1
        if result.get('duration'):
            total_duration += result['duration']
        if result.get('relationships_count'):
            total_relationships += result['relationships_count']
    
    print(f"\n📊 RESULTS:")
    print(f"   Total filings: {len(results)}")
    print(f"   ✅ Completed: {status_counts.get('completed', 0)}")
    print(f"   ❌ Failed: {status_counts.get('failed', 0)}")
    print(f"   ⚠️  Errors: {status_counts.get('error', 0)}")
    print(f"   🔗 Total relationships: {total_relationships}")
    print(f"   ⏱️  Total processing time: {total_time:.1f}s")
    if results:
        print(f"   ⏱️  Average time per filing: {total_duration/len(results):.1f}s")
        if status_counts.get('completed', 0) > 0:
            avg_rels = total_relationships / status_counts.get('completed', 1)
            print(f"   🔗 Average relationships per filing: {avg_rels:.1f}")
    
    # Show failed filings
    failed = [r for r in results if r.get('status') in ['failed', 'error']]
    if failed:
        print(f"\n❌ FAILED FILINGS ({len(failed)}):")
        for result in failed[:20]:  # Show first 20
            file_path = result.get('file_path', '?')
            error = result.get('error', 'Unknown error')
            print(f"   {file_path.split('/')[-1]}: {error[:100]}")
        if len(failed) > 20:
            print(f"   ... and {len(failed) - 20} more")
    
    # Show successful filings summary
    successful = [r for r in results if r.get('status') == 'completed']
    if successful:
        print(f"\n✅ SUCCESSFUL FILINGS ({len(successful)}):")
        print(f"   Total relationships created: {total_relationships}")
        print(f"   Average relationships per filing: {total_relationships/len(successful):.1f}")
    
    print(f"\n{'='*80}\n")
    
    # Final checkpoint save (force immediate save)
    if CHECKPOINT_ENABLED:
        with checkpoint_lock:
            save_checkpoint(checkpoint_data)
        logger.info(f"💾 Final checkpoint saved ({len(checkpoint_data.get('processed', {}))} filing(s))")
    
    # Exit with error code if any failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
