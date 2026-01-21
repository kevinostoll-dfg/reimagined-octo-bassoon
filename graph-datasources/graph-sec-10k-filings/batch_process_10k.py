#!/usr/bin/env python3
"""
Batch Processing Script for SEC 10-K Filings
Discovers all 10-K filings in GCS bucket and processes them using v3.0-prototype.py
Runs up to 10 processes in parallel and monitors their progress.
"""

import os
import sys
import subprocess
import time
import re
import logging
import pickle
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from threading import Semaphore, Lock, Thread
from google.cloud import storage
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# GCS Configuration (matches v3.0-prototype.py)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "sec-10k")

# Script configuration
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "v3.0-prototype.py")
_CPU_COUNT = os.cpu_count() or 1
_DEFAULT_MAX_CONCURRENT = max(_CPU_COUNT - 1, 1)
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_PROCESSES", str(_DEFAULT_MAX_CONCURRENT)))
PYTHON_EXECUTABLE = os.getenv("PYTHON_EXECUTABLE", sys.executable)
PROCESS_TIMEOUT = int(os.getenv("PROCESS_TIMEOUT_SECONDS", "3600"))  # Default: 1 hour timeout per process
HEARTBEAT_INTERVAL = int(os.getenv("PROGRESS_HEARTBEAT_SECONDS", "30"))

# Checkpoint configuration
CHECKPOINT_GCS_PATH = os.getenv("CHECKPOINT_GCS_PATH", "checkpoints/10k_processing_checkpoint.pkl")
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
    Thread-safe operation.
    """
    if not CHECKPOINT_ENABLED:
        return
    
    with checkpoint_lock:
        try:
            # Update timestamp
            checkpoint_data['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            # Serialize to pickle
            checkpoint_bytes = pickle.dumps(checkpoint_data)
            
            # Upload to GCS
            blob = bucket.blob(CHECKPOINT_GCS_PATH)
            blob.upload_from_string(checkpoint_bytes, content_type='application/octet-stream')
            
            # Verify it was saved correctly by checking size
            blob.reload()
            if blob.size != len(checkpoint_bytes):
                raise ValueError(
                    f"Checkpoint size mismatch: uploaded {len(checkpoint_bytes)} bytes, "
                    f"but blob has {blob.size} bytes"
                )
            
            processed_count = len(checkpoint_data.get('processed', {}))
            logger.info(f"💾 Checkpoint saved and verified to gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH} ({processed_count} filing(s), {len(checkpoint_bytes)} bytes)")
        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())


def update_checkpoint_entry(checkpoint_data: Dict, symbol: str, year: str, status: str, 
                            duration: float = None, error: str = None):
    """
    Update checkpoint entry for a specific filing.
    Thread-safe operation.
    Always reloads checkpoint from GCS to avoid race conditions with concurrent updates.
    """
    if not CHECKPOINT_ENABLED:
        return
    
    with checkpoint_lock:
        # CRITICAL: Reload checkpoint from GCS to get latest state (prevents race conditions)
        # Even though we have a lock, we reload to ensure we have the latest data
        # in case the checkpoint was updated externally (e.g., by v3.0-prototype.py standalone)
        try:
            # Load fresh from GCS (without verbose logging)
            blob = bucket.blob(CHECKPOINT_GCS_PATH)
            if blob.exists():
                checkpoint_bytes = blob.download_as_bytes()
                latest_checkpoint = pickle.loads(checkpoint_bytes)
                # Use the latest checkpoint from GCS
                if latest_checkpoint.get('processed'):
                    checkpoint_data = latest_checkpoint
                elif 'processed' not in checkpoint_data:
                    checkpoint_data['processed'] = {}
            elif 'processed' not in checkpoint_data:
                checkpoint_data['processed'] = {}
        except Exception as e:
            logger.warning(f"⚠️  Failed to reload checkpoint from GCS, using passed data: {e}")
            # Fallback to passed data if reload fails
            if 'processed' not in checkpoint_data:
                checkpoint_data['processed'] = {}
        
        key = f"{symbol}_{year}"
        checkpoint_data['processed'][key] = {
            'symbol': symbol,
            'year': year,
            'status': status,  # 'completed', 'failed', 'error'
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'duration': duration,
            'error': error[:200] if error else None  # Truncate long errors
        }
        
        # Save checkpoint after each update
        save_checkpoint(checkpoint_data)


def is_filing_processed(checkpoint_data: Dict, symbol: str, year: str) -> bool:
    """
    Check if a filing has already been successfully processed.
    """
    if not CHECKPOINT_ENABLED:
        return False
    
    key = f"{symbol}_{year}"
    entry = checkpoint_data.get('processed', {}).get(key)
    
    if entry and entry.get('status') == 'completed':
        return True
    
    return False


def filter_processed_filings(filings: List[Tuple[str, str]], checkpoint_data: Dict) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Filter out already processed filings.
    Returns (pending_filings, already_processed_filings)
    """
    if not CHECKPOINT_ENABLED:
        return filings, []
    
    pending = []
    processed = []
    
    for symbol, year in filings:
        if is_filing_processed(checkpoint_data, symbol, year):
            processed.append((symbol, year))
        else:
            pending.append((symbol, year))
    
    return pending, processed


def discover_10k_filings() -> List[Tuple[str, str]]:
    """
    Discover all 10-K filings in GCS bucket.
    Returns list of (symbol, year) tuples.
    
    Structure: {GCS_BASE_PATH}/{symbol}/{year}/
    """
    logger.info(f"🔍 Discovering 10-K filings in GCS bucket: {GCS_BUCKET_NAME}")
    logger.info(f"   Base path: {GCS_BASE_PATH}/")
    
    prefix = f"{GCS_BASE_PATH}/"
    filings = []
    seen_pairs = set()
    
    try:
        # List all blobs with the prefix and extract unique symbol/year pairs
        # Path format: sec-10k/SYMBOL/YEAR/section.json
        all_blobs = list(bucket.list_blobs(prefix=prefix))
        
        for blob in all_blobs:
            # Extract symbol and year from path using regex
            # Example: "sec-10k/AAPL/2024/1.json" -> symbol="AAPL", year="2024"
            match = re.match(rf'{re.escape(GCS_BASE_PATH)}/([A-Z0-9]+)/(\d{{4}})/', blob.name)
            if match:
                symbol = match.group(1).upper()
                year = match.group(2)
                
                # Validate: year should be 4 digits, symbol should be reasonable
                if year.isdigit() and len(year) == 4 and len(symbol) >= 1 and len(symbol) <= 10:
                    pair = (symbol, year)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        filings.append(pair)
        
        # Remove duplicates while preserving order
        filings = list(dict.fromkeys(filings))
        
        # Sort by symbol, then year
        filings.sort(key=lambda x: (x[0], x[1]))
        
        logger.info(f"✅ Found {len(filings)} unique 10-K filing(s)")
        return filings
        
    except Exception as e:
        logger.error(f"❌ Error discovering filings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def run_extraction_script(symbol: str, year: str, script_path: str) -> Dict:
    """
    Run v3.0-prototype.py script for a specific symbol and year.
    Returns dict with execution results.
    """
    start_time = time.time()
    process_id = f"{symbol}_{year}"
    
    logger.info(f"🚀 Starting: {symbol} - {year}")
    
    # Build command
    cmd = [
        PYTHON_EXECUTABLE,
        script_path,
        "--symbol", symbol,
        "--year", year
    ]
    
    # Set environment variables (inherit from current process)
    env = os.environ.copy()
    env["SYMBOL"] = symbol
    env["YEAR"] = year
    # Tell v3.0-prototype.py to skip checkpoint updates (batch_process_10k.py handles them)
    env["SKIP_CHECKPOINT_UPDATE"] = "true"
    
    result = {
        'symbol': symbol,
        'year': year,
        'process_id': process_id,
        'status': 'running',
        'start_time': start_time,
        'end_time': None,
        'duration': None,
        'returncode': None,
        'stdout': '',
        'stderr': '',
        'error': None
    }
    
    try:
        # Run the script and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1  # Line buffered
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        def stream_pipe(pipe, sink: List[str], level: int, prefix: str):
            for line in iter(pipe.readline, ''):
                sink.append(line)
                logger.log(level, f"[{prefix}] {line.rstrip()}")
            pipe.close()

        # Start streaming threads
        t_out = Thread(target=stream_pipe, args=(process.stdout, stdout_lines, logging.INFO, process_id), daemon=True)
        t_err = Thread(target=stream_pipe, args=(process.stderr, stderr_lines, logging.ERROR, process_id), daemon=True)
        t_out.start()
        t_err.start()

        try:
            process.wait(timeout=PROCESS_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning(f"⏱️  Process timeout ({PROCESS_TIMEOUT}s) exceeded for {symbol} - {year}, terminating...")
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

            end_time = time.time()
            duration = end_time - start_time

            result.update({
                'status': 'timeout',
                'end_time': end_time,
                'duration': duration,
                'returncode': -1,
                'stdout': ''.join(stdout_lines),
                'stderr': ''.join(stderr_lines) or f"Process timed out after {PROCESS_TIMEOUT}s"
            })

            logger.error(f"⏱️  Timeout: {symbol} - {year} (exceeded {PROCESS_TIMEOUT}s, took {duration:.1f}s)")
            return result

        # Ensure threads finish
        t_out.join(timeout=5)
        t_err.join(timeout=5)

        end_time = time.time()
        duration = end_time - start_time
        returncode = process.returncode

        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)

        stderr_preview = '\n'.join(stderr_lines[-10:]) if stderr_lines else ''

        result.update({
            'status': 'completed' if returncode == 0 else 'failed',
            'end_time': end_time,
            'duration': duration,
            'returncode': returncode,
            'stdout': stdout,
            'stderr': stderr
        })

        if returncode == 0:
            logger.info(f"✅ Completed: {symbol} - {year} (took {duration:.1f}s)")
        else:
            logger.error(f"❌ Failed: {symbol} - {year} (exit code {returncode}, took {duration:.1f}s)")
            if stderr_preview:
                logger.error(f"   Error preview:\n{stderr_preview}")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        result.update({
            'status': 'error',
            'end_time': end_time,
            'duration': duration,
            'error': str(e)
        })

        logger.error(f"❌ Exception running {symbol} - {year}: {e}")
    
    return result


def process_with_semaphore(symbol: str, year: str, script_path: str, semaphore: Semaphore, 
                            checkpoint_data: Dict) -> Dict:
    """
    Process a filing with semaphore-based concurrency control.
    Updates checkpoint after processing.
    """
    with semaphore:
        result = run_extraction_script(symbol, year, script_path)
        
        # Update checkpoint
        if CHECKPOINT_ENABLED:
            status = result.get('status', 'error')
            duration = result.get('duration')
            error = result.get('error') or result.get('stderr', '')[:200] if result.get('stderr') else None
            update_checkpoint_entry(checkpoint_data, symbol, year, status, duration, error)
        
        return result


def main():
    """
    Main execution: discover filings and process them in parallel.
    """
    print("="*80)
    print("BATCH PROCESSING: SEC 10-K FILINGS")
    print("="*80)
    print(f"📦 GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 Base Path: {GCS_BASE_PATH}/")
    print(f"🔧 Script: {SCRIPT_PATH}")
    concurrency_source = "env MAX_CONCURRENT_PROCESSES" if os.getenv("MAX_CONCURRENT_PROCESSES") else "auto (CPU-1)"
    print(f"⚙️  Max Concurrent: {MAX_CONCURRENT} [{concurrency_source}]")
    print(f"⏱️  Process Timeout: {PROCESS_TIMEOUT}s per filing")
    if CHECKPOINT_ENABLED:
        print(f"💾 Checkpoint: gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}")
    else:
        print(f"💾 Checkpoint: Disabled")
    print()
    
    # Verify script exists
    if not os.path.exists(SCRIPT_PATH):
        logger.error(f"❌ Script not found: {SCRIPT_PATH}")
        sys.exit(1)
    
    # Load checkpoint
    checkpoint_data = load_checkpoint()
    
    # Discover all filings
    filings = discover_10k_filings()
    
    if not filings:
        logger.warning("⚠️  No filings found. Exiting.")
        return
    
    # Filter out already processed filings
    pending_filings, processed_filings = filter_processed_filings(filings, checkpoint_data)
    
    # Display discovered filings
    print(f"\n📋 DISCOVERED FILINGS ({len(filings)} total):")
    if processed_filings:
        print(f"   ✅ Already processed ({len(processed_filings)}):")
        for symbol, year in processed_filings[:10]:  # Show first 10
            print(f"      {symbol} - {year}")
        if len(processed_filings) > 10:
            print(f"      ... and {len(processed_filings) - 10} more")
    
    if pending_filings:
        print(f"   ⏳ Pending processing ({len(pending_filings)}):")
        for symbol, year in pending_filings:
            print(f"      {symbol} - {year}")
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
    
    start_time = time.time()
    results = []
    running_tasks = {}  # Track running tasks for progress display
    
    # Use ThreadPoolExecutor with semaphore for concurrency control
    semaphore = Semaphore(MAX_CONCURRENT)
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        # Submit all tasks (only pending filings)
        future_to_filing = {
            executor.submit(process_with_semaphore, symbol, year, SCRIPT_PATH, semaphore, checkpoint_data): (symbol, year)
            for symbol, year in pending_filings
        }
        
        # Track running tasks
        for future, (symbol, year) in future_to_filing.items():
            running_tasks[future] = (symbol, year, time.time())
        
        # Process completed tasks with heartbeat updates
        completed = 0
        pending_futures = set(future_to_filing.keys())
        last_heartbeat = time.time()
        
        while pending_futures:
            done, pending = wait(pending_futures, timeout=HEARTBEAT_INTERVAL, return_when=FIRST_COMPLETED)
            
            # Heartbeat: no completions during the interval
            if not done:
                running_count = len(pending_futures)
                longest_running = 0.0
                if running_tasks:
                    longest_running = max(time.time() - start for (_, _, start) in running_tasks.values())
                progress = (completed / len(pending_filings)) * 100
                logger.info(
                    f"⏳ Heartbeat: {completed}/{len(pending_filings)} completed "
                    f"({progress:.1f}%), running {running_count}, "
                    f"longest running {longest_running:.1f}s"
                )
                last_heartbeat = time.time()
                continue
            
            for future in done:
                pending_futures.discard(future)
                
                symbol, year = future_to_filing[future]
                start_task_time = running_tasks.get(future, (None, None, time.time()))[2]
                task_duration = time.time() - start_task_time
                
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Remove from running tasks
                    if future in running_tasks:
                        del running_tasks[future]
                    
                    # Progress update
                    progress = (completed / len(pending_filings)) * 100
                    remaining = len(pending_filings) - completed
                    running_count = len(running_tasks)
                    
                    status_emoji = "✅" if result.get('status') == 'completed' else "❌"
                    logger.info(
                        f"{status_emoji} [{completed}/{len(pending_filings)}] {symbol} - {year} "
                        f"({task_duration:.1f}s) | Running: {running_count} | Remaining: {remaining}"
                    )
                    
                except Exception as e:
                    logger.error(f"❌ Exception processing {symbol} - {year}: {e}")
                    results.append({
                        'symbol': symbol,
                        'year': year,
                        'status': 'error',
                        'error': str(e),
                        'duration': task_duration
                    })
                    if future in running_tasks:
                        del running_tasks[future]
    
    total_time = time.time() - start_time
    
    # Summary statistics
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    status_counts = defaultdict(int)
    total_duration = 0
    
    for result in results:
        status = result.get('status', 'unknown')
        status_counts[status] += 1
        if result.get('duration'):
            total_duration += result['duration']
    
    print(f"\n📊 RESULTS:")
    print(f"   Total filings: {len(results)}")
    print(f"   ✅ Completed: {status_counts.get('completed', 0)}")
    print(f"   ❌ Failed: {status_counts.get('failed', 0)}")
    print(f"   ⏱️  Timeouts: {status_counts.get('timeout', 0)}")
    print(f"   ⚠️  Errors: {status_counts.get('error', 0)}")
    print(f"   ⏱️  Total processing time: {total_time:.1f}s")
    print(f"   ⏱️  Average time per filing: {total_duration/len(results):.1f}s" if results else "")
    
    # Show failed filings
    failed = [r for r in results if r.get('status') in ['failed', 'error', 'timeout']]
    if failed:
        print(f"\n❌ FAILED FILINGS ({len(failed)}):")
        for result in failed:
            symbol = result.get('symbol', '?')
            year = result.get('year', '?')
            status = result.get('status', 'unknown')
            error = result.get('error', result.get('stderr', 'Unknown error'))
            status_label = "⏱️ TIMEOUT" if status == 'timeout' else "❌ FAILED"
            print(f"   {status_label} {symbol} - {year}: {error[:100]}")
    
    # Show successful filings
    successful = [r for r in results if r.get('status') == 'completed']
    if successful:
        print(f"\n✅ SUCCESSFUL FILINGS ({len(successful)}):")
        for result in successful:
            symbol = result.get('symbol', '?')
            year = result.get('year', '?')
            duration = result.get('duration', 0)
            print(f"   {symbol} - {year} ({duration:.1f}s)")
    
    print(f"\n{'='*80}\n")
    
    # Final checkpoint save
    if CHECKPOINT_ENABLED:
        save_checkpoint(checkpoint_data)
        logger.info(f"💾 Final checkpoint saved")
    
    # Exit with error code if any failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

