    #!/usr/bin/env python3
"""
Batch Processing Script for Earnings Announcement Transcripts
Discovers all earnings announcement transcripts in GCS bucket and processes them using v1.0-graph-ea-scripts.py
Uses multiprocessing with shared model loading: each worker process loads the spaCy model once and processes multiple transcripts.
Runs up to 3 workers in parallel by default (each worker loads one model instance, so 3 models total).
Can be overridden via MAX_CONCURRENT_PROCESSES environment variable.
Includes retry logic: 3 attempts before marking as failed.
"""

import os
import sys
import subprocess
import time
import re
import logging
import pickle
import multiprocessing
import io
import contextlib
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from multiprocessing import Pool
from threading import Lock
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables from .env file
# Expected environment variables:
#   - FMP_API_KEY: Financial Modeling Prep API key (for transcript downloads)
#   - HF_TOKEN: Hugging Face token (for model downloads)
#   - NOVITA_API_KEY: Novita.ai API key (for LLM enrichment)
#   - SKIP_LLM: Set to "true" to skip LLM processing (faster, spaCy only)
#   - MEMGRAPH_HOST: MemgraphDB host (default: localhost)
#   - MEMGRAPH_PORT: MemgraphDB port (default: 7687)
#   - MEMGRAPH_USER: MemgraphDB username (optional)
#   - MEMGRAPH_PASSWORD: MemgraphDB password (optional)
#   - DATA_DIR: Data directory path (optional)
#   - LOG_LEVEL: Logging level (default: INFO)
#   - MAX_CONCURRENT_PROCESSES: Override worker count (default: CPU-1)
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

# GCS Configuration (matches v1.0-graph-ea-scripts.py)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "earnings-announcement-transcripts")

# Script configuration
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "v1.0-graph-ea-scripts.py")

# Calculate optimal worker count
# NOTE: With multiprocessing, each worker process loads the spaCy transformer model (en_core_web_trf) once.
# Each model instance uses 2-4GB RAM, so 3 workers = 3 model instances = ~6-12GB total.
# This is much better than subprocess approach where each transcript loads its own model.
# This can be overridden via MAX_CONCURRENT_PROCESSES environment variable.
CPU_COUNT = multiprocessing.cpu_count() or 4
DEFAULT_MAX_CONCURRENT = 3  # Conservative default to avoid memory exhaustion

# Allow override via environment variable
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_PROCESSES", str(DEFAULT_MAX_CONCURRENT)))

# Global variable to store the extraction module in each worker process
_worker_extraction_module = None
PYTHON_EXECUTABLE = os.getenv("PYTHON_EXECUTABLE", sys.executable)
MAX_RETRIES = 3  # Number of retry attempts before giving up
SKIP_LLM = os.getenv("SKIP_LLM", "false").lower() == "true"  # Skip LLM enrichment if true

# Checkpoint configuration
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", f"{GCS_BASE_PATH}/.checkpoint/processed_transcripts.pkl")
CHECKPOINT_UPDATE_INTERVAL = int(os.getenv("CHECKPOINT_UPDATE_INTERVAL", "10"))  # Update every N completions

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Thread-safe checkpoint tracking
checkpoint_lock = Lock()
processed_transcripts: Set[Tuple[str, str, str]] = set()


def load_checkpoint() -> Set[Tuple[str, str, str]]:
    """
    Load checkpoint from GCS pickle file.
    Returns set of (symbol, year, quarter) tuples that have been processed.
    """
    try:
        blob = bucket.blob(CHECKPOINT_PATH)
        if blob.exists():
            logger.info(f"📥 Loading checkpoint from: gs://{GCS_BUCKET_NAME}/{CHECKPOINT_PATH}")
            checkpoint_data = blob.download_as_bytes()
            processed = pickle.loads(checkpoint_data)
            logger.info(f"✅ Loaded {len(processed)} processed transcript(s) from checkpoint")
            return processed
        else:
            logger.info(f"ℹ️  No checkpoint found at gs://{GCS_BUCKET_NAME}/{CHECKPOINT_PATH}, starting fresh")
            return set()
    except Exception as e:
        logger.warning(f"⚠️  Error loading checkpoint: {e}. Starting fresh.")
        return set()


def save_checkpoint(processed: Set[Tuple[str, str, str]]):
    """
    Save checkpoint to GCS pickle file.
    Verifies that the checkpoint was saved correctly by downloading and comparing.
    
    Args:
        processed: Set of (symbol, year, quarter) tuples that have been processed
    """
    try:
        checkpoint_data = pickle.dumps(processed)
        blob = bucket.blob(CHECKPOINT_PATH)
        blob.upload_from_string(checkpoint_data, content_type='application/octet-stream')
        
        # Reload blob to refresh metadata
        blob.reload()
        
        # Verify the checkpoint was saved correctly by downloading and comparing
        if not blob.exists():
            raise Exception(f"Checkpoint blob does not exist after upload")
        
        if blob.size != len(checkpoint_data):
            raise Exception(f"Checkpoint size mismatch: uploaded {len(checkpoint_data)} bytes, blob has {blob.size} bytes")
        
        # Download and verify content matches
        downloaded_data = blob.download_as_bytes()
        if downloaded_data != checkpoint_data:
            raise Exception(f"Checkpoint content mismatch: uploaded data does not match downloaded data")
        
        # Verify it can be unpickled correctly
        try:
            loaded_processed = pickle.loads(downloaded_data)
            if loaded_processed != processed:
                raise Exception(f"Checkpoint data mismatch: loaded set does not match original set")
        except Exception as unpickle_error:
            raise Exception(f"Checkpoint cannot be unpickled: {unpickle_error}")
        
        logger.info(f"💾 Saved checkpoint to gs://{GCS_BUCKET_NAME}/{CHECKPOINT_PATH} ({len(processed)} transcript(s), {len(checkpoint_data)} bytes) - verified")
    except Exception as e:
        logger.error(f"❌ Error saving checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise  # Re-raise to ensure caller knows save failed


def mark_as_processed(symbol: str, year: str, quarter: str):
    """
    Mark a transcript as processed in the checkpoint (thread-safe).
    Periodically saves to GCS.
    
    Args:
        symbol: Stock symbol
        year: Year
        quarter: Quarter
    """
    global processed_transcripts
    
    with checkpoint_lock:
        triple = (symbol.upper(), year, quarter)
        processed_transcripts.add(triple)
        
        # Periodically save checkpoint (every N completions)
        if len(processed_transcripts) % CHECKPOINT_UPDATE_INTERVAL == 0:
            try:
                save_checkpoint(processed_transcripts)
            except Exception as e:
                # Log error but don't fail the entire batch process
                # The checkpoint will be saved again at the end
                logger.warning(f"⚠️  Failed to save periodic checkpoint (will retry at end): {e}")


def discover_earnings_transcripts() -> List[Tuple[str, str, str]]:
    """
    Discover all earnings announcement transcripts in GCS bucket.
    Filters out already-processed transcripts based on checkpoint.
    Returns list of (symbol, year, quarter) tuples that need processing.
    
    Structure: {GCS_BASE_PATH}/{symbol}/{year}.{quarter}.json
    Example: earnings-announcement-transcripts/AAPL/2024.1.json
    """
    logger.info(f"🔍 Discovering earnings transcripts in GCS bucket: {GCS_BUCKET_NAME}")
    logger.info(f"   Base path: {GCS_BASE_PATH}/")
    
    # Load checkpoint to filter out already-processed transcripts
    global processed_transcripts
    processed_transcripts = load_checkpoint()
    
    prefix = f"{GCS_BASE_PATH}/"
    transcripts = []
    seen_triples = set()
    
    try:
        # List all blobs with the prefix and extract unique symbol/year/quarter triples
        # Path format: earnings-announcement-transcripts/SYMBOL/YEAR.QUARTER.json
        all_blobs = list(bucket.list_blobs(prefix=prefix))
        
        for blob in all_blobs:
            if not blob.name.endswith('.json'):
                continue
            
            # Skip checkpoint files
            if '.checkpoint' in blob.name:
                continue
            
            # Extract symbol, year, and quarter from path using regex
            # Example: "earnings-announcement-transcripts/AAPL/2024.1.json" -> symbol="AAPL", year="2024", quarter="1"
            match = re.match(rf'{re.escape(GCS_BASE_PATH)}/([A-Z0-9]+)/(\d{{4}})\.(\d)\.json', blob.name)
            if match:
                symbol = match.group(1).upper()
                year = match.group(2)
                quarter = match.group(3)
                
                # Validate: year should be 4 digits, quarter should be 1-4, symbol should be reasonable
                if (year.isdigit() and len(year) == 4 and 
                    quarter.isdigit() and 1 <= int(quarter) <= 4 and
                    len(symbol) >= 1 and len(symbol) <= 10):
                    triple = (symbol, year, quarter)
                    if triple not in seen_triples:
                        seen_triples.add(triple)
                        transcripts.append(triple)
        
        # Remove duplicates while preserving order
        transcripts = list(dict.fromkeys(transcripts))
        
        # Filter out already-processed transcripts
        total_found = len(transcripts)
        transcripts = [t for t in transcripts if t not in processed_transcripts]
        skipped = total_found - len(transcripts)
        
        # Sort by symbol, then year, then quarter
        transcripts.sort(key=lambda x: (x[0], x[1], x[2]))
        
        logger.info(f"✅ Found {total_found} unique earnings transcript(s)")
        if skipped > 0:
            logger.info(f"⏭️  Skipping {skipped} already-processed transcript(s) (from checkpoint)")
        logger.info(f"📋 {len(transcripts)} transcript(s) need processing")
        return transcripts
        
    except Exception as e:
        logger.error(f"❌ Error discovering transcripts: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def _init_worker():
    """
    Initialize worker process by importing the extraction module.
    This will load the spaCy model once per worker process.
    Environment variables (including SKIP_LLM) are inherited from parent process,
    so the extraction module will read them correctly when it loads.
    """
    global _worker_extraction_module
    try:
        # Import the extraction module - this will load the model once per worker
        # Note: Environment variables are inherited by multiprocessing workers,
        # so SKIP_LLM will be read correctly from os.getenv() in the extraction module
        script_dir = os.path.dirname(SCRIPT_PATH)
        sys.path.insert(0, script_dir)
        import importlib.util
        spec = importlib.util.spec_from_file_location("v1_0_graph_ea_scripts", SCRIPT_PATH)
        _worker_extraction_module = importlib.util.module_from_spec(spec)
        # Suppress stdout during model loading to reduce noise
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                spec.loader.exec_module(_worker_extraction_module)
    except Exception as e:
        # Note: logger might not work in worker process, so print to stderr
        print(f"❌ Error initializing worker (loading extraction module): {e}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        raise


def process_single_transcript(args: Tuple[str, str, str, int]) -> Dict:
    """
    Process a single transcript using the imported extraction module.
    This function runs in worker processes where the model is already loaded.
    
    Args:
        args: Tuple of (symbol, year, quarter, attempt)
    """
    symbol, year, quarter, attempt = args
    start_time = time.time()
    process_id = f"{symbol}_{year}_Q{quarter}"
    
    result = {
        'symbol': symbol,
        'year': year,
        'quarter': quarter,
        'process_id': process_id,
        'status': 'running',
        'start_time': start_time,
        'end_time': None,
        'duration': None,
        'returncode': None,
        'stdout': '',
        'stderr': '',
        'error': None,
        'attempt': attempt
    }
    
    try:
        # Redirect stdout/stderr to capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Call the extraction function
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            extraction_result = _worker_extraction_module.process_documents(
                symbol=symbol,
                year=year,
                quarter=quarter
            )
        
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if processing was successful (returns None on failure)
        if extraction_result is not None:
            result.update({
                'status': 'completed',
                'end_time': end_time,
                'duration': duration,
                'returncode': 0,
                'stdout': stdout,
                'stderr': stderr
            })
        else:
            result.update({
                'status': 'failed',
                'end_time': end_time,
                'duration': duration,
                'returncode': 1,
                'stdout': stdout,
                'stderr': stderr,
                'error': 'process_documents returned None'
            })
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        result.update({
            'status': 'error',
            'end_time': end_time,
            'duration': duration,
            'error': str(e)
        })
        import traceback
        result['stderr'] = traceback.format_exc()
    
    return result


def run_extraction_script(symbol: str, year: str, quarter: str, script_path: str, attempt: int = 1) -> Dict:
    """
    Run v1.0-graph-ea-scripts.py script for a specific symbol, year, and quarter.
    Returns dict with execution results.
    
    Args:
        symbol: Stock symbol (e.g., AAPL)
        year: Year (e.g., 2024)
        quarter: Quarter (e.g., 1, 2, 3, 4)
        script_path: Path to the extraction script
        attempt: Current attempt number (for retry logic)
    """
    start_time = time.time()
    process_id = f"{symbol}_{year}_Q{quarter}"
    
    if attempt > 1:
        logger.info(f"🔄 Retry {attempt}/{MAX_RETRIES}: {symbol} - {year} Q{quarter}")
    else:
        logger.info(f"🚀 Starting: {symbol} - {year} Q{quarter}")
    
    # Build command
    cmd = [
        PYTHON_EXECUTABLE,
        script_path,
        "--symbol", symbol,
        "--year", year,
        "--quarter", quarter
    ]
    
    # Add --skip-llm flag if SKIP_LLM is enabled
    if SKIP_LLM:
        cmd.append("--skip-llm")
    
    # Set environment variables (inherit all from current process, including .env)
    # All environment variables (FMP_API_KEY, HF_TOKEN, NOVITA_API_KEY, etc.)
    # are automatically passed through to the subprocess via os.environ.copy()
    env = os.environ.copy()
    env["SYMBOL"] = symbol
    env["YEAR"] = year
    env["QUARTER"] = quarter
    
    result = {
        'symbol': symbol,
        'year': year,
        'quarter': quarter,
        'process_id': process_id,
        'status': 'running',
        'start_time': start_time,
        'end_time': None,
        'duration': None,
        'returncode': None,
        'stdout': '',
        'stderr': '',
        'error': None,
        'attempt': attempt
    }
    
    try:
        # Run the script and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1  # Line buffered
        )
        
        # Read all output (will block until process completes)
        stdout, stderr = process.communicate()
        
        returncode = process.returncode
        end_time = time.time()
        duration = end_time - start_time
        
        # Process output
        stdout_lines = stdout.splitlines() if stdout else []
        stderr_lines = stderr.splitlines() if stderr else []
        
        result.update({
            'status': 'completed' if returncode == 0 else 'failed',
            'end_time': end_time,
            'duration': duration,
            'returncode': returncode,
            'stdout': stdout,
            'stderr': stderr
        })
        
        if returncode == 0:
            if attempt > 1:
                logger.info(f"✅ Completed (after {attempt} attempts): {symbol} - {year} Q{quarter} (took {duration:.1f}s)")
            else:
                logger.info(f"✅ Completed: {symbol} - {year} Q{quarter} (took {duration:.1f}s)")
        else:
            if attempt < MAX_RETRIES:
                logger.warning(f"⚠️  Failed (attempt {attempt}/{MAX_RETRIES}): {symbol} - {year} Q{quarter} (exit code {returncode}, took {duration:.1f}s)")
            else:
                logger.error(f"❌ Failed (final attempt): {symbol} - {year} Q{quarter} (exit code {returncode}, took {duration:.1f}s)")
                if stderr_lines:
                    # Show last few error lines
                    error_preview = '\n'.join(stderr_lines[-10:])
                    logger.error(f"   Error preview:\n{error_preview}")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        result.update({
            'status': 'error',
            'end_time': end_time,
            'duration': duration,
            'error': str(e)
        })
        
        if attempt < MAX_RETRIES:
            logger.warning(f"⚠️  Exception (attempt {attempt}/{MAX_RETRIES}) running {symbol} - {year} Q{quarter}: {e}")
        else:
            logger.error(f"❌ Exception (final attempt) running {symbol} - {year} Q{quarter}: {e}")
    
    return result


def process_with_retry_mp(args: Tuple[str, str, str, int]) -> Dict:
    """
    Process a transcript with retry logic (for multiprocessing).
    This wraps process_single_transcript with retry logic.
    
    Args:
        args: Tuple of (symbol, year, quarter, attempt_number)
    """
    symbol, year, quarter, attempt = args
    last_result = None
    
    for retry_attempt in range(1, MAX_RETRIES + 1):
        result = process_single_transcript((symbol, year, quarter, retry_attempt))
        last_result = result
        
        # If successful, return immediately
        if result.get('status') == 'completed':
            return result
        
        # If not the last attempt, wait a bit before retrying
        if retry_attempt < MAX_RETRIES:
            wait_time = retry_attempt * 2  # Exponential backoff: 2s, 4s, 6s
            time.sleep(wait_time)
    
    # All retries exhausted
    last_result['status'] = 'failed_after_retries'
    return last_result


def main():
    """
    Main execution: discover transcripts and process them in parallel.
    """
    print("="*80)
    print("BATCH PROCESSING: EARNINGS ANNOUNCEMENT TRANSCRIPTS")
    print("="*80)
    print(f"📦 GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 Base Path: {GCS_BASE_PATH}/")
    print(f"🔧 Script: {SCRIPT_PATH}")
    print(f"🖥️  CPU Cores: {CPU_COUNT}")
    print(f"⚙️  Max Concurrent Workers: {MAX_CONCURRENT} (default: {DEFAULT_MAX_CONCURRENT}, can override via MAX_CONCURRENT_PROCESSES env var)")
    print(f"💾 Model Loading: Each worker loads spaCy model once (shared across transcripts in that worker)")
    print(f"🔄 Max Retries: {MAX_RETRIES}")
    print(f"💾 Checkpoint: gs://{GCS_BUCKET_NAME}/{CHECKPOINT_PATH}")
    print(f"🤖 Skip LLM: {SKIP_LLM}")
    
    # Show MemgraphDB connection info if configured
    memgraph_host = os.getenv("MEMGRAPH_HOST", "localhost")
    memgraph_port = os.getenv("MEMGRAPH_PORT", "7687")
    memgraph_user = os.getenv("MEMGRAPH_USER", "")
    memgraph_password = os.getenv("MEMGRAPH_PASSWORD", "")
    print(f"🗄️  MemgraphDB: {memgraph_host}:{memgraph_port}" + 
          (f" (user: {memgraph_user})" if memgraph_user else ""))
    
    # Show key environment variables status (for debugging/config verification)
    key_env_vars = {
        "FMP_API_KEY": os.getenv("FMP_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "NOVITA_API_KEY": os.getenv("NOVITA_API_KEY"),
        "DATA_DIR": os.getenv("DATA_DIR"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }
    
    print(f"\n🔑 Environment Variables Status:")
    for var_name, var_value in key_env_vars.items():
        if var_value:
            # Mask sensitive values
            if "KEY" in var_name or "TOKEN" in var_name or "PASSWORD" in var_name:
                display_value = f"{var_value[:8]}...{var_value[-4:]}" if len(var_value) > 12 else "***"
            else:
                display_value = var_value
            print(f"   ✅ {var_name}: {display_value}")
        else:
            print(f"   ⚠️  {var_name}: not set")
    print()
    
    # Verify script exists
    if not os.path.exists(SCRIPT_PATH):
        logger.error(f"❌ Script not found: {SCRIPT_PATH}")
        sys.exit(1)
    
    # Discover all transcripts
    transcripts = discover_earnings_transcripts()
    
    if not transcripts:
        logger.warning("⚠️  No transcripts found. Exiting.")
        return
    
    # Display discovered transcripts
    print(f"\n📋 DISCOVERED TRANSCRIPTS ({len(transcripts)} total):")
    for symbol, year, quarter in transcripts:
        print(f"   {symbol} - {year} Q{quarter}")
    print()
    
    # Ask for confirmation if processing many transcripts
    if len(transcripts) > 10:
        response = input(f"⚠️  About to process {len(transcripts)} transcripts. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Cancelled by user.")
            return
    
    # Process transcripts with multiprocessing (shared model loading)
    print(f"\n🚀 PROCESSING {len(transcripts)} TRANSCRIPT(S) (max {MAX_CONCURRENT} workers, model loaded once per worker)...")
    print("="*80)
    print()
    
    start_time = time.time()
    results = []
    
    # Prepare tasks for multiprocessing: (symbol, year, quarter, attempt)
    # attempt starts at 1, retry logic is handled in process_with_retry_mp
    tasks = [(symbol, year, quarter, 1) for symbol, year, quarter in transcripts]
    
    # Use multiprocessing.Pool with worker initialization
    # Each worker will load the model once when it starts
    with Pool(processes=MAX_CONCURRENT, initializer=_init_worker) as pool:
        # Process all transcripts
        completed = 0
        for result in pool.imap(process_with_retry_mp, tasks):
            results.append(result)
            completed += 1
            
            symbol = result.get('symbol', '?')
            year = result.get('year', '?')
            quarter = result.get('quarter', '?')
            duration = result.get('duration', 0)
            attempt = result.get('attempt', 1)
            
            # Progress update
            progress = (completed / len(transcripts)) * 100
            remaining = len(transcripts) - completed
            
            status_emoji = "✅" if result.get('status') == 'completed' else "❌"
            attempt_info = f" (attempt {attempt})" if attempt > 1 else ""
            logger.info(
                f"{status_emoji} [{completed}/{len(transcripts)}] {symbol} - {year} Q{quarter}{attempt_info} "
                f"({duration:.1f}s) | Remaining: {remaining}"
            )
            
            # Mark as processed if successful
            if result.get('status') == 'completed':
                mark_as_processed(symbol, year, quarter)
    
    total_time = time.time() - start_time
    
    # Summary statistics
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    status_counts = defaultdict(int)
    total_duration = 0
    retry_counts = defaultdict(int)
    
    for result in results:
        status = result.get('status', 'unknown')
        status_counts[status] += 1
        if result.get('duration'):
            total_duration += result['duration']
        attempt = result.get('attempt', 1)
        if attempt > 1:
            retry_counts[attempt] += 1
    
    print(f"\n📊 RESULTS:")
    print(f"   Total transcripts: {len(results)}")
    print(f"   ✅ Completed: {status_counts.get('completed', 0)}")
    print(f"   ❌ Failed: {status_counts.get('failed', 0)}")
    print(f"   ❌ Failed after retries: {status_counts.get('failed_after_retries', 0)}")
    print(f"   ⚠️  Errors: {status_counts.get('error', 0)}")
    print(f"   ⏱️  Total processing time: {total_time:.1f}s")
    if results:
        print(f"   ⏱️  Average time per transcript: {total_duration/len(results):.1f}s")
    
    if retry_counts:
        print(f"\n🔄 RETRY STATISTICS:")
        for attempt, count in sorted(retry_counts.items()):
            print(f"   Attempt {attempt}: {count} transcript(s)")
    
    # Show failed transcripts
    failed = [r for r in results if r.get('status') in ['failed', 'failed_after_retries', 'error']]
    if failed:
        print(f"\n❌ FAILED TRANSCRIPTS ({len(failed)}):")
        for result in failed:
            symbol = result.get('symbol', '?')
            year = result.get('year', '?')
            quarter = result.get('quarter', '?')
            attempt = result.get('attempt', 1)
            error = result.get('error', result.get('stderr', 'Unknown error'))
            error_preview = error[:100] if isinstance(error, str) else str(error)[:100]
            print(f"   {symbol} - {year} Q{quarter} (attempt {attempt}): {error_preview}")
    
    # Show successful transcripts
    successful = [r for r in results if r.get('status') == 'completed']
    if successful:
        print(f"\n✅ SUCCESSFUL TRANSCRIPTS ({len(successful)}):")
        for result in successful:
            symbol = result.get('symbol', '?')
            year = result.get('year', '?')
            quarter = result.get('quarter', '?')
            duration = result.get('duration', 0)
            attempt = result.get('attempt', 1)
            attempt_str = f" (after {attempt} attempts)" if attempt > 1 else ""
            print(f"   {symbol} - {year} Q{quarter}{attempt_str} ({duration:.1f}s)")
    
    print(f"\n{'='*80}\n")
    
    # Final checkpoint save
    logger.info("💾 Saving final checkpoint...")
    try:
        with checkpoint_lock:
            save_checkpoint(processed_transcripts)
        logger.info("✅ Final checkpoint saved successfully")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to save final checkpoint: {e}")
        logger.error("   Processed transcripts may be lost if script is interrupted")
        # Don't exit here - let the normal exit logic handle it
    
    # Exit with error code if any failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

