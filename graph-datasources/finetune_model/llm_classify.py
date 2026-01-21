"""
LLM-based classification script for labeling training data.

This script uses Novita API (Qwen3 model) to classify text segments
with sentiment labels (0=Neutral, 1=Positive, 2=Negative).

Usage:
    python llm_classify.py --bucket blacksmith-sec-filings --gcs_prefix fine_tuning
"""

import os
import csv
import argparse
import logging
import tempfile
import json
import pickle
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import time
import random
import xxhash

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not installed. Install with: pip install google-cloud-storage")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Install with: pip install openai")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
NOVITA_API_BASE = "https://api.novita.ai/openai"
NOVITA_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
BATCH_SIZE = 5  # Number of concurrent API calls (lowered to avoid Novita rate limits)
MAX_RETRIES = 4
RETRY_DELAY = 1.0  # seconds


def compute_record_id(text: str, index: int) -> str:
    """
    Compute a stable identifier for a row using xxhash on index + text.
    Including the index disambiguates duplicate texts.
    """
    payload = f"{index}|{text}".encode("utf-8")
    return xxhash.xxh64(payload).hexdigest()


def load_state(state_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not state_path or not os.path.exists(state_path):
        return {}
    with open(state_path, "rb") as f:
        return pickle.load(f)


def save_state(state: Dict[str, Dict[str, Any]], state_path: Optional[str]) -> None:
    if not state_path:
        return
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(state_path) or ".") as tmp:
        pickle.dump(state, tmp)
        tmp_path = tmp.name
    os.replace(tmp_path, state_path)


def write_labeled_csv_from_state(
    rows: List[dict],
    state: Dict[str, Dict[str, Any]],
    output_csv: str,
) -> Tuple[int, int]:
    """
    Regenerate the labeled CSV in input order using state as the source of truth.
    Only processed rows with labels are emitted to avoid stale/partial data.
    Returns (written_rows, total_rows).
    """
    written = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for idx, row in enumerate(rows):
            record_id = compute_record_id(row["text"].strip('"'), idx)
            entry = state.get(record_id)
            if entry and entry.get("processed") and isinstance(entry.get("label"), int):
                writer.writerow([row["text"], entry["label"]])
                written += 1
    return written, len(rows)


def download_csv_from_gcs(
    bucket_name: str,
    csv_filename: str,
    gcp_project: Optional[str] = None,
    gcs_prefix: str = "fine_tuning",
    local_path: Optional[str] = None
) -> str:
    """
    Download a CSV file from GCS.
    
    Args:
        bucket_name: GCS bucket name
        csv_filename: Name of CSV file (e.g., "train.csv")
        gcp_project: GCP project ID (optional)
        gcs_prefix: Prefix path in bucket (default: fine_tuning)
        local_path: Local path to save file (default: same as csv_filename)
        
    Returns:
        Local path to downloaded file
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install with: pip install google-cloud-storage")
    
    # Initialize GCS client
    if gcp_project:
        client = storage.Client(project=gcp_project)
    else:
        client = storage.Client()
    
    bucket = client.bucket(bucket_name)
    
    if not bucket.exists():
        raise ValueError(f"Bucket '{bucket_name}' does not exist or is not accessible")
    
    # Construct blob path
    prefix = gcs_prefix if gcs_prefix.endswith('/') or not gcs_prefix else f"{gcs_prefix}/"
    blob_name = f"{prefix}{csv_filename}"
    
    # Download blob
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"File not found: gs://{bucket_name}/{blob_name}")
    
    # Determine local path
    if local_path is None:
        local_path = csv_filename
    
    logger.info(f"Downloading gs://{bucket_name}/{blob_name} to {local_path}")
    blob.download_to_filename(local_path)
    
    return local_path


def upload_csv_to_gcs(
    bucket_name: str,
    csv_path: str,
    csv_filename: str,
    gcp_project: Optional[str] = None,
    gcs_prefix: str = "fine_tuning"
) -> str:
    """
    Upload a CSV file to GCS.
    
    Args:
        bucket_name: GCS bucket name
        csv_path: Local path to CSV file
        csv_filename: Name for the file in GCS (e.g., "train_labeled.csv")
        gcp_project: GCP project ID (optional)
        gcs_prefix: Prefix path in bucket (default: fine_tuning)
        
    Returns:
        GCS path of uploaded file
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install with: pip install google-cloud-storage")
    
    # Initialize GCS client
    if gcp_project:
        client = storage.Client(project=gcp_project)
    else:
        client = storage.Client()
    
    bucket = client.bucket(bucket_name)
    
    if not bucket.exists():
        raise ValueError(f"Bucket '{bucket_name}' does not exist or is not accessible")
    
    # Construct blob path
    prefix = gcs_prefix if gcs_prefix.endswith('/') or not gcs_prefix else f"{gcs_prefix}/"
    blob_name = f"{prefix}{csv_filename}"
    
    # Upload blob
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(csv_path, content_type='text/csv')
    
    gcs_path = f"gs://{bucket_name}/{blob_name}"
    logger.info(f"Uploaded {csv_path} to {gcs_path}")
    
    return gcs_path


def upload_binary_to_gcs(
    bucket_name: str,
    local_path: str,
    dest_filename: str,
    gcp_project: Optional[str] = None,
    gcs_prefix: str = "fine_tuning",
) -> str:
    """
    Upload a binary file (e.g., PKL state) to GCS.
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install with: pip install google-cloud-storage")

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File to upload not found: {local_path}")

    if gcp_project:
        client = storage.Client(project=gcp_project)
    else:
        client = storage.Client()

    bucket = client.bucket(bucket_name)

    if not bucket.exists():
        raise ValueError(f"Bucket '{bucket_name}' does not exist or is not accessible")

    prefix = gcs_prefix if gcs_prefix.endswith('/') or not gcs_prefix else f"{gcs_prefix}/"
    blob_name = f"{prefix}{dest_filename}"

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type="application/octet-stream")

    gcs_path = f"gs://{bucket_name}/{blob_name}"
    logger.info(f"Uploaded {local_path} to {gcs_path}")
    return gcs_path


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of HTTP status code from an exception."""
    candidates = [
        getattr(exc, "status_code", None),
        getattr(exc, "http_status", None),
    ]
    response = getattr(exc, "response", None)
    if response is not None:
        candidates.append(getattr(response, "status_code", None))
    for code in candidates:
        if isinstance(code, int):
            return code
    # Sometimes the message contains the code; parse simple cases
    msg = str(exc)
    for code in (429, 500, 502, 503):
        if f" {code} " in msg or msg.strip().startswith(str(code)):
            return code
    return None


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Best-effort extraction of Retry-After seconds."""
    headers = None
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if isinstance(headers, dict):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                return None
    return None


def _compute_backoff_seconds(attempt: int, exc: Exception) -> float:
    """Compute backoff honoring Retry-After when available."""
    retry_after = _extract_retry_after(exc)
    if retry_after is not None:
        return max(0.5, retry_after)
    # Exponential with jitter, capped
    return min(30.0, RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.5))


def classify_text_batch(
    texts: List[str],
    client: OpenAI,
    model: str,
    max_retries: int = MAX_RETRIES,
) -> List[int]:
    """
    Classify a batch of texts using the LLM.
    """
    system_prompt = (
        "You are a financial analyst. Classify the sentiment of financial text excerpts "
        "(from earnings calls, SEC filings, or FOMC transcripts) as:\n"
        "- 0 for Neutral (no clear positive or negative sentiment)\n"
        "- 1 for Positive (optimistic, favorable, growth-oriented)\n"
        "- 2 for Negative (pessimistic, unfavorable, decline-oriented)\n\n"
        "Return a JSON object matching the provided schema with a single integer field 'label'."
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_label",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "integer",
                        "enum": [0, 1, 2],
                        "description": "Sentiment label for the provided text.",
                    }
                },
                "required": ["label"],
                "additionalProperties": False,
            },
        },
    }

    results: List[int] = []

    for text in texts:
        label: Optional[int] = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=10,
                    temperature=0.1,
                    response_format=response_format,
                )

                if not response.choices:
                    raise ValueError("No choices returned from model")
                if not response.choices[0].message or response.choices[0].message.content is None:
                    raise ValueError("No message content returned from model")

                content = response.choices[0].message.content.strip()
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Response is not valid JSON: {content}") from e

                label_value = parsed.get("label")
                if not isinstance(label_value, int) or label_value not in (0, 1, 2):
                    raise ValueError(f"Invalid label value in response: {parsed}")

                label = label_value
                break

            except Exception as e:
                status_code = _extract_status_code(e)
                logger.error(
                    "Error classifying text (attempt %s/%s, status=%s): %s",
                    attempt + 1,
                    max_retries,
                    status_code,
                    e,
                )
                if attempt < max_retries - 1:
                    delay = _compute_backoff_seconds(attempt + 1, e)
                    time.sleep(delay)

        if label is None:
            raise RuntimeError(
                f"Failed to classify text after {max_retries} attempts: {text[:80]}"
            )

        results.append(label)

    return results


def process_csv_with_llm(
    input_csv: str,
    output_csv: str,
    client: OpenAI,
    model: str,
    batch_size: int = BATCH_SIZE,
    state_path: Optional[str] = None,
    dataset_name: str = "train",
    state_upload_bucket: Optional[str] = None,
    state_upload_prefix: Optional[str] = None,
    state_upload_gcp_project: Optional[str] = None,
    state_upload_rows: int = 2000,
    state_upload_seconds: int = 600,
) -> Tuple[int, int, int]:
    """
    Process a CSV file and classify all texts using LLM.
    Uses a PKL state file keyed by xxhash(record) to resume without rework.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        client: OpenAI client configured for Novita API
        model: Model name to use
        batch_size: Number of concurrent API calls
        state_path: Path to PKL state file for this dataset
        dataset_name: Name for logging context
        state_upload_bucket: GCS bucket to upload state file for safekeeping
        state_upload_prefix: GCS prefix for state uploads
        state_upload_gcp_project: GCP project for state uploads (default: env/default)
        state_upload_rows: Upload state every N newly processed rows
        state_upload_seconds: Upload state at least every N seconds
        
    Returns:
        Tuple of (total_rows, successful_classifications, failed_classifications)
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Read input CSV
    logger.info(f"Reading {input_csv}...")
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'text' not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'text' column")
        for row in reader:
            rows.append(row)
    
    total_rows = len(rows)
    logger.info(f"Found {total_rows} rows to classify")
    
    texts = [row['text'].strip('"') for row in rows]  # Remove CSV quoting

    # Load PKL state (source of truth for progress)
    state = load_state(state_path)
    logger.info(f"[{dataset_name}] Loaded state with {len(state)} records")

    # Ensure every row has an entry in state keyed by xxhash(id|text)
    record_ids: List[str] = []
    for idx, text in enumerate(texts):
        record_id = compute_record_id(text, idx)
        record_ids.append(record_id)
        entry = state.get(record_id)
        if entry:
            if entry.get("text") != text:
                logger.warning(
                    "[%s] Text changed for index %s; resetting entry to avoid mismatch",
                    dataset_name,
                    idx,
                )
                state[record_id] = {
                    "text": text,
                    "label": None,
                    "processed": False,
                    "error": None,
                    "updated_at": None,
                }
        else:
            state[record_id] = {
                "text": text,
                "label": None,
                "processed": False,
                "error": None,
                "updated_at": None,
            }

    # Identify work to do
    pending: List[Tuple[str, str, int]] = []
    for idx, (text, record_id) in enumerate(zip(texts, record_ids)):
        entry = state[record_id]
        if entry.get("processed") and isinstance(entry.get("label"), int):
            continue
        pending.append((record_id, text, idx))

    if not pending:
        logger.info(f"[{dataset_name}] All rows already processed. Regenerating CSV.")
        written, _ = write_labeled_csv_from_state(rows, state, output_csv)
        logger.info(f"[{dataset_name}] Wrote {written}/{total_rows} rows to {output_csv}")
        return total_rows, written, 0

    logger.info(f"[{dataset_name}] {len(pending)} rows to classify; total {total_rows}")

    successful = sum(
        1 for rid in record_ids
        if state[rid].get("processed") and isinstance(state[rid].get("label"), int)
    )
    failed = 0

    state_upload_enabled = bool(state_upload_bucket and state_upload_prefix and state_path)
    last_state_upload_count = successful
    last_state_upload_time = time.time()

    def maybe_upload_state(force: bool = False) -> None:
        nonlocal last_state_upload_count, last_state_upload_time
        if not state_upload_enabled:
            return
        now = time.time()
        delta_rows = successful - last_state_upload_count
        delta_time = now - last_state_upload_time
        if force or delta_rows >= state_upload_rows or delta_time >= state_upload_seconds:
            upload_binary_to_gcs(
                bucket_name=state_upload_bucket,
                local_path=state_path,
                dest_filename=os.path.basename(state_path),
                gcp_project=state_upload_gcp_project,
                gcs_prefix=state_upload_prefix,
            )
            last_state_upload_count = successful
            last_state_upload_time = now

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            batch_texts = [item[1] for item in batch]
            future = executor.submit(classify_text_batch, batch_texts, client, model)
            futures.append((i, batch, future))

        completed_batches = 0
        for batch_offset, batch_items, future in futures:
            try:
                batch_labels = future.result()
            except Exception as e:
                now = datetime.utcnow().isoformat()
                logger.error(
                    "[%s] Batch starting at offset %s failed (%s items): %s",
                    dataset_name,
                    batch_offset,
                    len(batch_items),
                    e,
                )
                for (record_id, text, idx) in batch_items:
                    state[record_id]["error"] = str(e)
                    state[record_id]["processed"] = False
                    state[record_id]["label"] = None
                    state[record_id]["updated_at"] = now
                failed += len(batch_items)
                save_state(state, state_path)
                maybe_upload_state()
                written, _ = write_labeled_csv_from_state(rows, state, output_csv)
                completed_batches += 1
                logger.info(
                    "[%s] Progress: %s/%s rows classified (%s successful, %s failed) | CSV rows written: %s | batches done: %s",
                    dataset_name,
                    successful,
                    total_rows,
                    successful,
                    failed,
                    written,
                    completed_batches,
                )
                continue
            now = datetime.utcnow().isoformat()

            for (record_id, text, idx), label in zip(batch_items, batch_labels):
                state[record_id]["label"] = label
                state[record_id]["processed"] = True
                state[record_id]["error"] = None
                state[record_id]["updated_at"] = now
                successful += 1

            completed_batches += 1
            save_state(state, state_path)
            maybe_upload_state()
            written, _ = write_labeled_csv_from_state(rows, state, output_csv)
            logger.info(
                "[%s] Progress: %s/%s rows classified (%s successful, %s failed) | CSV rows written: %s | batches done: %s",
                dataset_name,
                successful,
                total_rows,
                successful,
                failed,
                written,
                completed_batches,
            )

    # Final upload of state
    save_state(state, state_path)
    maybe_upload_state(force=True)

    logger.info(
        "[%s] Completed: %s successful, %s failed out of %s total",
        dataset_name,
        successful,
        failed,
        total_rows,
    )

    return total_rows, successful, failed


def run_self_test() -> None:
    """Run a small, isolated test without external services."""
    logger.info("Running self test with mock client...")

    class _MockMessage:
        def __init__(self, content: str):
            self.content = content

    class _MockChoice:
        def __init__(self, content: str):
            self.message = _MockMessage(content)

    class _MockCompletions:
        def create(self, model: str, messages: List[dict], max_tokens: int, temperature: float):
            _ = model  # unused but kept for signature parity
            user_text = messages[-1]["content"].lower()
            if "good" in user_text:
                label = 1
            elif "bad" in user_text:
                label = 2
            else:
                label = 0
            return type("Response", (), {"choices": [_MockChoice(json.dumps({"label": label}))]})

    class _MockChat:
        def __init__(self):
            self.completions = _MockCompletions()

    class _MockClient:
        def __init__(self):
            self.chat = _MockChat()

    mock_client = _MockClient()

    sample_rows = [
        {"text": "The outlook is good", "label": ""},
        {"text": "The results were bad", "label": ""},
        {"text": "Flat performance with no change", "label": ""},
    ]

    input_fd = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
    output_fd = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
    input_path, output_path = input_fd.name, output_fd.name
    input_fd.close()
    output_fd.close()

    try:
        with open(input_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(sample_rows)

        _, _, _ = process_csv_with_llm(
            input_csv=input_path,
            output_csv=output_path,
            client=mock_client,
            model=NOVITA_MODEL,
            batch_size=2,
        )

        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            labels = [int(row["label"]) for row in reader]

        assert labels == [1, 2, 0], f"Unexpected labels from self test: {labels}"
        logger.info("Self test passed ✅")
    finally:
        for path in (input_path, output_path):
            try:
                os.remove(path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Classify text segments using Novita API (Qwen3 model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify train.csv and dev.csv from GCS:
  python llm_classify.py --bucket blacksmith-sec-filings

  # Specify custom GCS prefix:
  python llm_classify.py --bucket blacksmith-sec-filings --gcs_prefix fine_tuning

  # Use custom batch size:
  python llm_classify.py --bucket blacksmith-sec-filings --batch_size 10
        """
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="blacksmith-sec-filings",
        help="GCS bucket name (default: blacksmith-sec-filings)"
    )
    parser.add_argument(
        "--gcs_prefix",
        type=str,
        default="fine_tuning",
        help="GCS prefix path (default: fine_tuning)"
    )
    parser.add_argument(
        "--gcp_project",
        type=str,
        default=None,
        help="GCP project ID (optional, uses default credentials if not provided)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of concurrent API calls (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=NOVITA_MODEL,
        help=f"Model name to use (default: {NOVITA_MODEL})"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=NOVITA_API_BASE,
        help=f"Novita API base URL (default: {NOVITA_API_BASE})"
    )
    parser.add_argument(
        "--state_dir",
        type=str,
        default=".",
        help="Directory to store PKL state files for resume (default: current directory)"
    )
    parser.add_argument(
        "--state_upload_rows",
        type=int,
        default=2000,
        help="Upload state to GCS every N newly processed rows (default: 2000)"
    )
    parser.add_argument(
        "--state_upload_minutes",
        type=int,
        default=10,
        help="Upload state to GCS at least every N minutes (default: 10)"
    )
    parser.add_argument(
        "--state_upload_bucket",
        type=str,
        default=None,
        help="Override bucket for state uploads (default: use --bucket)"
    )
    parser.add_argument(
        "--state_upload_prefix",
        type=str,
        default=None,
        help="Override prefix for state uploads (default: use --gcs_prefix)"
    )
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run built-in self test with a mock client and exit"
    )
    
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    
    # Check for API key
    api_key = os.getenv("NOVITA_API_KEY")
    if not api_key:
        raise ValueError(
            "NOVITA_API_KEY environment variable not set. "
            "Please set it before running this script."
        )
    
    # Check dependencies
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is required. Install with: pip install google-cloud-storage")
    
    # Initialize OpenAI client for Novita API
    logger.info(f"Initializing Novita API client...")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  API Base: {args.api_base}")
    logger.info(f"  Batch Size: {args.batch_size}")
    
    client = OpenAI(
        api_key=api_key,
        base_url=args.api_base
    )
    
    os.makedirs(args.state_dir or ".", exist_ok=True)
    train_state_path = os.path.join(args.state_dir, "train_state.pkl")
    dev_state_path = os.path.join(args.state_dir, "dev_state.pkl")
    state_upload_rows = args.state_upload_rows
    state_upload_seconds = args.state_upload_minutes * 60
    state_upload_bucket = args.state_upload_bucket or args.bucket
    state_upload_prefix = args.state_upload_prefix or args.gcs_prefix

    # Process train.csv
    logger.info("=" * 60)
    logger.info("Processing train.csv")
    logger.info("=" * 60)
    
    try:
        train_input = download_csv_from_gcs(
            bucket_name=args.bucket,
            csv_filename="train.csv",
            gcp_project=args.gcp_project,
            gcs_prefix=args.gcs_prefix
        )
        
        train_output = "train_labeled.csv"
        train_total, train_success, train_failed = process_csv_with_llm(
            input_csv=train_input,
            output_csv=train_output,
            client=client,
            model=args.model,
            batch_size=args.batch_size,
            state_path=train_state_path,
            dataset_name="train",
            state_upload_bucket=state_upload_bucket,
            state_upload_prefix=state_upload_prefix,
            state_upload_gcp_project=args.gcp_project,
            state_upload_rows=state_upload_rows,
            state_upload_seconds=state_upload_seconds,
        )
        
        upload_csv_to_gcs(
            bucket_name=args.bucket,
            csv_path=train_output,
            csv_filename=train_output,
            gcp_project=args.gcp_project,
            gcs_prefix=args.gcs_prefix
        )
        
        logger.info(f"✅ Train CSV: {train_success}/{train_total} successful, {train_failed} failed")
        
    except Exception as e:
        logger.error(f"❌ Error processing train.csv: {e}")
        raise
    
    # Process dev.csv
    logger.info("=" * 60)
    logger.info("Processing dev.csv")
    logger.info("=" * 60)
    
    try:
        dev_input = download_csv_from_gcs(
            bucket_name=args.bucket,
            csv_filename="dev.csv",
            gcp_project=args.gcp_project,
            gcs_prefix=args.gcs_prefix
        )
        
        dev_output = "dev_labeled.csv"
        dev_total, dev_success, dev_failed = process_csv_with_llm(
            input_csv=dev_input,
            output_csv=dev_output,
            client=client,
            model=args.model,
            batch_size=args.batch_size,
            state_path=dev_state_path,
            dataset_name="dev",
            state_upload_bucket=state_upload_bucket,
            state_upload_prefix=state_upload_prefix,
            state_upload_gcp_project=args.gcp_project,
            state_upload_rows=state_upload_rows,
            state_upload_seconds=state_upload_seconds,
        )
        
        upload_csv_to_gcs(
            bucket_name=args.bucket,
            csv_path=dev_output,
            csv_filename=dev_output,
            gcp_project=args.gcp_project,
            gcs_prefix=args.gcs_prefix
        )
        
        logger.info(f"✅ Dev CSV: {dev_success}/{dev_total} successful, {dev_failed} failed")
        
    except Exception as e:
        logger.error(f"❌ Error processing dev.csv: {e}")
        raise
    
    # Summary
    logger.info("=" * 60)
    logger.info("Classification Complete!")
    logger.info("=" * 60)
    logger.info(f"Train: {train_success}/{train_total} successful ({train_success/train_total*100:.1f}%)")
    logger.info(f"Dev: {dev_success}/{dev_total} successful ({dev_success/dev_total*100:.1f}%)")
    logger.info(f"Total: {train_success + dev_success}/{train_total + dev_total} successful")
    logger.info("")
    logger.info(f"Files uploaded to GCS:")
    logger.info(f"  gs://{args.bucket}/{args.gcs_prefix}/train_labeled.csv")
    logger.info(f"  gs://{args.bucket}/{args.gcs_prefix}/dev_labeled.csv")


if __name__ == "__main__":
    main()
