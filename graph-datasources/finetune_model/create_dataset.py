"""
Script to create training dataset from earnings call transcript JSON files in GCS.

This script processes JSON files from Google Cloud Storage buckets containing 
earnings call transcripts and creates train.csv and dev.csv files for fine-tuning 
the SEC-BERT model.

Usage:
    python create_dataset.py --bucket blacksmith-sec-filings --prefixes earnings-announcement-transcripts --output_dir . --strategy sentences
"""

import json
import os
import csv
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not installed. Install with: pip install google-cloud-storage")


def get_available_processors() -> int:
    """
    Get the number of available CPU processors.
    Uses CPU count from multiprocessing with a sensible default.
    """
    try:
        cpu_count = mp.cpu_count()
        # Use all CPUs, but leave 1 for system if more than 4 CPUs available
        if cpu_count > 4:
            return cpu_count - 1
        return cpu_count
    except Exception:
        # Fallback if cpu_count() fails
        return 4


def load_json_from_gcs(bucket: storage.Bucket, blob_name: str) -> List[Dict]:
    """Load and parse a JSON file from GCS bucket."""
    try:
        blob = bucket.blob(blob_name)
        content = blob.download_as_text(encoding='utf-8')
        data = json.loads(content)
        # Handle both single objects and arrays
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"Unexpected JSON structure in gs://{bucket.name}/{blob_name}")
    except Exception as e:
        raise ValueError(f"Error loading gs://{bucket.name}/{blob_name}: {e}")


def upload_csv_to_gcs(
    bucket_name: str,
    train_csv: str,
    dev_csv: str,
    gcp_project: Optional[str] = None,
    gcs_prefix: str = "fine_tuning"
) -> Tuple[str, str]:
    """
    Upload train.csv and dev.csv files to GCS bucket.
    
    Args:
        bucket_name: GCS bucket name
        train_csv: Local path to train.csv file
        dev_csv: Local path to dev.csv file
        gcp_project: GCP project ID (optional)
        gcs_prefix: Prefix path in bucket for storing files (default: fine_tuning)
        
    Returns:
        Tuple of (gcs_train_path, gcs_dev_path)
    """
    if not GCS_AVAILABLE:
        print("Warning: google-cloud-storage not available. Skipping GCS upload.")
        return None, None
    
    try:
        # Initialize GCS client
        if gcp_project:
            client = storage.Client(project=gcp_project)
        else:
            client = storage.Client()
        
        bucket = client.bucket(bucket_name)
        
        if not bucket.exists():
            print(f"Warning: Bucket '{bucket_name}' does not exist. Skipping GCS upload.")
            return None, None
        
        # Ensure prefix ends with / if not empty
        prefix = gcs_prefix if gcs_prefix.endswith('/') or not gcs_prefix else f"{gcs_prefix}/"
        
        # Upload train.csv
        train_blob_name = f"{prefix}train.csv"
        train_blob = bucket.blob(train_blob_name)
        train_blob.upload_from_filename(train_csv, content_type='text/csv')
        gcs_train_path = f"gs://{bucket_name}/{train_blob_name}"
        
        # Upload dev.csv
        dev_blob_name = f"{prefix}dev.csv"
        dev_blob = bucket.blob(dev_blob_name)
        dev_blob.upload_from_filename(dev_csv, content_type='text/csv')
        gcs_dev_path = f"gs://{bucket_name}/{dev_blob_name}"
        
        print(f"\nUploaded to GCS:")
        print(f"  {gcs_train_path}")
        print(f"  {gcs_dev_path}")
        
        return gcs_train_path, gcs_dev_path
        
    except Exception as e:
        print(f"Warning: Failed to upload files to GCS: {e}")
        print("  Files are still available locally.")
        return None, None


def list_json_files_from_gcs(
    bucket_name: str,
    prefixes: List[str],
    gcp_project: Optional[str] = None
) -> List[Tuple[storage.Bucket, str]]:
    """
    List all JSON files from GCS bucket with given prefixes.
    
    Args:
        bucket_name: Name of the GCS bucket
        prefixes: List of prefix paths within the bucket to search
        gcp_project: GCP project ID (optional, uses default if not provided)
        
    Returns:
        List of tuples: (bucket, blob_name)
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
    
    json_files = []
    all_blobs = set()  # Track unique blobs to avoid duplicates
    
    for prefix in prefixes:
        print(f"  Listing files with prefix: {prefix}")
        # Ensure prefix ends with / if not empty
        search_prefix = prefix if prefix.endswith('/') or not prefix else f"{prefix}/"
        
        blobs = bucket.list_blobs(prefix=search_prefix)
        for blob in blobs:
            # Only process JSON files
            if blob.name.endswith('.json') and blob.name not in all_blobs:
                all_blobs.add(blob.name)
                json_files.append((bucket, blob.name))
    
    return json_files


def split_by_sentences(text: str, min_length: int = 50, max_length: int = 512) -> List[str]:
    """
    Split text into sentences, filtering by length.
    
    Args:
        text: Input text
        min_length: Minimum character length for a sentence to be included
        max_length: Maximum character length (longer sentences are truncated)
        
    Returns:
        List of sentence strings
    """
    # Split by sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    filtered = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        # Remove speaker labels (e.g., "Tim Cook:", "Operator:")
        sent = re.sub(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s*', '', sent)
        
        if len(sent) < min_length:
            continue
            
        # Truncate if too long (model max is 512 tokens, roughly ~400 chars)
        if len(sent) > max_length:
            sent = sent[:max_length].rsplit(' ', 1)[0] + '.'
            
        filtered.append(sent)
    
    return filtered


def split_by_paragraphs(text: str, min_length: int = 100, max_length: int = 512) -> List[str]:
    """
    Split text into paragraphs (separated by double newlines).
    
    Args:
        text: Input text
        min_length: Minimum character length for a paragraph to be included
        max_length: Maximum character length (longer paragraphs are truncated)
        
    Returns:
        List of paragraph strings
    """
    paragraphs = text.split('\n\n')
    
    filtered = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Remove speaker labels at start
        para = re.sub(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s*', '', para)
        
        # Remove single newlines and extra whitespace
        para = ' '.join(para.split())
        
        if len(para) < min_length:
            continue
            
        # Truncate if too long
        if len(para) > max_length:
            para = para[:max_length].rsplit(' ', 1)[0] + '.'
            
        filtered.append(para)
    
    return filtered


def split_by_qna(text: str, min_length: int = 100, max_length: int = 512) -> List[str]:
    """
    Split text into Q&A pairs or individual questions/answers.
    
    Args:
        text: Input text
        min_length: Minimum character length
        max_length: Maximum character length
        
    Returns:
        List of Q&A text segments
    """
    # Look for Q&A patterns like "Q -", "A -", "Question:", "Answer:"
    # or analyst names followed by questions
    segments = []
    
    # Split by common Q&A markers
    qa_pattern = r'(?:Q\s*[:\-]\s*|A\s*[:\-]\s*|Question\s*[:\-]\s*|Answer\s*[:\-]\s*|Operator\s*[:\-]\s*)'
    parts = re.split(qa_pattern, text, flags=re.IGNORECASE)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Remove speaker labels
        part = re.sub(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:\s*', '', part)
        part = ' '.join(part.split())
        
        if len(part) < min_length:
            continue
            
        if len(part) > max_length:
            # Split long Q&A into sentences
            sentences = split_by_sentences(part, min_length, max_length)
            segments.extend(sentences)
        else:
            segments.append(part)
    
    return segments


def extract_text_segments(
    transcript_data: List[Dict],
    strategy: str = "sentences",
    min_length: int = 50,
    max_length: int = 512
) -> List[Tuple[str, Dict]]:
    """
    Extract text segments from transcript data.
    
    Args:
        transcript_data: List of transcript dictionaries with 'content' field
        strategy: Extraction strategy ('sentences', 'paragraphs', 'qna', 'full')
        min_length: Minimum character length for segments
        max_length: Maximum character length for segments
        
    Returns:
        List of tuples: (text_segment, metadata_dict)
    """
    all_segments = []
    
    for item in transcript_data:
        content = item.get('content', '')
        if not content:
            continue
            
        metadata = {
            'symbol': item.get('symbol', 'UNKNOWN'),
            'quarter': item.get('quarter', '?'),
            'year': item.get('year', '?'),
            'date': item.get('date', '?')
        }
        
        if strategy == "sentences":
            segments = split_by_sentences(content, min_length, max_length)
        elif strategy == "paragraphs":
            segments = split_by_paragraphs(content, min_length, max_length)
        elif strategy == "qna":
            segments = split_by_qna(content, min_length, max_length)
        elif strategy == "full":
            # Use entire transcript as one segment (may need truncation)
            text = ' '.join(content.split())
            if len(text) > max_length:
                text = text[:max_length].rsplit(' ', 1)[0] + '.'
            segments = [text] if len(text) >= min_length else []
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        for segment in segments:
            all_segments.append((segment, metadata))
    
    return all_segments


def process_single_json_file(args_tuple: Tuple[storage.Bucket, str, str, int, int]) -> Tuple[str, List[Tuple[str, Dict]]]:
    """
    Process a single JSON file from GCS.
    This function is designed to be called in parallel.
    
    Args:
        args_tuple: Tuple of (bucket, blob_name, strategy, min_length, max_length)
        
    Returns:
        Tuple of (filename, list of segments)
    """
    bucket, blob_name, strategy, min_length, max_length = args_tuple
    
    try:
        transcript_data = load_json_from_gcs(bucket, blob_name)
        segments = extract_text_segments(transcript_data, strategy, min_length, max_length)
        filename = blob_name.split('/')[-1]
        return filename, segments
    except Exception as e:
        filename = blob_name.split('/')[-1]
        raise Exception(f"Error processing {filename}: {e}")


def create_dataset_from_transcripts(
    bucket_name: str,
    prefixes: List[str],
    output_dir: str = ".",
    strategy: str = "sentences",
    train_split: float = 0.8,
    min_length: int = 50,
    max_length: int = 512,
    label_value: int = 0,
    seed: int = 42,
    gcp_project: Optional[str] = None,
    max_files: Optional[int] = None,
    num_workers: Optional[int] = None
) -> Tuple[str, str]:
    """
    Create train.csv and dev.csv from transcript JSON files in GCS.
    
    Args:
        bucket_name: GCS bucket name
        prefixes: List of prefix paths within bucket to search for JSON files
        output_dir: Directory to save output CSV files
        strategy: Text extraction strategy ('sentences', 'paragraphs', 'qna', 'full')
        train_split: Fraction of data for training (rest goes to dev)
        min_length: Minimum character length for text segments
        max_length: Maximum character length for text segments
        label_value: Label to assign to all segments (default 0 - you'll need to label manually)
        seed: Random seed for train/dev split
        gcp_project: GCP project ID (optional)
        max_files: Maximum number of files to process (optional, for testing)
        
    Returns:
        Tuple of (train_csv_path, dev_csv_path)
    """
    # List all JSON files from GCS
    print(f"Listing JSON files from bucket: {bucket_name}")
    json_files = list_json_files_from_gcs(bucket_name, prefixes, gcp_project)
    
    if not json_files:
        raise ValueError(f"No JSON files found in bucket '{bucket_name}' with prefixes: {prefixes}")
    
    if max_files:
        json_files = json_files[:max_files]
        print(f"Limiting to first {max_files} files for processing")
    
    print(f"Found {len(json_files)} JSON files")
    
    # Determine number of workers for parallel processing
    if num_workers is None:
        num_workers = get_available_processors()
    print(f"Using {num_workers} parallel workers for processing")
    
    # Extract all text segments in parallel
    all_segments = []
    processed_count = 0
    error_count = 0
    
    # Prepare arguments for parallel processing
    # Note: We need to pass bucket object, but it's not pickleable
    # So we'll use ThreadPoolExecutor instead of ProcessPoolExecutor for GCS operations
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_blob = {
            executor.submit(
                process_single_json_file,
                (bucket, blob_name, strategy, min_length, max_length)
            ): blob_name
            for bucket, blob_name in json_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_blob):
            blob_name = future_to_blob[future]
            try:
                filename, segments = future.result()
                all_segments.extend(segments)
                processed_count += 1
                print(f"  [{processed_count}/{len(json_files)}] Processed {filename}: {len(segments)} segments")
            except Exception as e:
                error_count += 1
                filename = blob_name.split('/')[-1]
                print(f"  Error processing {filename}: {e}")
    
    if error_count > 0:
        print(f"\nWarning: {error_count} files had errors during processing")
    
    if not all_segments:
        raise ValueError("No text segments extracted from JSON files")
    
    print(f"\nTotal segments extracted: {len(all_segments)}")
    
    # Shuffle and split into train/dev
    random.seed(seed)
    random.shuffle(all_segments)
    
    split_idx = int(len(all_segments) * train_split)
    train_segments = all_segments[:split_idx]
    dev_segments = all_segments[split_idx:]
    
    print(f"Train segments: {len(train_segments)}")
    print(f"Dev segments: {len(dev_segments)}")
    
    # Write to CSV files locally
    train_csv = os.path.join(output_dir, "train.csv")
    dev_csv = os.path.join(output_dir, "dev.csv")
    
    # Write train.csv
    with open(train_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for segment, metadata in train_segments:
            # Escape quotes in text for CSV
            text = segment.replace('"', '""')
            writer.writerow([f'"{text}"', label_value])
    
    # Write dev.csv
    with open(dev_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for segment, metadata in dev_segments:
            text = segment.replace('"', '""')
            writer.writerow([f'"{text}"', label_value])
    
    print(f"\nCreated {train_csv}")
    print(f"Created {dev_csv}")
    
    return train_csv, dev_csv


def create_and_upload_dataset(
    bucket_name: str,
    prefixes: List[str],
    output_dir: str = ".",
    strategy: str = "sentences",
    train_split: float = 0.8,
    min_length: int = 50,
    max_length: int = 512,
    label_value: int = 0,
    seed: int = 42,
    gcp_project: Optional[str] = None,
    max_files: Optional[int] = None,
    gcs_prefix: str = "fine_tuning",
    upload_to_gcs: bool = True,
    num_workers: Optional[int] = None
) -> Tuple[str, str]:
    """
    Create train.csv and dev.csv from transcript JSON files in GCS and upload to GCS.
    
    This is a convenience wrapper that calls create_dataset_from_transcripts and uploads.
    """
    # Create dataset locally
    train_csv, dev_csv = create_dataset_from_transcripts(
        bucket_name=bucket_name,
        prefixes=prefixes,
        output_dir=output_dir,
        strategy=strategy,
        train_split=train_split,
        min_length=min_length,
        max_length=max_length,
        label_value=label_value,
        seed=seed,
        gcp_project=gcp_project,
        max_files=max_files,
        num_workers=num_workers
    )
    
    # Upload to GCS if requested
    if upload_to_gcs:
        gcs_train_path, gcs_dev_path = upload_csv_to_gcs(
            bucket_name=bucket_name,
            train_csv=train_csv,
            dev_csv=dev_csv,
            gcp_project=gcp_project,
            gcs_prefix=gcs_prefix
        )
    else:
        print("\nSkipping GCS upload (--no_upload flag set)")
    
    print(f"\n⚠️  NOTE: All segments are labeled as {label_value}")
    print(f"   You need to manually label them based on your classification task!")
    
    return train_csv, dev_csv


def main():
    parser = argparse.ArgumentParser(
        description="Create training dataset from earnings call transcript JSON files in GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process earnings announcements only:
  python create_dataset.py --bucket blacksmith-sec-filings --prefixes earnings-announcement-transcripts

  # Process multiple sources:
  python create_dataset.py --bucket blacksmith-sec-filings \\
      --prefixes earnings-announcement-transcripts fomc sec-10k form-4-filings

  # Limit number of files for testing:
  python create_dataset.py --bucket blacksmith-sec-filings \\
      --prefixes earnings-announcement-transcripts --max_files 10
        """
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="blacksmith-sec-filings",
        help="GCS bucket name (default: blacksmith-sec-filings)"
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        nargs="+",
        default=["earnings-announcement-transcripts", "fomc", "form-4-filings", "sec-10k"],
        help="List of prefix paths within bucket to search for JSON files "
             "(default: earnings-announcement-transcripts fomc form-4-filings sec-10k)"
    )
    parser.add_argument(
        "--gcp_project",
        type=str,
        default=None,
        help="GCP project ID (optional, uses default credentials if not provided)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output CSV files (default: current directory)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="sentences",
        choices=["sentences", "paragraphs", "qna", "full"],
        help="Text extraction strategy (default: sentences)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=50,
        help="Minimum character length for text segments (default: 50)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum character length for text segments (default: 512)"
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Initial label value for all segments (default: 0 - you'll need to label manually)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/dev split (default: 42)"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (optional, useful for testing)"
    )
    parser.add_argument(
        "--gcs_prefix",
        type=str,
        default="fine_tuning",
        help="GCS prefix path for uploading CSV files (default: fine_tuning)"
    )
    parser.add_argument(
        "--no_upload",
        action="store_true",
        help="Skip uploading files to GCS (keep local files only)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect, uses CPU count - 1 if > 4 CPUs)"
    )
    
    args = parser.parse_args()
    
    # Print CPU information
    total_cpus = mp.cpu_count()
    if args.workers:
        workers = args.workers
    else:
        workers = get_available_processors()
    
    print("=" * 60)
    print(f"CPU Information:")
    print(f"  Total CPUs: {total_cpus}")
    print(f"  Workers to use: {workers}")
    print("=" * 60)
    print()
    
    create_and_upload_dataset(
        bucket_name=args.bucket,
        prefixes=args.prefixes,
        output_dir=args.output_dir,
        strategy=args.strategy,
        train_split=args.train_split,
        min_length=args.min_length,
        max_length=args.max_length,
        label_value=args.label,
        seed=args.seed,
        gcp_project=args.gcp_project,
        max_files=args.max_files,
        gcs_prefix=args.gcs_prefix,
        upload_to_gcs=not args.no_upload,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
