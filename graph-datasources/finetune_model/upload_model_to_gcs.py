#!/usr/bin/env python3
"""
Upload fine-tuned model to GCS bucket.

Uploads the final fine-tuned model from ./fine-tuned-model/ to GCS bucket
gs://blacksmith-sec-filings/fine_tuning/ with folder name format: yyyy-mm-dd-hh-mm-ss-uuid.

Also maintains a metadata file (fine_tuning/models_metadata.json) that tracks:
- Model path
- Date created
- Model label

Each time a new model is uploaded, the script updates the 'latest:stable' label,
removing it from the previous model and applying it to the newly uploaded model.
"""

import os
import sys
import argparse
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage


# Configuration
GCS_BUCKET_NAME = "blacksmith-sec-filings"
GCS_BASE_PATH = "fine_tuning"
DEFAULT_MODEL_DIR = "./fine-tuned-model"
METADATA_FILE_PATH = "fine_tuning/models_metadata.json"
LATEST_STABLE_LABEL = "latest:stable"


def generate_folder_name() -> str:
    """
    Generate folder name in format: yyyy-mm-dd-hh-mm-ss-uuid
    
    Returns:
        Folder name string
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    unique_id = str(uuid.uuid4())
    return f"{timestamp}-{unique_id}"


def collect_model_files(model_path: Path, include_checkpoints: bool = False) -> List[Tuple[Path, str]]:
    """
    Collect all files from the fine-tuned model directory.
    
    Args:
        model_path: Path to the fine-tuned model directory
        include_checkpoints: Whether to include checkpoint directories
        
    Returns:
        List of (local_path, relative_path) tuples
    """
    files = []
    model_path = Path(model_path).resolve()
    
    if not model_path.exists():
        return files
    
    # Walk through all files in the model directory
    for root, dirs, filenames in os.walk(model_path):
        root_path = Path(root)
        
        # Skip checkpoint directories unless explicitly included
        if not include_checkpoints and 'checkpoint-' in str(root_path):
            continue
        
        # Skip .cache directories and other cache artifacts
        if '.cache' in str(root_path):
            continue
        
        for filename in filenames:
            file_path = root_path / filename
            
            # Skip hidden files, __pycache__, and .metadata files
            if (filename.startswith('.') or 
                '__pycache__' in str(file_path) or 
                filename.endswith('.metadata')):
                continue
            
            # Calculate relative path from model root
            try:
                relative_path = file_path.relative_to(model_path)
            except ValueError:
                continue
            
            files.append((file_path, str(relative_path)))
    
    return files


def check_gcs_files_exist(bucket: storage.Bucket, gcs_paths: List[str]) -> dict:
    """
    Check which files already exist in GCS.
    Returns dict mapping gcs_path -> (exists: bool, size: int or None)
    """
    print(f"🔍 Checking GCS bucket for existing files...", flush=True)
    existing = {}
    
    # Check files in batches to avoid too many API calls
    batch_size = 100
    for i in range(0, len(gcs_paths), batch_size):
        batch = gcs_paths[i:i + batch_size]
        for gcs_path in batch:
            blob = bucket.blob(gcs_path)
            try:
                blob.reload()
                existing[gcs_path] = (True, blob.size if blob.size is not None else 0)
            except Exception:
                existing[gcs_path] = (False, None)
    
    existing_count = sum(1 for exists, _ in existing.values() if exists)
    print(f"   Found {existing_count}/{len(gcs_paths)} files already in GCS", flush=True)
    return existing


def read_metadata_file(bucket: storage.Bucket, metadata_path: str) -> List[Dict[str, Any]]:
    """
    Read the models metadata file from GCS.
    
    Args:
        bucket: GCS bucket object
        metadata_path: Path to metadata file in bucket
        
    Returns:
        List of model metadata dictionaries
    """
    blob = bucket.blob(metadata_path)
    
    if not blob.exists():
        return []
    
    try:
        content = blob.download_as_text()
        metadata = json.loads(content)
        if not isinstance(metadata, list):
            # If it's not a list, wrap it or return empty
            return []
        return metadata
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  Warning: Failed to read metadata file: {e}. Starting with empty metadata.", flush=True)
        return []


def write_metadata_file(bucket: storage.Bucket, metadata_path: str, metadata: List[Dict[str, Any]]) -> None:
    """
    Write the models metadata file to GCS.
    
    Args:
        bucket: GCS bucket object
        metadata_path: Path to metadata file in bucket
        metadata: List of model metadata dictionaries
    """
    blob = bucket.blob(metadata_path)
    content = json.dumps(metadata, indent=2)
    blob.upload_from_string(content, content_type='application/json')


def update_model_metadata(
    bucket: storage.Bucket,
    metadata_path: str,
    gcs_model_path: str,
    model_label: Optional[str] = None
) -> None:
    """
    Update the models metadata file with the new model entry.
    Removes 'latest:stable' label from previous model and adds it to the new one.
    
    Args:
        bucket: GCS bucket object
        metadata_path: Path to metadata file in bucket
        gcs_model_path: Full GCS path to the uploaded model (gs://bucket/path/folder_name/)
        model_label: Optional label for the model (defaults to 'latest:stable')
    """
    if model_label is None:
        model_label = LATEST_STABLE_LABEL
    
    # Extract bucket name and path from gs:// URL
    if gcs_model_path.startswith("gs://"):
        path_parts = gcs_model_path[5:].split("/", 1)
        if len(path_parts) == 2:
            model_path = path_parts[1].rstrip("/")
        else:
            model_path = path_parts[0]
    else:
        model_path = gcs_model_path
    
    # Read existing metadata
    metadata = read_metadata_file(bucket, metadata_path)
    
    # Remove 'latest:stable' label from any existing models
    for model in metadata:
        if model.get("label") == LATEST_STABLE_LABEL:
            # Remove the label (set to None or empty string)
            model["label"] = None
    
    # Create new model entry
    new_model_entry = {
        "path": f"gs://{bucket.name}/{model_path}/",
        "date_created": datetime.now().isoformat(),
        "label": model_label
    }
    
    # Add new model to metadata
    metadata.append(new_model_entry)
    
    # Write updated metadata back to GCS
    write_metadata_file(bucket, metadata_path, metadata)
    
    print(f"📝 Updated metadata file: {metadata_path}", flush=True)
    print(f"   Added model: {new_model_entry['path']}", flush=True)
    print(f"   Label: {model_label}", flush=True)


def upload_model_to_gcs(
    model_dir: str,
    bucket_name: str,
    gcs_base_path: str,
    folder_name: Optional[str] = None,
    include_checkpoints: bool = False,
    gcp_project: Optional[str] = None,
    model_label: Optional[str] = None
) -> str:
    """
    Upload fine-tuned model to GCS.
    
    Args:
        model_dir: Local path to fine-tuned model directory
        bucket_name: GCS bucket name
        gcs_base_path: Base path in bucket (folder name)
        folder_name: Optional folder name (if not provided, generates one)
        include_checkpoints: Whether to include checkpoint directories
        gcp_project: Optional GCP project ID
        model_label: Optional label for the model (defaults to 'latest:stable')
        
    Returns:
        GCS path to uploaded model (gs://bucket/path/folder_name)
    """
    model_path = Path(model_dir).resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    # Generate folder name if not provided
    if folder_name is None:
        folder_name = generate_folder_name()
    
    print(f"\n{'='*80}")
    print(f"UPLOADING FINE-TUNED MODEL TO GCS")
    print(f"{'='*80}")
    print(f"📁 Model Directory: {model_path}")
    print(f"☁️  Bucket: {bucket_name}")
    print(f"📂 GCS Path: {gcs_base_path}/{folder_name}/")
    print(f"📦 Include Checkpoints: {include_checkpoints}")
    print()
    
    # Initialize GCS client
    print("🔌 Connecting to GCS...", flush=True)
    try:
        if gcp_project:
            storage_client = storage.Client(project=gcp_project)
        else:
            storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Verify bucket exists
        if not bucket.exists():
            raise ValueError(f"Bucket {bucket_name} does not exist")
        
        print(f"✅ Connected to bucket: {bucket_name}\n", flush=True)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to GCS: {e}") from e
    
    # Collect all files from the model directory
    print("📋 Collecting model files...", flush=True)
    model_files = collect_model_files(model_path, include_checkpoints=include_checkpoints)
    
    if not model_files:
        raise ValueError(f"No files found in model directory: {model_path}")
    
    print(f"   Found {len(model_files)} files\n", flush=True)
    
    # Construct GCS paths
    all_files = []
    for local_path, relative_path in model_files:
        gcs_path = f"{gcs_base_path}/{folder_name}/{relative_path}"
        all_files.append((local_path, gcs_path))
    
    # Check which files already exist in GCS
    gcs_paths = [gcs_path for _, gcs_path in all_files]
    existing_files = check_gcs_files_exist(bucket, gcs_paths)
    
    # Filter out files that already exist and match size
    files_to_upload = []
    skipped_count = 0
    
    for local_path, gcs_path in all_files:
        exists, gcs_size = existing_files.get(gcs_path, (False, None))
        
        if exists:
            # Verify file size matches
            local_size = local_path.stat().st_size
            if gcs_size is not None and gcs_size == local_size:
                skipped_count += 1
                continue
        
        files_to_upload.append((local_path, gcs_path))
    
    if skipped_count > 0:
        print(f"⏭️  Skipping {skipped_count} files that already exist in GCS\n", flush=True)
    
    if not files_to_upload:
        print("✅ All files already exist in GCS. Nothing to upload.\n", flush=True)
        gcs_model_path = f"gs://{bucket_name}/{gcs_base_path}/{folder_name}/"
        
        # Still update metadata even if all files already exist
        try:
            print(f"📝 Updating model metadata...", flush=True)
            update_model_metadata(
                bucket=bucket,
                metadata_path=METADATA_FILE_PATH,
                gcs_model_path=gcs_model_path,
                model_label=model_label
            )
        except Exception as e:
            print(f"⚠️  Warning: Failed to update metadata file: {e}", flush=True)
        
        return gcs_model_path
    
    # Upload files in parallel
    print(f"🚀 Uploading {len(files_to_upload)} files to GCS...", flush=True)
    
    max_workers = (os.cpu_count() or 4) * 2
    
    def upload_single_file(local_path: Path, gcs_path: str) -> Optional[Exception]:
        """Upload a single file to GCS with retry logic. Returns None on success, Exception on failure."""
        import time
        max_retries = 3
        retry_delay = 5  # seconds
        
        file_size = local_path.stat().st_size
        is_large_file = file_size > 100 * 1024 * 1024  # 100MB
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                blob = bucket.blob(gcs_path)
                
                # For large files (>100MB), use resumable upload with increased timeout
                if is_large_file:
                    blob.upload_from_filename(
                        str(local_path),
                        timeout=600,  # 10 minutes for large files
                        if_generation_match=None  # Allow overwrite
                    )
                else:
                    blob.upload_from_filename(str(local_path), timeout=300)  # 5 minutes for smaller files
                return None
            except Exception as e:
                last_exception = e
                # If it's a timeout and we have retries left, retry
                if attempt < max_retries - 1 and ('timeout' in str(e).lower() or 'timed out' in str(e).lower()):
                    wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                    print(f"   ⚠️  Retry {attempt + 1}/{max_retries} for {gcs_path} after {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                    continue
                # Otherwise, return the exception immediately
                return e
        
        # If all retries failed, return the last exception
        return last_exception if last_exception else Exception(f"Failed after {max_retries} attempts")
    
    # Upload files in parallel
    successful = 0
    failed = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_file = {
            executor.submit(upload_single_file, local_path, gcs_path): (local_path, gcs_path)
            for local_path, gcs_path in files_to_upload
        }
        
        # Process completed uploads
        completed = 0
        for future in as_completed(future_to_file):
            completed += 1
            local_path, gcs_path = future_to_file[future]
            result = future.result()
            
            if result is None:
                successful += 1
                if completed % 10 == 0:
                    print(f"   Progress: {completed}/{len(files_to_upload)} files uploaded...", flush=True)
            else:
                failed += 1
                failed_files.append((gcs_path, result))
    
    if failed > 0:
        print(f"\n⚠️  {failed} files failed to upload:", flush=True)
        for gcs_path, error in failed_files[:10]:  # Show first 10 errors
            print(f"   ❌ {gcs_path}: {error}", flush=True)
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more errors", flush=True)
    
    total_uploaded = successful + skipped_count
    gcs_model_path = f"gs://{bucket_name}/{gcs_base_path}/{folder_name}/"
    
    # Update metadata file if upload was successful (no failures)
    if failed == 0:
        try:
            print(f"\n📝 Updating model metadata...", flush=True)
            update_model_metadata(
                bucket=bucket,
                metadata_path=METADATA_FILE_PATH,
                gcs_model_path=gcs_model_path,
                model_label=model_label
            )
        except Exception as e:
            print(f"⚠️  Warning: Failed to update metadata file: {e}", flush=True)
            # Don't fail the entire upload if metadata update fails
    else:
        print(f"\n⚠️  Skipping metadata update due to upload failures", flush=True)
    
    print(f"\n{'='*80}")
    print(f"✅ UPLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Total in GCS: {total_uploaded}/{len(all_files)}")
    print(f"   - Uploaded: {successful}")
    print(f"   - Skipped (already exists): {skipped_count}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(all_files)}")
    print(f"📁 GCS Path: {gcs_model_path}")
    print(f"{'='*80}\n", flush=True)
    
    return gcs_model_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Upload fine-tuned model to GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload model with auto-generated folder name:
  python upload_model_to_gcs.py

  # Upload model with custom folder name:
  python upload_model_to_gcs.py --folder-name 2024-01-15-10-30-00-abc123

  # Upload model including checkpoints:
  python upload_model_to_gcs.py --include-checkpoints

  # Upload from custom model directory:
  python upload_model_to_gcs.py --model-dir ./my-model
        """
    )
    parser.add_argument(
        '--model-dir', '-m',
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f'Path to fine-tuned model directory (default: {DEFAULT_MODEL_DIR})'
    )
    parser.add_argument(
        '--bucket', '-b',
        type=str,
        default=GCS_BUCKET_NAME,
        help=f'GCS bucket name (default: {GCS_BUCKET_NAME})'
    )
    parser.add_argument(
        '--gcs-path',
        type=str,
        default=GCS_BASE_PATH,
        help=f'Base path in bucket (default: {GCS_BASE_PATH})'
    )
    parser.add_argument(
        '--folder-name', '-f',
        type=str,
        default=None,
        help='Folder name in GCS (default: auto-generated as yyyy-mm-dd-hh-mm-ss-uuid)'
    )
    parser.add_argument(
        '--include-checkpoints',
        action='store_true',
        help='Include checkpoint directories in upload'
    )
    parser.add_argument(
        '--gcp-project',
        type=str,
        default=None,
        help='Optional GCP project ID (uses default credentials if omitted)'
    )
    parser.add_argument(
        '--model-label',
        type=str,
        default=None,
        help=f'Label for the model in metadata (default: {LATEST_STABLE_LABEL})'
    )
    
    args = parser.parse_args()
    
    try:
        gcs_path = upload_model_to_gcs(
            model_dir=args.model_dir,
            bucket_name=args.bucket,
            gcs_base_path=args.gcs_path,
            folder_name=args.folder_name,
            include_checkpoints=args.include_checkpoints,
            gcp_project=args.gcp_project,
            model_label=args.model_label
        )
        print(f"✅ Model uploaded successfully to: {gcs_path}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to upload model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

