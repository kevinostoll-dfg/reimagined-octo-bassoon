#!/usr/bin/env python3
"""
Stream spaCy models from Hugging Face directly to GCS using transfer manager.

This script downloads spaCy models from Hugging Face and streams them directly
to GCS bucket without storing locally. Uses transfer_manager for fast parallel uploads.
"""

import os
import sys
import argparse
import json
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from google.cloud.storage import transfer_manager


# Configuration
GCS_BUCKET_NAME = "blacksmith-sec-filings"
GCS_BASE_PATH = "spacy-models"  # Folder in bucket

# Hugging Face model repositories for spaCy models
SPACY_MODELS = {
    # Official spaCy pipelines pinned to the 3.7.x line on Hugging Face (no cross-major fallback).
    # We validate the downloaded model's meta.json "version" to start with "3.7".
    "en_core_web_sm": {"repo": "spacy/en_core_web_sm", "spacy_version": "3.7"},
    "en_core_web_md": {"repo": "spacy/en_core_web_md", "spacy_version": "3.7"},
    "en_core_web_lg": {"repo": "spacy/en_core_web_lg", "spacy_version": "3.7"},
    "en_core_web_trf": {"repo": "spacy/en_core_web_trf", "spacy_version": "3.7"},
    # Non-spaCy-base models (leave unchanged)
    "sec_bert_shape": {"repo": "nlpaueb/sec-bert-shape", "spacy_version": None},
    "sec_bert_base": {"repo": "nlpaueb/sec-bert-base", "spacy_version": None},
}


def check_hf_cache(repo_id: str, token: Optional[str] = None, revision: Optional[str] = None) -> Optional[Path]:
    """
    Check if model exists in Hugging Face cache.
    Returns path to cached model if found, None otherwise.
    """
    try:
        from huggingface_hub import snapshot_download
        
        # Try to get model from cache only (no download)
        # If model is cached, this will return the cached path
        try:
            cached_path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                cache_dir=None,  # Use default HF cache
                local_files_only=True,  # Only check cache, don't download
                token=token,
                revision=revision
            )
            cached_path_obj = Path(cached_path)
            if cached_path_obj.exists():
                return cached_path_obj
            return None
        except (OSError, FileNotFoundError, Exception):
            # Model not in cache or cache check failed
            return None
    except ImportError:
        return None
    except Exception:
        return None


def validate_spacy_model_version(
    model_path: Path,
    model_name: str,
    expected_version: Optional[str],
    require_exact: bool = False,
) -> bool:
    """
    Validate downloaded spaCy model version using meta.json.
    - If require_exact is True, the version must equal expected_version.
    - Otherwise, the version must start with expected_version (major.minor prefix).
    """
    if not expected_version:
        return True
    meta_path = model_path / "meta.json"
    if not meta_path.exists():
        print(f"❌ {model_name}: meta.json not found for version verification", flush=True)
        return False
    try:
        meta = json.loads(meta_path.read_text())
        version = str(meta.get("version", "")).strip()
        if not version:
            print(f"❌ {model_name}: meta.json missing version field", flush=True)
            return False
        if require_exact:
            if version != expected_version:
                print(f"❌ {model_name}: expected version {expected_version}, found {version}", flush=True)
                return False
        else:
            if not version.startswith(f"{expected_version}.") and version != expected_version:
                print(f"❌ {model_name}: expected version starting with {expected_version}, found {version}", flush=True)
                return False
        print(f"✅ {model_name}: version {version} validated", flush=True)
        return True
    except Exception as e:
        print(f"❌ {model_name}: failed to read/validate meta.json: {e}", flush=True)
        return False


def download_model_from_hf(
    model_name: str,
    repo_id: str,
    token: Optional[str] = None,
    revision: Optional[str] = None
) -> Optional[Path]:
    """
    Download spaCy model from Hugging Face to a temporary directory.
    Validates local cache before downloading.
    Returns path to downloaded model directory or None if failed.
    """
    try:
        from huggingface_hub import snapshot_download
        
        # Check if model exists in Hugging Face cache first
        print(f"🔍 Checking local cache for {model_name}...", flush=True)
        cached_path = check_hf_cache(repo_id, token, revision)
        
        if cached_path and cached_path.exists():
            print(f"✅ Found {model_name} in local cache: {cached_path}", flush=True)
            # Copy from cache to temp directory to avoid modifying cache
            import shutil
            temp_dir = tempfile.mkdtemp(prefix=f"spacy_{model_name}_")
            temp_path = Path(temp_dir)
            # Copy entire directory contents to temp_path
            for item in cached_path.iterdir():
                dest = temp_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            print(f"✅ Copied {model_name} from cache to temporary directory", flush=True)
            return temp_path
        
        print(f"📥 Cache miss. Downloading {model_name} from Hugging Face ({repo_id})...", flush=True)
        
        # Create temporary directory for this model
        temp_dir = tempfile.mkdtemp(prefix=f"spacy_{model_name}_")
        temp_path = Path(temp_dir)
        
        # Download model snapshot
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            cache_dir=None,  # Don't use HF cache for temp directory
            local_dir=str(temp_path),
            token=token,
            local_dir_use_symlinks=False,
            revision=revision
        )
        
        print(f"✅ Downloaded {model_name} to temporary directory", flush=True)
        return Path(downloaded_path)
        
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub", flush=True)
        return None
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}", flush=True)
        return None


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
                # Try to reload blob to get metadata (this will fail if blob doesn't exist)
                blob.reload()
                # If reload succeeds, blob exists and we have size
                existing[gcs_path] = (True, blob.size if blob.size is not None else 0)
            except Exception:
                # Blob doesn't exist or reload failed
                existing[gcs_path] = (False, None)
    
    existing_count = sum(1 for exists, _ in existing.values() if exists)
    print(f"   Found {existing_count}/{len(gcs_paths)} files already in GCS", flush=True)
    return existing


def stream_model_from_hf_to_gcs(
    model_name: str,
    repo_id: str,
    bucket: storage.Bucket,
    gcs_base_path: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    expected_version: Optional[str] = None,
    require_exact_version: bool = False,
) -> bool:
    """
    Stream model files from Hugging Face directly to GCS.
    Downloads to temp directory first, then uploads to GCS.
    Validates GCS bucket before uploading.
    """
    # Download model to temporary directory
    model_path = download_model_from_hf(model_name, repo_id, token, revision)
    if not model_path or not model_path.exists():
        return False
    # Validate spaCy version (no fallback across major/minor)
    if expected_version:
        if not validate_spacy_model_version(
            model_path=model_path,
            model_name=model_name,
            expected_version=expected_version,
            require_exact=require_exact_version,
        ):
            # Abort and clean up
            import shutil
            shutil.rmtree(model_path, ignore_errors=True)
            return False
    
    try:
        # Collect all files from the downloaded model
        print(f"📋 Collecting files from {model_name}...", flush=True)
        files = []
        
        for root, dirs, filenames in os.walk(model_path):
            root_path = Path(root)
            
            # Skip .cache directories (Hugging Face cache artifacts)
            if '.cache' in str(root_path):
                continue
            
            for filename in filenames:
                file_path = root_path / filename
                
                # Skip hidden files, __pycache__, and .metadata files (cache artifacts)
                if (filename.startswith('.') or 
                    '__pycache__' in str(file_path) or 
                    filename.endswith('.metadata')):
                    continue
                
                # Calculate relative path from model root
                try:
                    relative_path = file_path.relative_to(model_path)
                except ValueError:
                    continue
                
                # Construct GCS path
                gcs_path = f"{gcs_base_path}/{model_name}/{relative_path}"
                files.append((file_path, gcs_path))
        
        if not files:
            print(f"⚠️  No files found in {model_name}", flush=True)
            return False
        
        print(f"   Found {len(files)} files\n", flush=True)
        
        # Check which files already exist in GCS
        gcs_paths = [gcs_path for _, gcs_path in files]
        existing_files = check_gcs_files_exist(bucket, gcs_paths)
        
        # Filter out files that already exist and match size
        files_to_upload = []
        skipped_count = 0
        size_mismatch_count = 0
        
        for local_path, gcs_path in files:
            exists, gcs_size = existing_files.get(gcs_path, (False, None))
            
            if exists:
                # Verify file size matches
                local_size = local_path.stat().st_size
                if gcs_size is not None and gcs_size == local_size:
                    skipped_count += 1
                    continue
                else:
                    # File exists but size doesn't match - re-upload
                    size_mismatch_count += 1
                    if gcs_size is None:
                        print(f"   ⚠️  {gcs_path}: exists but size unknown, will re-upload", flush=True)
                    else:
                        print(f"   ⚠️  {gcs_path}: size mismatch (GCS: {gcs_size}, local: {local_size}), will re-upload", flush=True)
            
            files_to_upload.append((local_path, gcs_path))
        
        if skipped_count > 0:
            print(f"⏭️  Skipping {skipped_count} files that already exist in GCS", flush=True)
        if size_mismatch_count > 0:
            print(f"⚠️  Re-uploading {size_mismatch_count} files due to size mismatch", flush=True)
        
        if not files_to_upload:
            print(f"✅ {model_name}: All files already exist in GCS. Nothing to upload.\n", flush=True)
            return True
        
        if skipped_count > 0 or size_mismatch_count > 0:
            print()  # Add blank line for readability
        
        # Upload files using ThreadPoolExecutor for parallel uploads with custom destinations
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
                        # Use resumable upload with longer timeout for large files
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
        print(f"✅ {model_name}: {total_uploaded}/{len(files)} files in GCS ({successful} uploaded, {skipped_count} skipped)\n", flush=True)
        return successful > 0 or skipped_count > 0
        
    finally:
        # Clean up temporary directory
        import shutil
        try:
            shutil.rmtree(model_path, ignore_errors=True)
            print(f"🧹 Cleaned up temporary directory for {model_name}", flush=True)
        except Exception as e:
            print(f"⚠️  Failed to clean up temp directory: {e}", flush=True)


def find_spacy_models() -> List[Tuple[str, Path]]:
    """
    Find installed spaCy models.
    Returns list of (model_name, model_path) tuples.
    """
    models = []
    
    # Try to import spacy to find where models are installed
    try:
        import spacy
        import site
        
        # Get all site-packages directories
        site_packages = site.getsitepackages()
        if hasattr(site, 'getsitepackages'):
            user_site = site.getusersitepackages()
            if user_site:
                site_packages.append(user_site)
        
        # Also check current environment
        import sysconfig
        site_packages.append(sysconfig.get_path('purelib'))
        
        # Look for spaCy models in site-packages
        for site_pkg in site_packages:
            if not site_pkg or not os.path.exists(site_pkg):
                continue
            
            # Check for en_core_web_sm
            sm_path = Path(site_pkg) / "en_core_web_sm"
            if sm_path.exists() and sm_path.is_dir():
                models.append(("en_core_web_sm", sm_path))
                print(f"✅ Found en_core_web_sm at: {sm_path}")
            
            # Check for en_core_web_trf
            trf_path = Path(site_pkg) / "en_core_web_trf"
            if trf_path.exists() and trf_path.is_dir():
                models.append(("en_core_web_trf", trf_path))
                print(f"✅ Found en_core_web_trf at: {trf_path}")
        
        # Also try direct import to get model path
        try:
            import en_core_web_sm
            sm_path = Path(en_core_web_sm.__file__).parent
            if sm_path.exists():
                models.append(("en_core_web_sm", sm_path))
        except ImportError:
            pass
        
        try:
            import en_core_web_trf
            trf_path = Path(en_core_web_trf.__file__).parent
            if trf_path.exists():
                models.append(("en_core_web_trf", trf_path))
        except ImportError:
            pass
        
    except ImportError:
        print("⚠️  spaCy not installed. Cannot find models.")
        return []
    
    # Deduplicate models (keep first occurrence)
    seen = set()
    unique_models = []
    for model_name, model_path in models:
        if str(model_path) not in seen:
            seen.add(str(model_path))
            unique_models.append((model_name, model_path))
    
    return unique_models


def collect_model_files(model_path: Path) -> List[Tuple[Path, str]]:
    """
    Collect all files from a model directory.
    Returns list of (local_path, relative_path) tuples.
    """
    files = []
    model_path = Path(model_path).resolve()
    
    if not model_path.exists():
        return files
    
    # Walk through all files in the model directory
    for root, dirs, filenames in os.walk(model_path):
        root_path = Path(root)
        
        # Skip .cache directories (Hugging Face cache artifacts)
        if '.cache' in str(root_path):
            continue
        
        for filename in filenames:
            file_path = root_path / filename
            
            # Skip hidden files, __pycache__, and .metadata files (cache artifacts)
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


def upload_models_from_hf_to_gcs(
    model_names: List[str],
    bucket_name: str,
    gcs_base_path: str,
    hf_token: Optional[str] = None,
    spacy_version: Optional[str] = None
) -> None:
    """
    Stream spaCy models from Hugging Face directly to GCS.
    
    Args:
        model_names: List of model names to download (e.g., ["en_core_web_sm", "en_core_web_trf"])
        bucket_name: GCS bucket name
        gcs_base_path: Base path in bucket (folder name)
        hf_token: Optional Hugging Face token for private models
    """
    if not model_names:
        print("❌ No models specified")
        return
    
    print(f"\n{'='*80}")
    print(f"STREAMING SPACY MODELS FROM HUGGING FACE TO GCS")
    print(f"{'='*80}")
    print(f"☁️  Bucket: {bucket_name}")
    print(f"📁 GCS Path: {gcs_base_path}/")
    print(f"📦 Models: {', '.join(model_names)}")
    print()
    
    # Initialize GCS client
    print("🔌 Connecting to GCS...", flush=True)
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Verify bucket exists
        if not bucket.exists():
            print(f"❌ Bucket {bucket_name} does not exist")
            return
        
        print(f"✅ Connected to bucket: {bucket_name}\n", flush=True)
    except Exception as e:
        print(f"❌ Failed to connect to GCS: {e}", flush=True)
        return
    
    # Stream each model from HF to GCS
    results = {}
    for model_name in model_names:
        if model_name not in SPACY_MODELS:
            print(f"⚠️  Unknown model: {model_name}. Skipping.", flush=True)
            print(f"   Available models: {', '.join(SPACY_MODELS.keys())}", flush=True)
            results[model_name] = False
            continue
        
        model_info = SPACY_MODELS[model_name]
        repo_id = model_info["repo"]
        revision = None
        expected_version = spacy_version or model_info.get("spacy_version")
        require_exact = bool(spacy_version)
        success = stream_model_from_hf_to_gcs(
            model_name=model_name,
            repo_id=repo_id,
            bucket=bucket,
            gcs_base_path=gcs_base_path,
            token=hf_token,
            revision=revision,
            expected_version=expected_version,
            require_exact_version=require_exact,
        )
        results[model_name] = success
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ UPLOAD COMPLETE")
    print(f"{'='*80}")
    for model_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model_name}")
    print(f"📁 GCS Path: gs://{bucket_name}/{gcs_base_path}/")
    print(f"{'='*80}\n", flush=True)


def upload_models_to_gcs(
    models: List[Tuple[str, Path]],
    bucket_name: str,
    gcs_base_path: str,
    max_workers: int = None
) -> None:
    """
    Upload spaCy models to GCS using transfer manager for fast parallel uploads.
    
    Args:
        models: List of (model_name, model_path) tuples
        bucket_name: GCS bucket name
        gcs_base_path: Base path in bucket (folder name)
        max_workers: Number of parallel workers (default: 2x CPU cores)
    """
    if not models:
        print("❌ No models found to upload")
        return
    
    print(f"\n{'='*80}")
    print(f"UPLOADING SPACY MODELS TO GCS")
    print(f"{'='*80}")
    print(f"☁️  Bucket: {bucket_name}")
    print(f"📁 GCS Path: {gcs_base_path}/")
    print(f"📦 Models: {len(models)}")
    print()
    
    # Initialize GCS client
    print("🔌 Connecting to GCS...", flush=True)
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Verify bucket exists
        if not bucket.exists():
            print(f"❌ Bucket {bucket_name} does not exist")
            return
        
        print(f"✅ Connected to bucket: {bucket_name}\n", flush=True)
    except Exception as e:
        print(f"❌ Failed to connect to GCS: {e}", flush=True)
        return
    
    # Collect all files from all models
    print("📋 Collecting model files...", flush=True)
    all_files = []  # List of (local_path, gcs_path) tuples
    
    for model_name, model_path in models:
        print(f"   Scanning {model_name}...", flush=True)
        model_files = collect_model_files(model_path)
        
        for local_path, relative_path in model_files:
            # Construct GCS path: spacy-models/en_core_web_sm/relative/path/to/file
            gcs_path = f"{gcs_base_path}/{model_name}/{relative_path}"
            all_files.append((local_path, gcs_path))
        
        print(f"      Found {len(model_files)} files", flush=True)
    
    if not all_files:
        print("❌ No files found to upload")
        return
    
    print(f"\n✅ Collected {len(all_files)} files total\n", flush=True)
    
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
            if gcs_size == local_size:
                skipped_count += 1
                continue
        
        files_to_upload.append((local_path, gcs_path))
    
    if skipped_count > 0:
        print(f"⏭️  Skipping {skipped_count} files that already exist in GCS\n", flush=True)
    
    if not files_to_upload:
        print("✅ All files already exist in GCS. Nothing to upload.\n", flush=True)
        return
    
    # Set max_workers if not specified (2x CPU cores for optimal performance)
    if max_workers is None:
        max_workers = (os.cpu_count() or 4) * 2
    
    print(f"🚀 Starting upload with {max_workers} parallel workers...\n", flush=True)
    
    # Prepare file list for transfer_manager
    # transfer_manager.upload_many_from_filenames expects:
    # - bucket
    # - filenames_and_destinations: List of (local_path, gcs_path) tuples
    filenames_and_destinations = [
        (str(local_path), gcs_path)
        for local_path, gcs_path in files_to_upload
    ]
    
    try:
        # Use transfer_manager for fast parallel uploads
        results = transfer_manager.upload_many_from_filenames(
            bucket,
            filenames_and_destinations,
            max_workers=max_workers,
            skip_if_exists=False  # Overwrite existing files
        )
        
        # Count successes and failures
        successful = sum(1 for result in results if result is None)
        failed = sum(1 for result in results if result is not None)
        
        if failed > 0:
            print(f"\n⚠️  Upload completed with {failed} errors:", flush=True)
            for i, result in enumerate(results):
                if result is not None:
                    local_path, gcs_path = filenames_and_destinations[i]
                    print(f"   ❌ {gcs_path}: {result}", flush=True)
        
        total_successful = successful + skipped_count
        print(f"\n{'='*80}")
        print(f"✅ UPLOAD COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Total in GCS: {total_successful}/{len(all_files)}")
        print(f"   - Uploaded: {successful}")
        print(f"   - Skipped (already exists): {skipped_count}")
        if failed > 0:
            print(f"❌ Failed: {failed}/{len(all_files)}")
        print(f"📁 GCS Path: gs://{bucket_name}/{gcs_base_path}/")
        print(f"{'='*80}\n", flush=True)
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Stream spaCy models from Hugging Face to GCS using transfer manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all available models from Hugging Face:
  python upload_to_gcs.py

  # Upload specific models:
  python upload_to_gcs.py --models en_core_web_sm en_core_web_trf

  # Upload to custom folder:
  python upload_to_gcs.py --folder my-spacy-models

  # Use Hugging Face token (for private models):
  python upload_to_gcs.py --hf-token YOUR_TOKEN
        """
    )
    parser.add_argument(
        '--bucket', '-b',
        type=str,
        default=GCS_BUCKET_NAME,
        help=f'GCS bucket name (default: {GCS_BUCKET_NAME})'
    )
    parser.add_argument(
        '--folder', '-f',
        type=str,
        default=GCS_BASE_PATH,
        help=f'Folder name in bucket (default: {GCS_BASE_PATH})'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        type=str,
        default=list(SPACY_MODELS.keys()),
        choices=list(SPACY_MODELS.keys()),
        help=f'Models to upload (default: all available: {", ".join(SPACY_MODELS.keys())})'
    )
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='Hugging Face token (optional, for private models or rate limits)'
    )
    parser.add_argument(
        '--spacy-version',
        type=str,
        default=None,
        help="spaCy model version to download (applies to spaCy official models). "
             "Default: use pinned per-model revisions (sm/md/lg=3.7.1, trf=3.7.3)"
    )
    parser.add_argument(
        '--from-installed',
        action='store_true',
        help='Upload from locally installed models instead of Hugging Face'
    )
    
    args = parser.parse_args()
    
    # Check if huggingface_hub is installed (unless using --from-installed)
    if not args.from_installed:
        try:
            import huggingface_hub
        except ImportError:
            print("❌ huggingface_hub not installed.", flush=True)
            print("   Install with: pip install huggingface_hub", flush=True)
            print("   Or use --from-installed to upload local models", flush=True)
            sys.exit(1)
    
    if args.from_installed:
        # Legacy mode: upload from installed models
        print("🔍 Searching for locally installed spaCy models...\n", flush=True)
        models = find_spacy_models()
        
        if not models:
            print("❌ No spaCy models found. Make sure models are installed:")
            print("   python -m spacy download en_core_web_sm")
            print("   python -m spacy download en_core_web_trf")
            sys.exit(1)
        
        print(f"\n✅ Found {len(models)} model(s): {', '.join([m[0] for m in models])}\n", flush=True)
        
        # Upload to GCS
        upload_models_to_gcs(
            models=models,
            bucket_name=args.bucket,
            gcs_base_path=args.folder,
            max_workers=None
        )
    else:
        # New mode: stream from Hugging Face
        # Get HF token from environment if not provided (check multiple env var names)
        hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN")
        
        upload_models_from_hf_to_gcs(
            model_names=args.models,
            bucket_name=args.bucket,
            gcs_base_path=args.folder,
            hf_token=hf_token,
            spacy_version=args.spacy_version
        )


if __name__ == "__main__":
    main()

