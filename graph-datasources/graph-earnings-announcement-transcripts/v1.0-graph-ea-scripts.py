#!/usr/bin/env python3
"""
Entity & Relationship Extraction Pipeline v2.4 (Production-Grade)
Extracts entities and relationships using spaCy with domain-agnostic patterns
Enhanced with LLM-based enrichment using Qwen3-Max via Novita.ai
Outputs Python data structures for knowledge graph construction

v2.4 ENHANCEMENTS:
- Speaker attribution with STATEMENT nodes + SAID relationships
- Role extraction (HAS_ROLE, WORKS_FOR)
- Metric normalization (METRIC_DEFINITION + SAME_AS)
- Enhanced causal relationship extraction (CAUSES, DRIVES, BOOSTS, HURTS)
- Lowered confidence threshold (0.62) for richer LLM relationships
- Financial-NLP optimized schema
"""

import os
import json
import time
import re
import logging
import argparse
import math
import torch
import threading
import atexit
from queue import Queue
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from gqlalchemy import Memgraph
from google.cloud import storage

# Load environment variables
load_dotenv()

# Initialize Novita client (OpenAI-compatible API)
novita_client = OpenAI(
    api_key=os.getenv("NOVITA_API_KEY"),  # Set in .env: NOVITA_API_KEY=your_key_here
    base_url="https://api.novita.ai/v3/openai"
)

# Reduce noise in logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# ------------------------------------------------------------------
# 1. Configuration from Environment
# ------------------------------------------------------------------
# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "earnings-announcement-transcripts")
SPACY_MODELS_BUCKET = os.getenv("SPACY_MODELS_BUCKET", "blacksmith-sec-filings")
SPACY_MODELS_GCS_PATH = os.getenv("SPACY_MODELS_GCS_PATH", "spacy-models")

# Fine-tuned model configuration (optional)
FINE_TUNED_MODEL_GCS_PATH = os.getenv("FINE_TUNED_MODEL_GCS_PATH", "")
FINE_TUNED_MODEL_NAME = os.getenv("FINE_TUNED_MODEL_NAME", "")

# Symbol, Year, Quarter - REQUIRED for processing
SYMBOL = os.getenv("SYMBOL", "").upper()
YEAR = os.getenv("YEAR", "")
QUARTER = os.getenv("QUARTER", "")

# Memgraph tuning
COMENTION_BATCH_SIZE = int(os.getenv("COMENTION_BATCH_SIZE", "500"))

# CPU utilization: use all cores except 1 by default
_cpu_total = os.cpu_count() or 2
CPU_WORKERS = max(1, int(os.getenv("CPU_WORKERS", max(_cpu_total - 1, 1))))
os.environ["OMP_NUM_THREADS"] = str(CPU_WORKERS)
os.environ["MKL_NUM_THREADS"] = str(CPU_WORKERS)
torch.set_num_threads(CPU_WORKERS)
torch.set_num_interop_threads(max(1, CPU_WORKERS // 2))
print(f"⚙️  CPU threads set to {CPU_WORKERS} (leaving 1 core free)", flush=True)

# Initialize GCS client (uses Application Default Credentials)
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Memgraph connection pool (per-process, reused across transcripts)
MEMGRAPH_POOL_SIZE = 20
_memgraph_pool: Optional[Queue] = None
_memgraph_pool_lock = threading.Lock()
_memgraph_pool_config: Optional[Dict] = None

# ------------------------------------------------------------------
# Fetch latest stable model from metadata
# ------------------------------------------------------------------
def get_latest_stable_model_path() -> Optional[Tuple[str, str]]:
    """
    Fetch models_metadata.json from GCS and find the model with label 'latest:stable'.
    Returns tuple of (gcs_path, model_name) or None if not found.
    The gcs_path is the path without the gs://bucket/ prefix.
    """
    try:
        metadata_bucket = storage_client.bucket(GCS_BUCKET_NAME)
        metadata_blob = metadata_bucket.blob("fine_tuning/models_metadata.json")
        
        if not metadata_blob.exists():
            print("ℹ️  models_metadata.json not found in GCS, using environment variables", flush=True)
            return None
        
        # Download and parse JSON
        metadata_content = metadata_blob.download_as_text()
        metadata = json.loads(metadata_content)
        
        # Find model with label "latest:stable"
        for model in metadata:
            if model.get("label") == "latest:stable":
                gcs_path = model.get("path", "")
                if not gcs_path:
                    continue
                
                # Parse GCS path: gs://bucket/path/ -> path/
                if gcs_path.startswith("gs://"):
                    # Remove gs:// prefix
                    path_without_prefix = gcs_path[5:]
                    # Remove bucket name and leading slash
                    if "/" in path_without_prefix:
                        bucket_name, path_part = path_without_prefix.split("/", 1)
                        # Ensure path ends with / for consistency
                        if not path_part.endswith("/"):
                            path_part += "/"
                        
                        # Try to determine model name from the directory structure
                        # The model name might be in a subdirectory
                        # For now, we'll use a default or try to infer from path
                        model_name = FINE_TUNED_MODEL_NAME if FINE_TUNED_MODEL_NAME else "fine-tuned-model"
                        
                        print(f"✅ Found latest:stable model at: {path_part}", flush=True)
                        return (path_part.rstrip("/"), model_name)
        
        print("ℹ️  No model with label 'latest:stable' found in metadata, using environment variables", flush=True)
        return None
        
    except Exception as e:
        print(f"⚠️  Error fetching model metadata: {e}, using environment variables", flush=True)
        return None

# Auto-fetch latest stable model if not explicitly set
if not FINE_TUNED_MODEL_GCS_PATH:
    model_info = get_latest_stable_model_path()
    if model_info:
        FINE_TUNED_MODEL_GCS_PATH, FINE_TUNED_MODEL_NAME = model_info
        if not os.getenv("FINE_TUNED_MODEL_NAME"):
            # Try to infer model name from the directory
            # List files in the model directory to find the actual model name
            try:
                model_bucket = storage_client.bucket(GCS_BUCKET_NAME)
                blobs = list(model_bucket.list_blobs(prefix=FINE_TUNED_MODEL_GCS_PATH + "/", max_results=10))
                # Look for common spaCy model files or directories
                for blob in blobs:
                    # Check if there's a subdirectory that might be the model name
                    relative_path = blob.name[len(FINE_TUNED_MODEL_GCS_PATH) + 1:]
                    if "/" in relative_path:
                        potential_model_name = relative_path.split("/")[0]
                        # Check if it looks like a spaCy model (has meta.json or similar)
                        meta_blob = model_bucket.blob(f"{FINE_TUNED_MODEL_GCS_PATH}/{potential_model_name}/meta.json")
                        if meta_blob.exists():
                            FINE_TUNED_MODEL_NAME = potential_model_name
                            print(f"✅ Auto-detected model name: {FINE_TUNED_MODEL_NAME}", flush=True)
                            break
            except Exception as e:
                print(f"⚠️  Could not auto-detect model name: {e}", flush=True)

# Test mode: limits processing to first N sentences for quick testing
# Set TEST_MODE=true or MAX_SENTENCES=50 in .env
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
MAX_SENTENCES = int(os.getenv("MAX_SENTENCES", "0"))  # 0 = no limit

if TEST_MODE and MAX_SENTENCES == 0:
    MAX_SENTENCES = 50  # Default to 50 sentences in test mode

if TEST_MODE or MAX_SENTENCES > 0:
    print(f"🧪 TEST MODE: Processing limited to {MAX_SENTENCES} sentences", flush=True)
    print()

# Skip LLM processing: Set SKIP_LLM=true to skip all LLM enrichment
# Can be overridden by --skip-llm command-line argument
SKIP_LLM = os.getenv("SKIP_LLM", "false").lower() == "true"

# ------------------------------------------------------------------
# 2. Metric Normalization Configuration (v2.4)
# ------------------------------------------------------------------
# Canonical metric mapping for financial earnings calls
METRIC_CANONICAL_MAP = {
    # Revenue patterns
    "consolidated revenue growth": "Revenue Growth YoY",
    "total revenue growth": "Revenue Growth YoY",
    "revenue.*year-over-year": "Revenue Growth YoY",
    "revenue.*y/y": "Revenue Growth YoY",
    "revenue.*yoy": "Revenue Growth YoY",
    "revenues.*\\d+%": "Revenue Growth YoY",
    
    # Operating Expenses & Headcount
    "non-gaap operating expenses": "Non-GAAP Operating Expenses",
    "operating expenses": "Operating Expenses",
    "opex": "Operating Expenses",
    "head ?count": "Headcount",
    "employee.*growth": "Headcount Growth YoY",
    
    # Margins
    "operating margin": "Operating Margin",
    "gross margin": "Gross Margin",
    "profit margin": "Profit Margin",
    
    # Tax & EPS
    "effective tax rate": "Effective Tax Rate",
    "diluted eps": "Diluted EPS",
    "earnings per share": "Diluted EPS",
    
    # Cash Flow
    "operating cash flow": "Operating Cash Flow",
    "free cash flow": "Free Cash Flow",
    "fcf": "Free Cash Flow",
}

def normalize_metric(text: str) -> Tuple[str, float, str]:
    """
    Normalize metric text to canonical form
    Returns: (canonical_name, value, unit) or (None, None, None)
    """
    text_lower = text.lower()
    
    for pattern, canonical in METRIC_CANONICAL_MAP.items():
        if re.search(pattern, text_lower):
            # Extract numeric value if present
            value_match = re.search(r'(\d+(?:\.\d+)?)%?', text)
            value = float(value_match.group(1)) if value_match else None
            unit = "%" if value_match and "%" in text else None
            return canonical, value, unit
    
    return None, None, None

# ------------------------------------------------------------------
# 3. Role Detection Patterns (v2.4)
# ------------------------------------------------------------------
ROLE_PATTERNS = {
    'ceo': ['CEO', 'Chief Executive Officer'],
    'cfo': ['CFO', 'Chief Financial Officer'],
    'coo': ['COO', 'Chief Operating Officer'],
    'cto': ['CTO', 'Chief Technology Officer'],
    'president': ['President'],
    'vp': ['VP', 'Vice President'],
    'director': ['Director'],
    'head': ['Head of'],
}

def extract_role_from_text(text: str) -> str:
    """Extract role/title from text"""
    text_lower = text.lower()
    
    for role_key, role_variants in ROLE_PATTERNS.items():
        for variant in role_variants:
            if variant.lower() in text_lower:
                return variant
    
    return None

# ------------------------------------------------------------------
# 4. Download spaCy Model from GCS
# ------------------------------------------------------------------
def download_spacy_model_from_gcs(model_name: str) -> str:
    """
    Download spaCy model from GCS bucket to persistent cache directory.
    Returns path to downloaded model directory.
    MUST SUCCEED - no fallbacks.
    
    Args:
        model_name: Model name (e.g., "en_core_web_trf", "en_core_web_sm")
        
    Returns:
        Path to model directory
        
    Raises:
        RuntimeError: If download fails
    """
    import shutil
    import fcntl
    from pathlib import Path
    
    def _cache_is_complete(path: Path) -> bool:
        """Minimal completeness check to avoid races between workers."""
        meta = path / "meta.json"
        cfg = path / "config.json"
        weights = [path / "model.safetensors", path / "pytorch_model.bin"]
        if meta.exists():
            return True
        if cfg.exists() and any(w.exists() for w in weights):
            return True
        return False
    
    print(f"📥 Downloading {model_name} from GCS...", flush=True)
    start_time = time.time()
    
    # Check if model already exists locally
    try:
        import spacy
        # Try to find existing model
        try:
            model_path = spacy.util.find(model_name)
            if model_path and Path(model_path).exists():
                print(f"✅ {model_name} already exists locally: {model_path}", flush=True)
                return model_path
        except:
            pass
    except:
        pass
    
    # Use persistent cache directory (user's home directory)
    cache_base = Path.home() / ".cache" / "spacy_models_gcs"
    cache_base.mkdir(parents=True, exist_ok=True)
    model_cache_path = cache_base / model_name
    lock_path = cache_base / f"{model_name}.lock"
    
    # Acquire per-model lock to prevent concurrent partial downloads
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # Re-check cache under lock to avoid races
            if model_cache_path.exists() and _cache_is_complete(model_cache_path):
                print(f"✅ {model_name} found in cache: {model_cache_path}", flush=True)
                return str(model_cache_path)
            
            # Prepare clean directory for download
            if model_cache_path.exists():
                shutil.rmtree(model_cache_path)
            model_cache_path.mkdir(parents=True, exist_ok=True)
            
            try:
                # Initialize GCS client
                storage_client = storage.Client()
                bucket = storage_client.bucket(SPACY_MODELS_BUCKET)
                
                # Use fine-tuned model path if specified, otherwise use default spacy-models path
                is_fine_tuned = FINE_TUNED_MODEL_NAME and model_name == FINE_TUNED_MODEL_NAME and FINE_TUNED_MODEL_GCS_PATH
                if is_fine_tuned:
                    # Fine-tuned runs store artifacts under the timestamped folder.
                    # We will detect the spaCy model root (the directory that contains meta.json).
                    gcs_prefix = f"{FINE_TUNED_MODEL_GCS_PATH}/"
                else:
                    gcs_prefix = f"{SPACY_MODELS_GCS_PATH}/{model_name}/"
                
                # List all files in the model directory
                blobs = list(bucket.list_blobs(prefix=gcs_prefix))
                
                if not blobs:
                    raise RuntimeError(
                        f"Model {model_name} not found in GCS bucket {SPACY_MODELS_BUCKET} at path {gcs_prefix}"
                    )
                
                # For fine-tuned models, locate the folder that actually contains meta.json.
                target_prefix = gcs_prefix  # default to the original prefix
                if is_fine_tuned:
                    meta_roots = set()
                    for blob in blobs:
                        if blob.name.endswith("meta.json") and blob.name.startswith(gcs_prefix):
                            meta_root = Path(blob.name).parent.as_posix()
                            if not meta_root.endswith("/"):
                                meta_root += "/"
                            meta_roots.add(meta_root)
                    
                    if meta_roots:
                        # Prefer the shallowest root (closest to the prefix)
                        target_prefix = sorted(meta_roots, key=len)[0]
                        # Narrow blob list to the detected spaCy root
                        blobs = [b for b in blobs if b.name.startswith(target_prefix)]
                        print(f"   📁 Detected spaCy model root at gs://{SPACY_MODELS_BUCKET}/{target_prefix}", flush=True)
                    else:
                        print("   ℹ️  No meta.json detected under fine-tuned run; downloading full run prefix for inspection", flush=True)
                
                print(f"   Found {len(blobs)} files to download...", flush=True)
                
                # Download all files, preserving directory structure
                downloaded = 0
                for blob in blobs:
                    # Get relative path from chosen root (target_prefix)
                    relative_path = blob.name[len(target_prefix):]
                    if not relative_path:  # Skip the directory itself
                        continue
                    
                    # Create local file path
                    local_file = model_cache_path / relative_path
                    # Ensure parent directories exist before downloading
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file (create empty file first if needed for nested paths)
                    try:
                        blob.download_to_filename(str(local_file))
                    except FileNotFoundError:
                        # If parent directory doesn't exist, create it and retry
                        local_file.parent.mkdir(parents=True, exist_ok=True)
                        blob.download_to_filename(str(local_file))
                    downloaded += 1
                    
                    if downloaded % 10 == 0:
                        print(f"   Progress: {downloaded}/{len(blobs)} files downloaded...", flush=True)
                
                print(f"✅ Downloaded {downloaded} files in {time.time() - start_time:.2f}s", flush=True)
                
                # Model config is kept as-is from GCS
                # No patching needed - the model should work as downloaded
                
                print(f"✅ Model cached at: {model_cache_path}", flush=True)
                return str(model_cache_path)
                
            except Exception as e:
                # Clean up cache directory on error
                if model_cache_path.exists():
                    shutil.rmtree(model_cache_path, ignore_errors=True)
                raise RuntimeError(
                    f"Failed to download {model_name} from GCS bucket {SPACY_MODELS_BUCKET}: {str(e)}"
                ) from e
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(SPACY_MODELS_BUCKET)
        
        # Use fine-tuned model path if specified, otherwise use default spacy-models path
        is_fine_tuned = FINE_TUNED_MODEL_NAME and model_name == FINE_TUNED_MODEL_NAME and FINE_TUNED_MODEL_GCS_PATH
        if is_fine_tuned:
            # Fine-tuned runs store artifacts under the timestamped folder.
            # We will detect the spaCy model root (the directory that contains meta.json).
            gcs_prefix = f"{FINE_TUNED_MODEL_GCS_PATH}/"
        else:
            gcs_prefix = f"{SPACY_MODELS_GCS_PATH}/{model_name}/"
        
        # List all files in the model directory
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        
        if not blobs:
            raise RuntimeError(
                f"Model {model_name} not found in GCS bucket {SPACY_MODELS_BUCKET} at path {gcs_prefix}"
            )
        
        # For fine-tuned models, locate the folder that actually contains meta.json.
        target_prefix = gcs_prefix  # default to the original prefix
        if is_fine_tuned:
            meta_roots = set()
            for blob in blobs:
                if blob.name.endswith("meta.json") and blob.name.startswith(gcs_prefix):
                    meta_root = Path(blob.name).parent.as_posix()
                    if not meta_root.endswith("/"):
                        meta_root += "/"
                    meta_roots.add(meta_root)
            
            if meta_roots:
                # Prefer the shallowest root (closest to the prefix)
                target_prefix = sorted(meta_roots, key=len)[0]
                # Narrow blob list to the detected spaCy root
                blobs = [b for b in blobs if b.name.startswith(target_prefix)]
                print(f"   📁 Detected spaCy model root at gs://{SPACY_MODELS_BUCKET}/{target_prefix}", flush=True)
            else:
                print("   ℹ️  No meta.json detected under fine-tuned run; downloading full run prefix for inspection", flush=True)
        
        print(f"   Found {len(blobs)} files to download...", flush=True)
        
        # Download all files, preserving directory structure
        downloaded = 0
        for blob in blobs:
            # Get relative path from chosen root (target_prefix)
            relative_path = blob.name[len(target_prefix):]
            if not relative_path:  # Skip the directory itself
                continue
            
            # Create local file path
            local_file = model_cache_path / relative_path
            # Ensure parent directories exist before downloading
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file (create empty file first if needed for nested paths)
            try:
                blob.download_to_filename(str(local_file))
            except FileNotFoundError:
                # If parent directory doesn't exist, create it and retry
                local_file.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(local_file))
            downloaded += 1
            
            if downloaded % 10 == 0:
                print(f"   Progress: {downloaded}/{len(blobs)} files downloaded...", flush=True)
        
        print(f"✅ Downloaded {downloaded} files in {time.time() - start_time:.2f}s", flush=True)
        
        # Model config is kept as-is from GCS
        # No patching needed - the model should work as downloaded
        
        print(f"✅ Model cached at: {model_cache_path}", flush=True)
        return str(model_cache_path)
        
    except Exception as e:
        # Clean up cache directory on error
        if model_cache_path.exists():
            shutil.rmtree(model_cache_path, ignore_errors=True)
        raise RuntimeError(
            f"Failed to download {model_name} from GCS bucket {SPACY_MODELS_BUCKET}: {str(e)}"
        ) from e


# ------------------------------------------------------------------
# 5. Model format detection (spaCy vs Hugging Face checkpoint)
# ------------------------------------------------------------------
def detect_model_format(model_dir):
    """
    Determine whether the downloaded model directory is:
    - "spacy": contains meta.json
    - "huggingface": contains config.json and weights (model.safetensors or pytorch_model.bin)
    Raises RuntimeError if neither is satisfied.
    """
    from pathlib import Path
    
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise RuntimeError(f"Model directory does not exist: {model_dir}")
    
    meta_path = model_dir / "meta.json"
    cfg_path = model_dir / "config.json"
    weight_paths = [model_dir / "model.safetensors", model_dir / "pytorch_model.bin"]
    
    has_spacy = meta_path.exists()
    has_cfg = cfg_path.exists()
    has_weights = any(p.exists() for p in weight_paths)
    
    if has_spacy:
        return "spacy"
    
    if has_cfg and has_weights:
        try:
            import json as _json
            cfg = _json.loads(cfg_path.read_text())
            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures")
            if model_type or (isinstance(architectures, list) and architectures):
                return "huggingface"
        except Exception as e:
            raise RuntimeError(f"config.json is unreadable in {model_dir}: {type(e).__name__}: {e}") from e
        raise RuntimeError(
            f"config.json in {model_dir} is missing required HuggingFace keys (model_type or architectures)."
        )
    
    found = []
    if has_cfg:
        found.append("config.json")
    if has_weights:
        found.append("weights")
    found_str = ", ".join(found) if found else "no expected files"
    raise RuntimeError(
        f"Could not determine model format in {model_dir}. Found: {found_str}. "
        f"Expected meta.json for spaCy or weights+config.json for HuggingFace."
    )


# ------------------------------------------------------------------
# 6. Initialize spaCy Model
# ------------------------------------------------------------------
def initialize_spacy():
    """
    Initialize spaCy model with required pipeline components.
    Downloads model from GCS if not available locally.
    MUST SUCCEED - no fallbacks.
    """
    print("Loading spaCy model...", flush=True)
    start_time = time.time()
    
    # Use fine-tuned model if specified, otherwise default to transformer model
    if FINE_TUNED_MODEL_NAME:
        model_name = FINE_TUNED_MODEL_NAME
        print(f"🎯 Using fine-tuned model: {model_name}", flush=True)
    else:
        # Preferred model: transformer (best accuracy)
        model_name = "en_core_web_trf"
    
    # Import spacy
    import spacy
    
    # Try to load model first (might already be installed)
    try:
        nlp = spacy.load(model_name)
        print(f"✅ Using existing {model_name} installation", flush=True)
    except (OSError, IOError):
        # Model not found locally, download from GCS - MUST SUCCEED
        print(f"📥 {model_name} not found locally, downloading from GCS...", flush=True)
        model_path = download_spacy_model_from_gcs(model_name)
        
        from pathlib import Path
        model_path_abs = Path(model_path).resolve()
        
        # Determine model format
        model_format = detect_model_format(model_path_abs)
        
        if model_format == "spacy":
            # Load model from downloaded path using spacy.load()
            nlp = spacy.load(str(model_path_abs))
            print(f"✅ Loaded {model_name} from GCS as spaCy model", flush=True)
        elif model_format == "huggingface":
            # Build a spaCy pipeline that wraps the local Hugging Face checkpoint
            try:
                import spacy_transformers
                from transformers import AutoTokenizer
            except ImportError as e:
                raise RuntimeError(
                    "Required packages for Hugging Face integration not installed. "
                    "Install with: pip install transformers spacy-transformers torch"
                ) from e
            
            # Validate tokenizer presence
            tokenizer_path = model_path_abs
            if not (tokenizer_path / "tokenizer_config.json").exists() and not (tokenizer_path / "vocab.txt").exists():
                # Check common subdir
                sub_tok = tokenizer_path / "tokenizer"
                if (sub_tok / "tokenizer_config.json").exists() or (sub_tok / "vocab.txt").exists():
                    tokenizer_path = sub_tok
                else:
                    raise RuntimeError(
                        f"Tokenizer files not found in {model_path_abs}. "
                        f"Expected tokenizer_config.json or vocab.txt."
                    )
            # Quick tokenizer load sanity check
            try:
                _ = AutoTokenizer.from_pretrained(str(model_path_abs))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load tokenizer from {model_path_abs}. Error: {type(e).__name__}: {e}"
                ) from e
            
            # Build blank pipeline + transformer
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer", first=True)
            transformer_cfg = {
                "model": {
                    "@architectures": "spacy-transformers.TransformerModel.v3",
                    "name": str(model_path_abs),
                }
            }
            nlp.add_pipe("transformer", config=transformer_cfg)
            
            # Initialize to load weights/tokenizer
            nlp.initialize(lambda: [])
            print(f"✅ Loaded Hugging Face checkpoint into spaCy pipeline from {model_path_abs}", flush=True)
        else:
            raise RuntimeError(f"Unknown model format: {model_format}")
    
    # Ensure dependency parsing + POS tagging exist (required for noun_chunks)
    # If missing, pull trained components from en_core_web_sm (GCS is the official source)
    def _ensure_dep_components():
        needs_dep = ('parser' not in nlp.pipe_names) or ('tagger' not in nlp.pipe_names)
        if not needs_dep:
            return
        
        import spacy
        from pathlib import Path
        
        def load_reference_model():
            ref_name = "en_core_web_sm"
            try:
                print("🔍 Loading en_core_web_sm for dependency parser/tagger...", flush=True)
                return spacy.load(ref_name)
            except (OSError, IOError):
                print("📥 en_core_web_sm not found locally, downloading from GCS...", flush=True)
                ref_path = download_spacy_model_from_gcs(ref_name)
                ref_abs = Path(ref_path).resolve()
                ref_nlp = spacy.load(str(ref_abs))
                print(f"✅ Loaded en_core_web_sm from GCS at {ref_abs}", flush=True)
                return ref_nlp
        
        ref_nlp = load_reference_model()
        
        def add_component(name: str, after: str):
            if name in nlp.pipe_names:
                return
            if name not in ref_nlp.pipe_names:
                raise RuntimeError(f"Reference model missing expected component '{name}'")
            nlp.add_pipe(name, source=ref_nlp, after=after)
            print(f"✅ Added '{name}' from en_core_web_sm", flush=True)
        
        # Preserve ordering relative to transformer
        add_component('tagger', after='transformer')
        add_component('parser', after='tagger')
        add_component('attribute_ruler', after='parser')
        add_component('lemmatizer', after='attribute_ruler')
        add_component('ner', after='lemmatizer')
        
        if 'parser' not in nlp.pipe_names:
            raise RuntimeError("Dependency parser still missing after adding components")
        print(f"⚙️  Dependency parsing enabled; active pipes: {', '.join(nlp.pipe_names)}", flush=True)
    
    _ensure_dep_components()
    
    # Ensure we have all needed components
    has_sentencizer = 'sentencizer' in nlp.pipe_names or 'senter' in nlp.pipe_names
    if not has_sentencizer:
        if 'ner' in nlp.pipe_names:
            nlp.add_pipe('sentencizer', before='ner')
        else:
            nlp.add_pipe('sentencizer', first=True)
        print("✅ Added 'sentencizer' component to pipeline", flush=True)
    
    # Keep all components - don't disable any
    print(f"🚀 Keeping all pipeline components (required for POS tagging in transformer models)", flush=True)
    
    load_time = time.time() - start_time
    print(f"✅ spaCy loaded: {model_name} (loaded in {load_time:.2f}s)", flush=True)
    print(f"   Active pipes: {', '.join(nlp.pipe_names)}\n", flush=True)
    
    return nlp

nlp = initialize_spacy()

# ------------------------------------------------------------------
# 3. Entity Clustering & Normalization
# ------------------------------------------------------------------
def normalize_entity(text: str) -> str:
    """Normalize entity text for comparison"""
    return text.strip().lower()

def cluster_entities(entities: List[Dict]) -> Dict[str, Dict]:
    """
    Cluster duplicate entities and assign canonical IDs
    Returns mapping of canonical_id -> entity_info
    """
    # Group entities by label type
    entities_by_type = defaultdict(list)
    for ent in entities:
        entities_by_type[ent['label']].append(ent)
    
    # Cluster within each type
    entity_clusters = {}
    entity_id_counter = 0
    
    for label_type, ent_list in entities_by_type.items():
        # Build clusters based on text matching
        clusters = []
        for ent in ent_list:
            normalized = normalize_entity(ent['text'])
            
            # Try to find existing cluster
            found_cluster = None
            for cluster in clusters:
                # Check if this entity matches any in the cluster
                for cluster_ent in cluster['members']:
                    cluster_norm = normalize_entity(cluster_ent)
                    
                    # Exact match or substring match (for name variants)
                    if (normalized == cluster_norm or 
                        normalized in cluster_norm or 
                        cluster_norm in normalized):
                        found_cluster = cluster
                        break
                if found_cluster:
                    break
            
            if found_cluster:
                found_cluster['members'].add(ent['text'])
                found_cluster['count'] += 1
            else:
                # Create new cluster
                clusters.append({
                    'members': {ent['text']},
                    'count': 1,
                    'label': label_type
                })
        
        # Create canonical entities from clusters
        for cluster in clusters:
            entity_id = f"E{entity_id_counter}"
            entity_id_counter += 1
            
            # Choose canonical name (longest version, usually most complete)
            canonical_name = max(cluster['members'], key=len)
            
            entity_clusters[entity_id] = {
                'id': entity_id,
                'canonical_name': canonical_name,
                'variants': list(cluster['members']),
                'label': cluster['label'],
                'mention_count': cluster['count']
            }
    
    return entity_clusters

# ------------------------------------------------------------------
# 4. Pre-process Text to Extract Speaker Patterns
# ------------------------------------------------------------------
def extract_speaker_patterns(text: str) -> List[Dict]:
    """
    Extract speaker attribution BEFORE sentence splitting
    Pattern: [Speaker Name]: [Statement]
    Returns list of speaker segments with character positions
    """
    speaker_segments = []
    
    # Pattern: Name (possibly with newline before): Statement
    # Handles cases like "Elon Musk: Thank you" and "\nMartin Viecha: Thank you"
    speaker_pattern = r'(?:^|\n)([A-Z][a-zA-Z\s\.\-]+?):\s*([^\n]+(?:\n(?![A-Z][a-zA-Z\s\.\-]+?:)[^\n]+)*)'
    
    for match in re.finditer(speaker_pattern, text, re.MULTILINE):
        speaker_name = match.group(1).strip()
        statement = match.group(2).strip()
        
        # Skip if it looks like a label rather than a person (e.g., "Operator:")
        # We'll keep it for now and filter later based on NER
        speaker_segments.append({
            'speaker': speaker_name,
            'statement': statement,
            'start_char': match.start(),
            'end_char': match.end()
        })
    
    return speaker_segments

def map_speakers_to_sentences(sentences_data: List[Dict], speaker_segments: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Map speaker segments to sentence IDs and create STATEMENT nodes
    
    v2.4: Returns (relationships, statements) where statements is a list of STATEMENT node data
    """
    relationships = []
    statements = []
    statement_id_counter = 0
    
    for speaker_seg in speaker_segments:
        speaker_name = speaker_seg['speaker']
        statement_text = speaker_seg['statement']
        start_char = speaker_seg['start_char']
        
        # Find sentences that overlap with this speaker segment
        matched_sentences = []
        for sent_data in sentences_data:
            if sent_data['start_char'] >= start_char and sent_data['start_char'] < speaker_seg['end_char']:
                matched_sentences.append(sent_data['sentence_id'])
        
        if matched_sentences:
            # Create STATEMENT node data
            statement_id = f"STMT_{statement_id_counter}"
            statement_id_counter += 1
            
            statements.append({
                'id': statement_id,
                'text': statement_text[:500],  # Limit statement length
                'speaker': speaker_name,
                'sentence_ids': matched_sentences,
                'start_char': start_char,
                'end_char': speaker_seg['end_char']
            })
            
            # Create SAID relationship
            relationships.append({
                'type': 'SAID',
                'source': speaker_name,
                'source_label': 'PERSON',
                'target': statement_id,  # Link to STATEMENT node ID
                'target_label': 'STATEMENT',
                'sentence_ids': matched_sentences,
                'method': 'speaker_attribution_v2.4'
            })
            
            # Also check for role in speaker intro
            role = extract_role_from_text(statement_text[:200])  # Check first 200 chars
            if role:
                relationships.append({
                    'type': 'HAS_ROLE',
                    'source': speaker_name,
                    'source_label': 'PERSON',
                    'target': role,
                    'target_label': 'ROLE',
                    'method': 'role_extraction_v2.4'
                })
    
    return relationships, statements

def extract_role_relationships(sent, entities: List[Dict]) -> List[Dict]:
    """
    Extract role/title relationships using patterns
    Patterns: "[Person], [Title]" or "[Title] [Person]" or "our [Title], [Person]"
    Args:
        sent: spaCy Span object (sentence)
        entities: List of entity dicts
    """
    relationships = []
    persons = [ent for ent in entities if ent['label'] == 'PERSON']
    
    # Common title indicators
    title_keywords = {
        'ceo', 'cfo', 'coo', 'president', 'director', 'manager', 'executive',
        'officer', 'head', 'chief', 'senior', 'vice', 'chairman', 'founder'
    }
    
    # Pattern 1: "Person, Title" (appositive)
    for ent in sent.ents:
        if ent.label_ == 'PERSON':
            # Get the doc from the span to access tokens by absolute index
            parent_doc = sent.doc
            # Look for comma after person name
            if ent.end < len(parent_doc) and parent_doc[ent.end].text == ',':
                # Get next few tokens
                title_tokens = []
                for i in range(ent.end + 1, min(ent.end + 8, len(parent_doc))):
                    if parent_doc[i].text in [',', '.', '\n']:
                        break
                    title_tokens.append(parent_doc[i].text)
                
                title_text = ' '.join(title_tokens).strip()
                # Check if it looks like a title
                if any(kw in title_text.lower() for kw in title_keywords):
                    relationships.append({
                        'type': 'HAS_ROLE',
                        'source': ent.text,
                        'source_label': 'PERSON',
                        'target': title_text,
                        'target_label': 'TITLE',
                        'method': 'pattern_appositive'
                    })
    
    # Pattern 2: "our/the [Title], [Person]"
    text_lower = sent.text.lower()
    for person_ent in persons:
        # Search for "our/the [title], [person]" pattern
        pattern = r'(?:our|the)\s+([a-z\s]+?),\s*' + re.escape(person_ent['text'].split()[0])
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        
        for match in matches:
            potential_title = match.group(1).strip()
            if any(kw in potential_title for kw in title_keywords):
                relationships.append({
                    'type': 'HAS_ROLE',
                    'source': person_ent['text'],
                    'source_label': 'PERSON',
                    'target': potential_title.title(),
                    'target_label': 'TITLE',
                    'method': 'pattern_possessive'
                })
    
    return relationships

# ------------------------------------------------------------------
# 5. Relationship Extraction - Dependency-Based
# ------------------------------------------------------------------
def extract_svo_triples(sent, entities: List[Dict]) -> List[Dict]:
    """
    Extract Subject-Verb-Object triples using dependency parsing
    Focus on business-relevant verbs
    Args:
        sent: spaCy Span object (sentence)
        entities: List of entity dicts
    """
    relationships = []
    
    # Business-relevant verbs (lemma forms)
    business_verbs = {
        'announce', 'launch', 'achieve', 'report', 'increase', 'decrease',
        'improve', 'build', 'open', 'close', 'acquire', 'sell', 'buy', 'invest',
        'deliver', 'produced', 'manufacture', 'reach', 'expect', 'plan', 'make',
        'introduce', 'ramp', 'start', 'finish', 'complete', 'accelerate', 'expand',
        'raise', 'reduce', 'settle', 'generate', 'spend', 'drive', 'exceed', 'grow',
        'see', 'show', 'add', 'continue', 'launch', 'release', 'end', 'turn'
    }
    
    # Find verb-based relationships
    for token in sent:
        if token.pos_ == 'VERB' and token.lemma_.lower() in business_verbs:
            # Find subject
            subject = None
            subject_token = None
            for child in token.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    # Get the full noun phrase
                    subject = ' '.join([t.text for t in child.subtree]).strip()
                    subject_token = child
                    break
            
            # If no subject found, check if pronoun (we, they, etc.)
            if not subject:
                for child in token.children:
                    if child.dep_ == 'nsubj' or (child.pos_ == 'PRON' and child.dep_ in ['nsubj', 'nsubjpass']):
                        subject = child.text
                        subject_token = child
                        break
            
            # Find object
            obj = None
            obj_token = None
            for child in token.children:
                if child.dep_ in ['dobj', 'attr', 'oprd', 'acomp']:
                    obj = ' '.join([t.text for t in child.subtree]).strip()
                    obj_token = child
                    break
                elif child.dep_ == 'prep':
                    # Handle prepositional objects
                    for pchild in child.children:
                        if pchild.dep_ == 'pobj':
                            obj = ' '.join([t.text for t in pchild.subtree]).strip()
                            obj_token = pchild
                            break
                    if obj:
                        break
            
            # Create relationship if we have both subject and object
            if subject and obj and len(subject) > 1 and len(obj) > 1:
                # Limit length for readability
                subject = subject[:100]
                obj = obj[:100]
                
                relationships.append({
                    'type': 'SVO_TRIPLE',
                    'source': subject,
                    'source_label': 'ENTITY',
                    'target': obj,
                    'target_label': 'ENTITY',
                    'verb': token.lemma_,
                    'verb_text': token.text,
                    'method': 'dependency_parse'
                })
    
    return relationships

def extract_temporal_links(sent, entities: List[Dict]) -> List[Dict]:
    """
    Link entities/events to temporal expressions in the sentence
    """
    relationships = []
    
    # Find time/date entities
    temporal_entities = [ent for ent in entities if ent['label'] in ['DATE', 'TIME']]
    
    if temporal_entities:
        # Link all other entities in sentence to temporal context
        non_temporal = [ent for ent in entities if ent['label'] not in ['DATE', 'TIME']]
        
        for temp_ent in temporal_entities:
            for other_ent in non_temporal:
                relationships.append({
                    'type': 'TEMPORAL_CONTEXT',
                    'source': other_ent['text'],
                    'source_label': other_ent['label'],
                    'target': temp_ent['text'],
                    'target_label': temp_ent['label'],
                    'method': 'temporal_cooccurrence'
                })
    
    return relationships

def extract_quantity_links(sent, entities: List[Dict]) -> List[Dict]:
    """
    Link quantities (money, percentages, numbers) to their associated concepts
    Args:
        sent: spaCy Span object (sentence)
        entities: List of entity dicts
    """
    relationships = []
    
    # Find quantity entities
    quantity_entities = [ent for ent in entities if ent['label'] in ['MONEY', 'PERCENT', 'CARDINAL', 'QUANTITY']]
    
    for qty_ent in quantity_entities:
        # Find the quantity token span in sentence
        matched_ent = None
        for ent in sent.ents:
            # Use normalized comparison
            if ent.text.strip() == qty_ent['text'].strip():
                matched_ent = ent
                break
            # Also try substring match for partial matches
            elif len(qty_ent['text']) > 3 and (qty_ent['text'].strip() in ent.text.strip() or ent.text.strip() in qty_ent['text'].strip()):
                matched_ent = ent
                break
        
        if matched_ent:
            # Look at the head/root of this entity
            head = matched_ent.root.head
            
            # Common patterns: "revenue of $X", "achieved $X", "increased by X%"
            if head.pos_ == 'NOUN':
                concept = head.text
                # Try to get fuller noun phrase by looking at the compound
                noun_phrase_tokens = [head.text]
                for child in head.children:
                    if child.dep_ in ['compound', 'amod']:
                        noun_phrase_tokens.insert(0, child.text)
                concept = ' '.join(noun_phrase_tokens)[:50]
                
                relationships.append({
                    'type': 'QUANTITY_OF',
                    'source': concept,
                    'source_label': 'CONCEPT',
                    'target': qty_ent['text'],
                    'target_label': qty_ent['label'],
                    'method': 'quantity_dependency'
                })
            elif head.pos_ == 'VERB':
                # Get the action
                action = head.lemma_
                
                relationships.append({
                    'type': 'QUANTITY_IN_ACTION',
                    'source': action,
                    'source_label': 'ACTION',
                    'target': qty_ent['text'],
                    'target_label': qty_ent['label'],
                    'method': 'quantity_dependency'
                })
    
    return relationships

# ------------------------------------------------------------------
# 6. Co-mention Network
# ------------------------------------------------------------------
def build_comention_network(sentences_data: List[Dict]) -> List[Dict]:
    """
    Build co-mention relationships: entities appearing together in sentences
    """
    relationships = []
    comention_counts = defaultdict(int)
    
    for sent_data in sentences_data:
        entities = sent_data['entities']
        
        # Create pairs of entities in same sentence
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                # Create sorted pair key for consistency
                pair_key = tuple(sorted([
                    (ent1['text'], ent1['label']),
                    (ent2['text'], ent2['label'])
                ]))
                comention_counts[pair_key] += 1
    
    # Convert to relationships
    for (ent1, ent2), count in comention_counts.items():
        relationships.append({
            'type': 'CO_MENTIONED',
            'source': ent1[0],
            'source_label': ent1[1],
            'target': ent2[0],
            'target_label': ent2[1],
            'count': count,
            'method': 'comention'
        })
    
    return relationships

# ------------------------------------------------------------------
# 7. Concept Filtering Configuration
# ------------------------------------------------------------------
# Option B: Minimum mention threshold for displaying concepts in summary
MIN_CONCEPT_MENTIONS = 2  # Only show concepts mentioned at least this many times

# Option A: Blacklist of generic/conversational root words to filter out
CONCEPT_ROOT_BLACKLIST = {
    'lot', 'line', 'question', 'call', 'time', 'day', 'thing', 'way', 'bit',
    'sense', 'kind', 'sort', 'type', 'couple', 'bunch', 'number', 'part',
    'side', 'end', 'point', 'fact', 'case', 'example', 'term', 'basis',
    'perspective', 'standpoint', 'context', 'regard', 'respect', 'view',
    'thanks', 'thank', 'hello', 'hi', 'good', 'great', 'nice', 'sure'
}

# Option D: Whitelist of valuable business/domain-specific root words
CONCEPT_ROOT_WHITELIST = {
    # Financial metrics
    'revenue', 'margin', 'profit', 'income', 'expense', 'cost', 'earnings',
    'ebitda', 'cash', 'debt', 'equity', 'valuation', 'growth', 'decline',
    'loss', 'gain', 'return', 'investment', 'roi', 'tac', 'capex', 'opex',
    
    # Business operations
    'customer', 'user', 'client', 'partner', 'supplier', 'vendor', 'employee',
    'headcount', 'capacity', 'production', 'deployment', 'launch', 'rollout',
    'acquisition', 'merger', 'expansion', 'optimization', 'efficiency',
    
    # Products & Services
    'product', 'service', 'platform', 'solution', 'offering', 'feature',
    'functionality', 'capability', 'technology', 'innovation', 'system',
    'infrastructure', 'software', 'hardware', 'application', 'tool',
    
    # Markets & Strategy
    'market', 'segment', 'vertical', 'industry', 'sector', 'region',
    'geography', 'competition', 'strategy', 'initiative', 'program',
    'opportunity', 'demand', 'supply', 'trend', 'adoption', 'penetration',
    
    # Performance & Quality
    'performance', 'quality', 'experience', 'engagement', 'retention',
    'conversion', 'monetization', 'impression', 'click', 'traffic',
    'volume', 'velocity', 'momentum', 'acceleration', 'improvement',
    
    # Technology & Innovation
    'machine', 'learning', 'artificial', 'intelligence', 'algorithm',
    'data', 'analytics', 'cloud', 'mobile', 'search', 'advertising',
    'content', 'video', 'streaming', 'network', 'device', 'interface'
}

def should_keep_concept(chunk) -> bool:
    """
    Determine if a noun chunk should be kept as a concept
    Combines Options A (blacklist) and D (whitelist)
    """
    root_lemma = chunk.root.lemma_.lower()
    
    # Whitelist overrides everything - if it's a valuable business term, keep it
    if root_lemma in CONCEPT_ROOT_WHITELIST:
        return True
    
    # Blacklist filters out generic/conversational terms
    if root_lemma in CONCEPT_ROOT_BLACKLIST:
        return False
    
    # Additional heuristics for quality
    # Filter out chunks that are mostly determiners/pronouns
    non_function_tokens = [t for t in chunk if t.pos_ not in ['DET', 'PRON', 'ADP', 'CONJ', 'PUNCT']]
    if len(non_function_tokens) < 2:
        return False
    
    return True

# ------------------------------------------------------------------
# 7. LLM-Based Entity Enrichment (Qwen3-Max via Novita.ai)
# ------------------------------------------------------------------
def enrich_entities_with_qwen3(sentences: List[str], batch_num: int = 0) -> List[Dict]:
    """
    Use Qwen3-Max to enrich raw sentences with custom entities, descriptions, and coreferences.
    Returns list of dicts with 'entities' key containing enriched entity data.
    """
    if not sentences:
        return []
    
    # Check if LLM processing is disabled
    if SKIP_LLM:
        return [{"entities": []} for _ in sentences]
    
    # Check if API key is set
    if not os.getenv("NOVITA_API_KEY"):
        print(f"⚠️  NOVITA_API_KEY not set, skipping LLM enrichment for batch {batch_num}", flush=True)
        return [{"entities": []} for _ in sentences]
    
    prompt = """You are an expert financial transcript analyst. Extract ALL important entities from the following sentences.

Include these labels:
- PERSON, ORG, PRODUCT, METRIC, EVENT, TECHNOLOGY, LOCATION, DATE
- Use CONCEPT for multi-word business terms not covered above

For each sentence, return entities as a JSON array. For each entity provide:
{
  "text": str (entity text),
  "label": str (one of above),
  "description": str (4-8 word explanation),
  "confidence": float (0.0-1.0)
}

Return a JSON object with "sentences" array, where each item has "entities" array.

Example format:
{
  "sentences": [
    {"entities": [{"text": "Tesla", "label": "ORG", "description": "electric vehicle manufacturer", "confidence": 0.95}]},
    {"entities": [{"text": "Model 3", "label": "PRODUCT", "description": "Tesla sedan model", "confidence": 0.9}]}
  ]
}

Sentences:
""" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])

    try:
        print(f"   🔄 Processing batch {batch_num} ({len(sentences)} sentences)...", flush=True)
        
        response = novita_client.chat.completions.create(
            model="qwen/qwen3-max",
            messages=[
                {"role": "system", "content": "You are a precise entity extraction system for earnings calls. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=8192,
            timeout=60.0,  # 60 second timeout
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Clean markdown code blocks if present
        if raw.startswith("```json"):
            raw = raw[7:-3].strip()
        elif raw.startswith("```"):
            raw = raw[3:-3].strip()
        
        data = json.loads(raw)
        
        # Extract sentences array
        sentences_data = data.get("sentences", [])
        
        # Ensure we have the right number of results
        result = []
        for i in range(len(sentences)):
            if i < len(sentences_data):
                result.append({"entities": sentences_data[i].get("entities", [])})
            else:
                result.append({"entities": []})
        
        print(f"   ✅ Batch {batch_num} completed", flush=True)
        return result
        
    except Exception as e:
        print(f"   ⚠️  Batch {batch_num} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
        # Return empty entities for all sentences
        return [{"entities": []} for _ in sentences]

# ------------------------------------------------------------------
# 8. LLM-Based Relationship Enrichment (v2.4 Enhanced)
# ------------------------------------------------------------------
def enrich_relationships_with_qwen3(sentences_data: List[Dict], entity_clusters: Dict) -> List[Dict]:
    """
    Extract rich relationships: causal, comparative, sentiment, events, etc.
    Uses Qwen3-Max to find semantic relationships beyond syntactic patterns.
    
    v2.4 ENHANCEMENTS:
    - Lowered confidence threshold to 0.62
    - Added DRIVES, BOOSTS, HURTS, MITIGATES, RESULTS_IN relationship types
    - Improved prompt for financial earnings calls
    - Process more sentences (30 vs 20)
    """
    # Check if LLM processing is disabled
    if SKIP_LLM:
        return []
    
    # Check if API key is set
    if not os.getenv("NOVITA_API_KEY"):
        print("⚠️  NOVITA_API_KEY not set - skipping LLM relationship enrichment", flush=True)
        return []
    
    # Sample 30 most entity-rich sentences (increased from 20)
    rich_sentences = sorted(
        sentences_data,
        key=lambda x: len(x['entities']),
        reverse=True
    )[:30]
    
    if not rich_sentences:
        return []
    
    texts = [s['text'] for s in rich_sentences]
    
    print(f"   Analyzing {len(texts)} entity-rich sentences...", flush=True)
    
    # v2.4 ENHANCED PROMPT - Financial earnings call optimized
    prompt = """You are an expert financial analyst extracting directed causal relationships from earnings call transcripts.

Extract ALL directed relationships in this exact JSON format:

{
  "relationships": [
    {
      "type": "CAUSES|DRIVES|RESULTS_IN|BOOSTS|HURTS|MITIGATES|LEADS_TO|TARGETS|OUTPERFORMED|POSITIVE_ABOUT|NEGATIVE_ABOUT|EVENT_INVOLVES",
      "source": "exact entity/phrase that is the cause or driver",
      "target": "exact entity/phrase that is the effect or outcome",
      "verb": "single verb used (e.g., drove, offset, boosted, hurt)",
      "quote": "exact snippet proving the relationship (max 120 chars)",
      "sentiment": "positive|negative|neutral",
      "confidence": 0.XX
    }
  ]
}

Relationship Types Guide:
- CAUSES/DRIVES: Direct causation (e.g., "strong demand drove revenue growth")
- RESULTS_IN: Outcome relationship
- BOOSTS/HURTS: Impact direction (positive/negative)
- MITIGATES: Offsetting effect
- TARGETS: Forward-looking guidance
- OUTPERFORMED: Comparative performance
- POSITIVE_ABOUT/NEGATIVE_ABOUT: Sentiment
- EVENT_INVOLVES: Major events (launches, acquisitions)

Rules:
- Only output valid JSON
- Use exact phrases from text as source/target
- Do NOT hallucinate entities
- Confidence >= 0.60 required
- If no clear relationship → return empty array

Sentences:
""" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])

    try:
        response = novita_client.chat.completions.create(
            model="qwen/qwen3-max",
            messages=[
                {"role": "system", "content": "You are a financial relationship extraction expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lowered from 0.4 for more consistent extraction
            max_tokens=8192,
            timeout=90.0,  # Increased to 90s for larger batch
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Clean markdown code blocks
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        
        data = json.loads(raw)
        rels = data.get("relationships", [])
        
        # Convert to standard format with v2.4 enhancements
        enriched = []
        for r in rels:
            confidence = r.get("confidence", 0.7)
            
            # v2.4: Lower threshold from 0.8 to 0.62
            if confidence >= 0.62:
                enriched.append({
                    "type": r.get("type", "LLM_REL"),
                    "source": r.get("source", "Unknown"),
                    "source_label": "ENTITY",
                    "target": r.get("target", "Unknown"),
                    "target_label": "ENTITY",
                    "verb": r.get("verb", ""),
                    "quote": r.get("quote", "")[:200],
                    "sentiment": r.get("sentiment", "neutral"),
                    "confidence": confidence,
                    "method": "qwen3_novita_v2.4"
                })
        
        print(f"   ✅ Extracted {len(enriched)} relationships (confidence >= 0.62)", flush=True)
        return enriched
        
    except Exception as e:
        print(f"   ⚠️  LLM relationship enrichment error: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return []

# ------------------------------------------------------------------
# 9. Single-Pass Sentence & Entity Extraction
# ------------------------------------------------------------------
def extract_sentences_and_entities(full_text: str) -> List[Dict]:
    """
    Extract sentences AND entities in a single pass through the document
    Enhanced with LLM-based entity enrichment using Qwen3-Max
    Returns sentences with entities (spaCy NER + noun chunks + LLM enrichment)
    
    Concept Quality Filtering:
    - Option A: Blacklist generic/conversational root words (e.g., 'lot', 'line', 'question')
    - Option D: Whitelist valuable business/domain root words (overrides blacklist)
    - Multi-word phrases only (2-8 tokens)
    - No overlap with NER entities
    - Note: Option B (minimum mentions) is applied during display, not extraction
    """
    print("📊 Pass 1a: Processing document with spaCy...", flush=True)
    start_time = time.time()
    
    # Process the entire document ONCE
    doc = nlp(full_text)
    
    processing_time = time.time() - start_time
    print(f"✅ spaCy processing completed in {processing_time:.2f}s\n", flush=True)
    
    # Extract sentences with their spaCy entities
    print("📊 Pass 1b: Extracting spaCy entities and concepts...", flush=True)
    sentences = []
    concept_stats = {
        'total_chunks': 0,
        'filtered_chunks': 0,
        'filtered_by_quality': 0,  # Filtered by blacklist/whitelist
        'filtered_by_overlap': 0,   # Filtered by NER overlap
        'added_concepts': 0
    }
    
    sentence_texts = []  # Collect for batch LLM processing
    
    for i, sent in enumerate(doc.sents):
        sentence_text = sent.text.strip()
        
        # Skip empty sentences
        if not sentence_text:
            continue
        
        # Test mode: limit number of sentences
        if MAX_SENTENCES > 0 and len(sentences) >= MAX_SENTENCES:
            print(f"\n🧪 Test mode limit reached: stopping at {MAX_SENTENCES} sentences\n", flush=True)
            break
        
        sentence_texts.append(sentence_text)
        
        # Extract named entities
        spacy_entities = []
        entity_spans = set()  # Track covered character spans to avoid duplicates
        
        for ent in sent.ents:
            entity_text = ent.text.strip()
            if len(entity_text) > 1:  # Filter out single-character entities
                spacy_entities.append({
                    "text": entity_text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "source": "spacy"
                })
                entity_spans.add((ent.start_char, ent.end_char))
        
        # Extract noun chunks as concepts (filtering out NER entities)
        for chunk in sent.noun_chunks:
            concept_stats['total_chunks'] += 1
            chunk_text = chunk.text.strip()
            
            # Filter criteria:
            # 1. At least 2 tokens (multi-word concepts are more valuable)
            # 2. Not too long (avoid full clauses)
            # 3. Passes quality filters (Options A & D)
            if (len(chunk) >= 2 and 
                len(chunk) <= 8 and
                should_keep_concept(chunk)):  # Quality filter (blacklist/whitelist)
                
                # Check: not already covered by NER or overlapping
                if (chunk.start_char, chunk.end_char) in entity_spans:
                    concept_stats['filtered_by_overlap'] += 1
                    continue
                
                # Additional check: ensure chunk isn't substring of an existing entity
                overlaps = False
                for (start, end) in entity_spans:
                    # Check if chunks overlap
                    if not (chunk.end_char <= start or chunk.start_char >= end):
                        overlaps = True
                        break
                
                if not overlaps:
                    spacy_entities.append({
                        "text": chunk_text,
                        "label": "CONCEPT",
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "source": "spacy"
                    })
                    concept_stats['added_concepts'] += 1
                else:
                    concept_stats['filtered_by_overlap'] += 1
            else:
                # Filtered by size or quality
                if len(chunk) < 2 or len(chunk) > 8:
                    concept_stats['filtered_chunks'] += 1
                else:
                    concept_stats['filtered_by_quality'] += 1
        
        sentences.append({
            "sentence_id": len(sentences),
            "text": sentence_text,
            "start_char": sent.start_char,
            "end_char": sent.end_char,
            "spacy_entities": spacy_entities,
            "entities": [],  # Will be populated after LLM enrichment
            "sent_obj": sent  # Store sentence span (has full parse info)
        })
    
    # Report concept extraction statistics
    total_filtered = (concept_stats['filtered_chunks'] + 
                     concept_stats['filtered_by_quality'] + 
                     concept_stats['filtered_by_overlap'])
    
    print(f"   Total noun chunks found: {concept_stats['total_chunks']}", flush=True)
    print(f"   ✅ Added as concepts: {concept_stats['added_concepts']}", flush=True)
    print(f"   ❌ Filtered out: {total_filtered}\n", flush=True)
    
    # LLM Enrichment: Process in batches
    print("📊 Pass 1c: Enriching entities with Qwen3-Max...", flush=True)
    
    # Check if LLM processing is disabled or API key is not available
    if SKIP_LLM:
        print("⏭️  SKIP_LLM=true - skipping LLM enrichment, using spaCy only", flush=True)
        # Use spaCy entities only
        for sent in sentences:
            sent['entities'] = sent['spacy_entities'].copy()
    elif not os.getenv("NOVITA_API_KEY"):
        print("⚠️  NOVITA_API_KEY not set - skipping LLM enrichment, using spaCy only", flush=True)
        # Use spaCy entities only
        for sent in sentences:
            sent['entities'] = sent['spacy_entities'].copy()
    else:
        llm_start = time.time()
        
        BATCH_SIZE = 8
        total_llm_entities = 0
        llm_failed_batches = 0
        total_batches = (len(sentence_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"   Processing {len(sentence_texts)} sentences in {total_batches} batches...", flush=True)
        
        for batch_start in range(0, len(sentence_texts), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(sentence_texts))
            batch_texts = sentence_texts[batch_start:batch_end]
            batch_num = batch_start // BATCH_SIZE + 1
            
            try:
                enriched_batch = enrich_entities_with_qwen3(batch_texts, batch_num)
                
                # Merge LLM entities with spaCy entities
                for i, enriched_item in enumerate(enriched_batch):
                    sent_idx = batch_start + i
                    if sent_idx >= len(sentences):
                        break
                    
                    # Start with spaCy entities
                    merged_entities = sentences[sent_idx]['spacy_entities'].copy()
                    
                    # Add LLM entities (with additional metadata)
                    for llm_ent in enriched_item.get('entities', []):
                        merged_entities.append({
                            "text": llm_ent.get("text", ""),
                            "label": llm_ent.get("label", "ENTITY"),
                            "start_char": -1,  # LLM doesn't provide char positions
                            "end_char": -1,
                            "description": llm_ent.get("description", ""),
                            "confidence": llm_ent.get("confidence", 0.9),
                            "source": "llm"
                        })
                        total_llm_entities += 1
                    
                    sentences[sent_idx]['entities'] = merged_entities
                    
            except Exception as e:
                llm_failed_batches += 1
                print(f"   ⚠️  Batch {batch_num} exception: {type(e).__name__}", flush=True)
                # Fallback: use spaCy entities
                for i in range(len(batch_texts)):
                    sent_idx = batch_start + i
                    if sent_idx >= len(sentences):
                        break
                    sentences[sent_idx]['entities'] = sentences[sent_idx]['spacy_entities'].copy()
        
        llm_time = time.time() - llm_start
        print(f"✅ LLM enrichment completed in {llm_time:.2f}s", flush=True)
        print(f"   Added {total_llm_entities} LLM-enriched entities", flush=True)
        if llm_failed_batches > 0:
            print(f"   ⚠️  {llm_failed_batches} batches failed (using spaCy fallback)", flush=True)
    
    print()
    
    return sentences, doc

# ------------------------------------------------------------------
# 10. Extract Relationships from All Sentences
# ------------------------------------------------------------------
def extract_all_relationships(sentences_data: List[Dict], speaker_segments: List[Dict]) -> Tuple[Dict, List[Dict]]:
    """
    Extract all relationships using hybrid approach
    
    v2.4: Returns (relationships, statements) tuple
    """
    print("📊 Pass 2: Extracting rule-based relationships...", flush=True)
    start_time = time.time()
    
    all_relationships = {
        'speaker_attribution': [],
        'roles': [],
        'temporal': [],
        'quantity': []
    }
    
    # Speaker attribution from pre-processed segments (v2.4: returns statements too)
    said_rels, statements = map_speakers_to_sentences(sentences_data, speaker_segments)
    all_relationships['speaker_attribution'] = said_rels
    
    # Other rule-based extraction per sentence
    for sent_data in sentences_data:
        sent_text = sent_data['text']
        entities = sent_data['entities']
        sent = sent_data['sent_obj']
        
        # Rule-based extraction
        role_rels = extract_role_relationships(sent, entities)
        temporal_rels = extract_temporal_links(sent, entities)
        quantity_rels = extract_quantity_links(sent, entities)
        
        # Store with sentence reference
        for rel in role_rels:
            rel['sentence_id'] = sent_data['sentence_id']
            all_relationships['roles'].append(rel)
        
        for rel in temporal_rels:
            rel['sentence_id'] = sent_data['sentence_id']
            all_relationships['temporal'].append(rel)
        
        for rel in quantity_rels:
            rel['sentence_id'] = sent_data['sentence_id']
            all_relationships['quantity'].append(rel)
    
    processing_time = time.time() - start_time
    print(f"✅ Rule-based relationships extracted in {processing_time:.2f}s\n", flush=True)
    
    # Pass 3: Dependency-based extraction
    print("📊 Pass 3: Extracting dependency-based relationships...", flush=True)
    start_time = time.time()
    
    all_relationships['svo_triples'] = []
    verb_stats = {'total_verbs': 0, 'business_verbs': 0, 'with_subject': 0, 'with_object': 0, 'complete_svo': 0}
    
    # Business verbs for reference
    business_verbs = {
        'announce', 'launch', 'achieve', 'report', 'increase', 'decrease',
        'improve', 'build', 'open', 'close', 'acquire', 'sell', 'buy', 'invest',
        'deliver', 'produce', 'manufacture', 'reach', 'expect', 'plan', 'make',
        'introduce', 'ramp', 'start', 'finish', 'complete', 'accelerate', 'expand',
        'raise', 'reduce', 'settle', 'generate', 'spend', 'drive', 'exceed'
    }
    
    for sent_data in sentences_data:
        sent = sent_data['sent_obj']
        entities = sent_data['entities']
        
        # Count verbs for stats
        for token in sent:
            if token.pos_ == 'VERB':
                verb_stats['total_verbs'] += 1
                if token.lemma_.lower() in business_verbs:
                    verb_stats['business_verbs'] += 1
        
        svo_rels = extract_svo_triples(sent, entities)
        
        for rel in svo_rels:
            rel['sentence_id'] = sent_data['sentence_id']
            all_relationships['svo_triples'].append(rel)
            verb_stats['complete_svo'] += 1
    
    processing_time = time.time() - start_time
    print(f"✅ Dependency-based relationships extracted in {processing_time:.2f}s", flush=True)
    print(f"   📊 Verb stats: {verb_stats['total_verbs']} total verbs, "
          f"{verb_stats['business_verbs']} business verbs, "
          f"{verb_stats['complete_svo']} complete SVO triples", flush=True)
    print()
    
    # Build co-mention network
    print("📊 Building co-mention network...", flush=True)
    start_time = time.time()
    all_relationships['comention'] = build_comention_network(sentences_data)
    processing_time = time.time() - start_time
    print(f"✅ Co-mention network built in {processing_time:.2f}s\n", flush=True)
    
    return all_relationships, statements

# ------------------------------------------------------------------
# 11. Graph Database Persistence
# ------------------------------------------------------------------
def _memgraph_cfg_dict(host: str, port: int, username: Optional[str], password: Optional[str]) -> Dict:
    cfg = {"host": host, "port": port}
    if username:
        cfg["username"] = username
    if password:
        cfg["password"] = password
    return cfg


def _create_memgraph_connection(cfg: Dict):
    """Create and validate a single Memgraph connection."""
    db = Memgraph(**cfg)
    db.execute("RETURN 1;")
    return db


def _ensure_memgraph_pool(cfg: Dict):
    """Initialize a per-process Memgraph connection pool."""
    global _memgraph_pool, _memgraph_pool_config
    with _memgraph_pool_lock:
        if _memgraph_pool and _memgraph_pool_config == cfg:
            return
        # Close any existing pool with different config
        if _memgraph_pool:
            _close_memgraph_pool()
        _memgraph_pool_config = cfg
        _memgraph_pool = Queue(maxsize=MEMGRAPH_POOL_SIZE)
        for _ in range(MEMGRAPH_POOL_SIZE):
            conn = _create_memgraph_connection(cfg)
            _memgraph_pool.put(conn)


def _acquire_memgraph_connection(cfg: Dict):
    """Borrow a connection from the pool; recreate if the borrowed one is unhealthy."""
    _ensure_memgraph_pool(cfg)
    try:
        db = _memgraph_pool.get(timeout=30)
    except Exception as e:
        raise RuntimeError(f"Memgraph pool exhausted or unavailable: {e}") from e

    try:
        db.execute("RETURN 1;")
        return db
    except Exception:
        # Connection is bad; replace with a fresh one
        try:
            db.close()
        except Exception:
            pass
        return _create_memgraph_connection(cfg)


def _release_memgraph_connection(db):
    """Return a connection to the pool; close if pool is gone."""
    global _memgraph_pool
    if not db:
        return
    pool = _memgraph_pool
    if pool is None:
        try:
            db.close()
        finally:
            return
    try:
        pool.put(db, block=False)
    except Exception:
        try:
            db.close()
        except Exception:
            pass


def _close_memgraph_pool():
    """Close all pooled connections (registered at exit)."""
    global _memgraph_pool, _memgraph_pool_config
    if not _memgraph_pool:
        return
    while True:
        try:
            conn = _memgraph_pool.get_nowait()
        except Exception:
            break
        try:
            conn.close()
        except Exception:
            pass
    _memgraph_pool = None
    _memgraph_pool_config = None


atexit.register(_close_memgraph_pool)


def connect_to_memgraph(host='localhost', port=7687, username: Optional[str] = None, password: Optional[str] = None):
    """Borrow a Memgraph connection from the managed pool (size 20)."""
    username = username or os.getenv("MEMGRAPH_USER")
    password = password or os.getenv("MEMGRAPH_PASSWORD")
    cfg = _memgraph_cfg_dict(host, port, username, password)
    print(f"Connecting to MemgraphDB via pool at {host}:{port}...", flush=True)
    try:
        db = _acquire_memgraph_connection(cfg)
        print("✅ Connection acquired\n", flush=True)
        return db
    except Exception as e:
        print(f"❌ Failed to acquire Memgraph connection: {e}", flush=True)
        return None

def create_entity_node(db: Memgraph, entity: Dict):
    """Create or update an entity node in the graph"""
    canonical_name = entity['canonical_name']
    label = entity['label']
    entity_id = entity['id']
    
    # Sanitize label name (remove special chars, spaces)
    safe_label = label.replace(' ', '_').replace('-', '_')
    
    # v2.4: Check for metric normalization
    canonical_metric, value, unit = normalize_metric(canonical_name)
    
    if canonical_metric:
        # Create METRIC node linked to METRIC_DEFINITION
        query_metric = """
        MERGE (md:METRIC_DEFINITION {name: $canonical_metric})
        MERGE (m:METRIC {canonical_name: $canonical_name})
        SET m.entity_id = $entity_id,
            m.mention_count = $mention_count,
            m.variants = $variants,
            m.value = $value,
            m.unit = $unit
        MERGE (m)-[:SAME_AS]->(md)
        RETURN m
        """
        db.execute(query_metric, {
            'canonical_metric': canonical_metric,
            'canonical_name': canonical_name,
            'entity_id': entity_id,
            'mention_count': entity['mention_count'],
            'variants': entity['variants'],
            'value': value,
            'unit': unit
        })
    else:
        # Standard entity creation
        query = f"""
        MERGE (e:{safe_label} {{canonical_name: $canonical_name}})
        SET e.entity_id = $entity_id,
            e.mention_count = $mention_count,
            e.variants = $variants
        RETURN e
        """
        
        db.execute(query, {
            'canonical_name': canonical_name,
            'entity_id': entity_id,
            'mention_count': entity['mention_count'],
            'variants': entity['variants']
        })

def create_statement_node(db: Memgraph, statement: Dict):
    """Create a STATEMENT node (v2.4)"""
    query = """
    CREATE (s:STATEMENT {
        statement_id: $statement_id,
        text: $text,
        speaker: $speaker,
        sentence_ids: $sentence_ids
    })
    RETURN s
    """
    
    db.execute(query, {
        'statement_id': statement['id'],
        'text': statement['text'],
        'speaker': statement['speaker'],
        'sentence_ids': statement['sentence_ids']
    })

def create_role_node(db: Memgraph, role_title: str):
    """Create a ROLE node (v2.4)"""
    query = """
    MERGE (r:ROLE {title: $title})
    RETURN r
    """
    
    db.execute(query, {'title': role_title})

def create_relationship(db: Memgraph, rel: Dict):
    """Create a relationship in the graph (v2.4 enhanced)"""
    source = rel['source']
    target = rel['target']
    rel_type = rel['type'].upper().replace(' ', '_').replace('-', '_')
    target_label = rel.get('target_label', '')
    
    # Build properties dict (only include non-None values)
    props = {}
    if 'verb' in rel and rel['verb']:
        props['verb'] = str(rel['verb'])[:100]  # Limit length
    if 'verb_text' in rel and rel['verb_text']:
        props['verb_text'] = str(rel['verb_text'])[:100]
    if 'method' in rel and rel['method']:
        props['method'] = str(rel['method'])[:50]
    if 'count' in rel and rel['count']:
        props['count'] = int(rel['count'])
    if 'sentence_id' in rel and rel['sentence_id'] is not None:
        props['sentence_id'] = int(rel['sentence_id'])
    if 'confidence' in rel and rel['confidence']:
        props['confidence'] = float(rel['confidence'])
    if 'sentiment' in rel and rel['sentiment']:
        props['sentiment'] = str(rel['sentiment'])[:20]
    if 'quote' in rel and rel['quote']:
        props['quote'] = str(rel['quote'])[:200]  # Limit quote length
    
    # Create Cypher query
    if props:
        props_str = ', '.join([f'r.{k} = ${k}' for k in props.keys()])
        set_clause = f"SET {props_str}"
    else:
        set_clause = "SET r.created = timestamp()"
    
    # v2.4: Handle special target types (STATEMENT, ROLE)
    if target_label == 'STATEMENT':
        # Target is a STATEMENT node ID
        query = f"""
        MATCH (a {{canonical_name: $source}})
        MATCH (b:STATEMENT {{statement_id: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        {set_clause}
        RETURN r
        """
    elif target_label == 'ROLE':
        # Target is a ROLE title
        query = f"""
        MATCH (a {{canonical_name: $source}})
        MERGE (b:ROLE {{title: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        {set_clause}
        RETURN r
        """
    else:
        # Standard relationship by canonical_name
        query = f"""
        MATCH (a {{canonical_name: $source}})
        MATCH (b {{canonical_name: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        {set_clause}
        RETURN r
        """
    
    try:
        db.execute(query, {
            'source': source,
            'target': target,
            **props
        })
    except Exception as e:
        # Silently skip relationships where nodes don't exist
        pass

def bulk_create_comentions(db: Memgraph, rels: List[Dict], batch_size: int = 500):
    """Batch insert co-mention relationships to reduce round-trips."""
    if batch_size <= 0:
        batch_size = 500
    total = len(rels)
    if total == 0:
        return 0
    batches = math.ceil(total / batch_size)
    created_total = 0
    
    for batch_idx in range(batches):
        start = batch_idx * batch_size
        end = start + batch_size
        chunk = rels[start:end]
        
        query = """
        UNWIND $rels AS rel
        MATCH (a {canonical_name: rel.source})
        MATCH (b {canonical_name: rel.target})
        MERGE (a)-[r:CO_MENTIONED]->(b)
        SET r.count = rel.count,
            r.method = rel.method
        RETURN count(r) AS created
        """
        try:
            batch_start = time.time()
            result = list(db.execute_and_fetch(query, {"rels": chunk}))
            if result:
                created_total += result[0].get("created", 0)
            print(f"      ✅ batch {batch_idx + 1}/{batches} created in {time.time() - batch_start:.2f}s", flush=True)
        except Exception as e:
            print(f"      ⚠️  batch {batch_idx + 1}/{batches} failed: {e}", flush=True)
    
    return created_total

def persist_to_memgraph(entities: Dict, relationships: Dict, statements: List[Dict],
                        host='localhost', port=7687, clear_existing=False):
    """
    Persist extracted entities and relationships to MemgraphDB
    
    v2.4: Added statements parameter for STATEMENT nodes
    
    Args:
        entities: Dict of entity clusters from cluster_entities()
        relationships: Dict of relationship lists by type
        statements: List of STATEMENT node data (v2.4)
        host: MemgraphDB host
        port: MemgraphDB port
        clear_existing: Ignored (graph is never cleared)
    """
    print("="*80)
    print("PERSISTING TO MEMGRAPHDB")
    print("="*80)
    print()
    
    db = None
    try:
        db = connect_to_memgraph(host, port)
        if not db:
            print("❌ Skipping graph persistence - no database connection", flush=True)
            return False
        
        # Never delete existing data; only upsert
        if clear_existing:
            print("ℹ️  clear_existing=True requested, but graph deletion is disabled; keeping existing data.\n", flush=True)
        
        # Create entity nodes
        print(f"📊 Creating {len(entities)} entity nodes...", flush=True)
        start_time = time.time()
        
        created_entities = 0
        failed_entities = 0
        for entity_id, entity in entities.items():
            try:
                create_entity_node(db, entity)
                created_entities += 1
            except Exception as e:
                failed_entities += 1
                if failed_entities <= 5:  # Only show first 5 errors
                    print(f"   ⚠️  Failed to create {entity['canonical_name']}: {e}", flush=True)
        
        entity_time = time.time() - start_time
        print(f"✅ Created {created_entities} entity nodes in {entity_time:.2f}s", flush=True)
        if failed_entities > 0:
            print(f"   ⚠️  {failed_entities} entities failed", flush=True)
        print()
        
        # Create STATEMENT nodes (v2.4)
        if statements:
            print(f"📊 Creating {len(statements)} STATEMENT nodes...", flush=True)
            start_time = time.time()
            
            created_statements = 0
            for statement in statements:
                try:
                    create_statement_node(db, statement)
                    created_statements += 1
                except Exception as e:
                    pass  # Silently skip failures
            
            statement_time = time.time() - start_time
            print(f"✅ Created {created_statements} STATEMENT nodes in {statement_time:.2f}s\n", flush=True)
        
        # Create relationships
        print(f"📊 Creating relationships...", flush=True)
        start_time = time.time()
        
        created_relationships = 0
        failed_relationships = 0
        
        # Process each relationship type
        for rel_type, rels in relationships.items():
            type_start = time.time()
            type_created = 0
            type_failed = 0
            
            print(f"   Processing {len(rels)} {rel_type} relationships...", flush=True)
            
            if rel_type == 'comention':
                batch_size = COMENTION_BATCH_SIZE if COMENTION_BATCH_SIZE > 0 else 500
                created = bulk_create_comentions(db, rels, batch_size=batch_size)
                type_created += created
                created_relationships += created
            else:
                for rel in rels:
                    try:
                        create_relationship(db, rel)
                        created_relationships += 1
                        type_created += 1
                    except Exception as e:
                        failed_relationships += 1
                        type_failed += 1
            
            type_time = time.time() - type_start
            print(f"      ✅ {type_created} created in {type_time:.2f}s" + 
                  (f" ({type_failed} failed)" if type_failed > 0 else ""), flush=True)
        
        rel_time = time.time() - start_time
        print(f"\n✅ Total: {created_relationships} relationships created in {rel_time:.2f}s", flush=True)
        if failed_relationships > 0:
            print(f"   ⚠️  {failed_relationships} relationships failed (likely missing nodes)", flush=True)
        
        # Create indexes for better query performance
        print(f"\n📊 Creating indexes...", flush=True)
        try:
            db.execute("CREATE INDEX ON :PERSON(canonical_name);")
            db.execute("CREATE INDEX ON :ORG(canonical_name);")
            db.execute("CREATE INDEX ON :CONCEPT(canonical_name);")
            db.execute("CREATE INDEX ON :PRODUCT(canonical_name);")
            print("✅ Indexes created", flush=True)
        except Exception as e:
            print(f"   ℹ️  Indexes may already exist: {e}", flush=True)
        
        # Show graph statistics
        print(f"\n📊 MEMGRAPHDB STATISTICS:")
        
        try:
            node_count = list(db.execute_and_fetch("MATCH (n) RETURN count(n) as count"))[0]['count']
            rel_count = list(db.execute_and_fetch("MATCH ()-[r]->() RETURN count(r) as count"))[0]['count']
            
            print(f"   Total nodes: {node_count}")
            print(f"   Total relationships: {rel_count}")
            
            # Show node breakdown by label
            label_query = "MATCH (n) RETURN labels(n)[0] as label, count(*) as count ORDER BY count DESC"
            label_results = list(db.execute_and_fetch(label_query))
            
            print(f"\n   Nodes by type:")
            for result in label_results:
                print(f"      {result['label']}: {result['count']}")
            
            # Show relationship breakdown
            rel_type_query = "MATCH ()-[r]->() RETURN type(r) as rel_type, count(*) as count ORDER BY count DESC LIMIT 10"
            rel_type_results = list(db.execute_and_fetch(rel_type_query))
            
            print(f"\n   Top relationship types:")
            for result in rel_type_results:
                print(f"      {result['rel_type']}: {result['count']}")
            
        except Exception as e:
            print(f"   ⚠️  Error fetching statistics: {e}", flush=True)
        
        print(f"\n{'='*80}\n")
        
        return True
    finally:
        if db:
            _release_memgraph_connection(db)

# ------------------------------------------------------------------
# 12. GCS Data Loading Functions
# ------------------------------------------------------------------
def load_transcript_from_gcs(symbol: str, year: str, quarter: str) -> Optional[Dict]:
    """
    Load a transcript JSON file from GCS
    Returns parsed JSON data or None if not found
    Expected format: earnings-announcement-transcripts/{symbol}/{year}.{quarter}.json
    Example: earnings-announcement-transcripts/AAPL/2024.1.json
    """
    # Normalize quarter to number (accept Q1, 1, q1, etc.)
    quarter_clean = quarter.upper().replace('Q', '')
    try:
        quarter_num = int(quarter_clean)
    except ValueError:
        print(f"   ⚠️  Invalid quarter format: {quarter}", flush=True)
        return None
    
    # GCS path format: {symbol}/{year}.{quarter}.json
    gcs_path = f"{GCS_BASE_PATH}/{symbol}/{year}.{quarter_num}.json"
    
    try:
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            return None
        
        # Download and parse JSON
        json_str = blob.download_as_text()
        transcript_data = json.loads(json_str)
        print(f"   ✅ Loaded from: {gcs_path}", flush=True)
        return transcript_data
    except Exception as e:
        print(f"   ⚠️  Error loading {gcs_path}: {e}", flush=True)
        return None

def list_transcripts_from_gcs(symbol: str, year: str) -> List[str]:
    """
    List all transcript files available in GCS for a given symbol and year
    Returns list of quarter identifiers (e.g., ["1", "2", "3", "4"])
    GCS path format: {symbol}/{year}.{quarter}.json
    """
    prefix = f"{GCS_BASE_PATH}/{symbol}/{year}."
    quarters = []
    
    try:
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith('.json'):
                # Extract quarter from filename (e.g., "2024.1.json" -> "1")
                filename = os.path.basename(blob.name)
                # Format is {year}.{quarter}.json
                parts = filename.replace('.json', '').split('.')
                if len(parts) == 2 and parts[0] == year:
                    quarter_id = parts[1]
                    if quarter_id not in quarters:
                        quarters.append(quarter_id)
        
        # Sort quarters numerically
        def quarter_sort_key(q: str) -> int:
            """Sort quarters: 1, 2, 3, 4"""
            try:
                return int(q)
            except ValueError:
                return 999
        
        quarters.sort(key=quarter_sort_key)
        return quarters
        
    except Exception as e:
        print(f"❌ Error listing transcripts from GCS: {e}", flush=True)
        return []

# ------------------------------------------------------------------
# 13. Main Processing Pipeline
# ------------------------------------------------------------------
def process_documents(symbol: str = None, year: str = None, quarter: str = None):
    """
    Main pipeline: Extract entities and relationships from earnings transcripts
    Reads from GCS bucket: blacksmith-sec-filings/earnings-announcement-transcripts
    """
    overall_start = time.time()
    
    # Get symbol, year, quarter from parameters or environment
    if symbol is None:
        symbol = SYMBOL
    if year is None:
        year = YEAR
    if quarter is None:
        quarter = QUARTER
    
    # Validate required parameters
    if not symbol:
        print("❌ ERROR: SYMBOL is required. Set SYMBOL environment variable or pass as argument.", flush=True)
        print("   Example: export SYMBOL=AAPL", flush=True)
        return None
    
    if not year:
        print("❌ ERROR: YEAR is required. Set YEAR environment variable or pass as argument.", flush=True)
        print("   Example: export YEAR=2024", flush=True)
        return None
    
    if not quarter:
        print("❌ ERROR: QUARTER is required. Set QUARTER environment variable or pass as argument.", flush=True)
        print("   Example: export QUARTER=Q1 or export QUARTER=1", flush=True)
        return None
    
    symbol = symbol.upper()
    # Normalize quarter format (accept Q1, 1, q1, etc.)
    quarter_clean = quarter.upper().replace('Q', '')
    try:
        quarter_num = int(quarter_clean)
        quarter_normalized = f"Q{quarter_num}"
    except ValueError:
        print(f"❌ Invalid quarter format: {quarter}", flush=True)
        print(f"   Expected: Q1, 1, Q2, 2, etc.", flush=True)
        return None
    
    print("="*80)
    print("ENTITY & RELATIONSHIP EXTRACTION PIPELINE (HYBRID)")
    print("="*80)
    print(f"📊 Processing: {symbol} - {year} {quarter_normalized}")
    print(f"☁️  GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 GCS Path: {GCS_BASE_PATH}/{symbol}/{year}.{quarter_num}.json")
    print()
    
    # Load transcript from GCS
    print(f"📥 Loading transcript from GCS...", flush=True)
    load_start = time.time()
    
    transcript_data = load_transcript_from_gcs(symbol, year, quarter_clean)
    
    if not transcript_data:
        print(f"❌ Transcript not found in GCS for {symbol}/{year}/{quarter_normalized}", flush=True)
        print(f"   Expected path: {GCS_BASE_PATH}/{symbol}/{year}.{quarter_num}.json", flush=True)
        # List available transcripts
        available = list_transcripts_from_gcs(symbol, year)
        if available:
            print(f"   Available transcripts for {symbol}/{year}: {', '.join([f'Q{q}' for q in available])}", flush=True)
        return None
    
    load_time = time.time() - load_start
    print(f"✅ Loaded transcript in {load_time:.2f}s\n", flush=True)
    
    # Extract transcript content
    # Handle both formats: {"content": "..."} or [{"symbol": "...", "content": "..."}]
    transcript_text = None
    
    if isinstance(transcript_data, dict):
        if 'content' in transcript_data:
            transcript_text = transcript_data['content']
        elif 'transcript' in transcript_data:
            transcript_text = transcript_data['transcript']
        else:
            # Try to find text in any field
            for key, value in transcript_data.items():
                if isinstance(value, str) and len(value) > 1000:  # Likely the transcript
                    transcript_text = value
                    break
    elif isinstance(transcript_data, list) and len(transcript_data) > 0:
        if isinstance(transcript_data[0], dict):
            if 'content' in transcript_data[0]:
                transcript_text = transcript_data[0]['content']
            elif 'transcript' in transcript_data[0]:
                transcript_text = transcript_data[0]['transcript']
    
    if not transcript_text:
        print(f"❌ Could not extract transcript content from JSON data", flush=True)
        return None
    
    print(f"📄 Transcript: {symbol} {quarter_normalized} {year} ({len(transcript_text):,} chars)", flush=True)
    print(f"\nTotal text length: {len(transcript_text):,} characters\n", flush=True)
    
    full_transcript_text = transcript_text
    
    # Pre-process: Extract speaker patterns BEFORE sentence splitting
    print("📊 Pre-processing: Extracting speaker patterns...", flush=True)
    start_time = time.time()
    speaker_segments = extract_speaker_patterns(full_transcript_text)
    processing_time = time.time() - start_time
    print(f"✅ Found {len(speaker_segments)} speaker segments in {processing_time:.2f}s\n", flush=True)
    
    # Extract sentences and entities (Pass 1)
    sentences_data, full_doc = extract_sentences_and_entities(full_transcript_text)
    print(f"Found {len(sentences_data)} sentences\n", flush=True)
    
    # Extract relationships (Pass 2 & 3) - v2.4: also returns statements
    relationships, statements = extract_all_relationships(sentences_data, speaker_segments)
    
    # Entity clustering and normalization (needed before LLM relationship enrichment)
    print("📊 Clustering and normalizing entities...", flush=True)
    start_time = time.time()
    
    all_entities = []
    for sent_data in sentences_data:
        all_entities.extend(sent_data['entities'])
    
    entity_clusters = cluster_entities(all_entities)
    
    processing_time = time.time() - start_time
    print(f"✅ Entity clustering completed in {processing_time:.2f}s\n", flush=True)
    
    # LLM-based relationship enrichment (Pass 4)
    if SKIP_LLM:
        print("⏭️  SKIP_LLM=true - skipping LLM relationship enrichment", flush=True)
        relationships['llm_enriched'] = []
        print()
    else:
        print("📊 Pass 4: Enriching relationships with Qwen3-Max...", flush=True)
        start_time = time.time()
        
        llm_relationships = enrich_relationships_with_qwen3(sentences_data, entity_clusters)
        relationships['llm_enriched'] = llm_relationships
        
        processing_time = time.time() - start_time
        print(f"✅ LLM relationship enrichment completed in {processing_time:.2f}s", flush=True)
        print(f"   Added {len(llm_relationships)} LLM-enriched relationships\n", flush=True)
    
    # Display summary statistics
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    
    # Count entity types
    entity_type_counts = defaultdict(int)
    for ent in all_entities:
        entity_type_counts[ent['label']] += 1
    
    print(f"\n📊 ENTITIES:")
    print(f"   Total entity mentions: {len(all_entities)}")
    print(f"   Unique entities (clustered): {len(entity_clusters)}")
    print(f"\n   Breakdown by type:")
    for label, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {label}: {count} mentions")
    
    print(f"\n📊 RELATIONSHIPS:")
    print(f"   Speaker attributions: {len(relationships['speaker_attribution'])}")
    print(f"   Role/Title relationships: {len(relationships['roles'])}")
    print(f"   SVO triples: {len(relationships['svo_triples'])}")
    print(f"   Temporal links: {len(relationships['temporal'])}")
    print(f"   Quantity links: {len(relationships['quantity'])}")
    print(f"   Co-mention pairs: {len(relationships['comention'])}")
    print(f"   🤖 LLM-enriched (causal, sentiment, etc.): {len(relationships['llm_enriched'])}")
    
    total_relationships = sum(len(rels) for rels in relationships.values())
    print(f"   Total relationships: {total_relationships}")
    
    # Performance metrics
    overall_time = time.time() - overall_start
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Total execution time: {overall_time:.2f}s")
    print(f"   Processing speed: {len(sentences_data)/overall_time:.1f} sentences/sec")
    
    # Show top entity clusters
    print(f"\n📊 TOP ENTITIES (by mention count):")
    sorted_entities = sorted(entity_clusters.values(), key=lambda x: x['mention_count'], reverse=True)
    
    # Show top 15 named entities (non-concepts)
    named_entities = [e for e in sorted_entities if e['label'] != 'CONCEPT']
    print(f"\n   Top Named Entities:")
    for ent in named_entities[:15]:
        print(f"      {ent['canonical_name']} ({ent['label']}): {ent['mention_count']} mentions")
        if len(ent['variants']) > 1:
            print(f"         Variants: {', '.join(ent['variants'][:3])}" + 
                  (f" + {len(ent['variants']) - 3} more" if len(ent['variants']) > 3 else ""))
    
    # Show top 15 concepts (Option B: filtered by minimum mention threshold)
    concepts = [e for e in sorted_entities 
                if e['label'] == 'CONCEPT' and e['mention_count'] >= MIN_CONCEPT_MENTIONS]
    if concepts:
        print(f"\n   Top Concepts (multi-word noun phrases, {MIN_CONCEPT_MENTIONS}+ mentions):")
        for ent in concepts[:15]:
            print(f"      {ent['canonical_name']}: {ent['mention_count']} mentions")
            if len(ent['variants']) > 1:
                print(f"         Variants: {', '.join(ent['variants'][:3])}" + 
                      (f" + {len(ent['variants']) - 3} more" if len(ent['variants']) > 3 else ""))
    
    # Also report how many concepts were filtered by the threshold
    total_concepts = len([e for e in sorted_entities if e['label'] == 'CONCEPT'])
    filtered_by_threshold = total_concepts - len(concepts)
    if filtered_by_threshold > 0:
        print(f"\n   ({filtered_by_threshold} additional concepts with only 1 mention not shown)")
    
    # Show sample relationships
    print(f"\n📊 SAMPLE RELATIONSHIPS:")
    
    print(f"\n   Speaker Attribution (showing first 15):")
    if relationships['speaker_attribution']:
        for rel in relationships['speaker_attribution'][:15]:
            statement_preview = rel['target'][:80] + ('...' if len(rel['target']) > 80 else '')
            print(f"      [{rel['source']}] SAID: \"{statement_preview}\"")
    else:
        print("      (None found)")
    
    print(f"\n   Roles (showing all):")
    if relationships['roles']:
        for rel in relationships['roles']:
            print(f"      [{rel['source']}] HAS_ROLE: [{rel['target']}]")
    else:
        print("      (None found)")
    
    print(f"\n   SVO Triples (showing first 20):")
    if relationships['svo_triples']:
        for rel in relationships['svo_triples'][:20]:
            print(f"      [{rel['source'][:50]}] --{rel['verb_text']}--> [{rel['target'][:50]}]")
    else:
        print("      (None found)")
    
    print(f"\n   Quantity Links (showing first 15):")
    if relationships['quantity']:
        for rel in relationships['quantity'][:15]:
            print(f"      [{rel['source']}] --> [{rel['target']}] ({rel['type']})")
    else:
        print("      (None found)")
    
    print(f"\n   Co-mentions (top 15 by frequency):")
    sorted_comentions = sorted(relationships['comention'], key=lambda x: x['count'], reverse=True)
    for rel in sorted_comentions[:15]:
        print(f"      [{rel['source'][:40]}] <--{rel['count']}x--> [{rel['target'][:40]}]")
    
    print(f"\n   🤖 LLM-Enriched Relationships (showing all):")
    if relationships['llm_enriched']:
        for rel in relationships['llm_enriched']:
            sentiment_emoji = {"positive": "✅", "negative": "❌", "neutral": "➖"}.get(rel.get('sentiment', 'neutral'), "➖")
            print(f"      {sentiment_emoji} [{rel['source'][:35]}] --{rel['type']}--> [{rel['target'][:35]}]")
            if rel.get('quote'):
                print(f"         Evidence: \"{rel['quote'][:80]}...\"")
            if rel.get('confidence'):
                print(f"         Confidence: {rel['confidence']:.2f}")
    else:
        print("      (None found)")
    
    print(f"\n{'='*80}\n")
    
    # Persist to MemgraphDB if enabled
    enable_graph_persistence = os.getenv("ENABLE_GRAPH_PERSISTENCE", "true").lower() == "true"
    if enable_graph_persistence:
        persist_to_memgraph(
            entities=entity_clusters,
            relationships=relationships,
            statements=statements,  # v2.4: pass statements
            host=os.getenv('MEMGRAPH_HOST', 'localhost'),
            port=int(os.getenv('MEMGRAPH_PORT', '7687')),
            clear_existing=os.getenv('MEMGRAPH_CLEAR_EXISTING', 'true').lower() == 'true'
        )
    else:
        print("ℹ️  Graph persistence disabled (set ENABLE_GRAPH_PERSISTENCE=true to enable)\n", flush=True)
    
    # Return data structures
    return {
        'sentences': sentences_data,
        'entities': entity_clusters,
        'relationships': relationships,
        'stats': {
            'total_sentences': len(sentences_data),
            'total_entity_mentions': len(all_entities),
            'unique_entities': len(entity_clusters),
            'total_relationships': total_relationships,
            'processing_time_seconds': overall_time
        }
    }

# ------------------------------------------------------------------
# 14. Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract entities and relationships from earnings announcement transcripts stored in GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variables:
  export SYMBOL=AAPL YEAR=2024 QUARTER=Q1
  python v1.0-graph-ea-scripts.py

  # Using command-line arguments:
  python v1.0-graph-ea-scripts.py --symbol AAPL --year 2024 --quarter Q1

  # Override environment variables:
  export SYMBOL=AAPL YEAR=2024 QUARTER=Q1
  python v1.0-graph-ea-scripts.py --quarter Q2  # Will use AAPL/2024 from env, Q2 from arg
        """
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default=None,
        help='Stock symbol (e.g., AAPL, MSFT). Can also be set via SYMBOL environment variable.'
    )
    parser.add_argument(
        '--year', '-y',
        type=str,
        default=None,
        help='Year of the earnings call (e.g., 2024). Can also be set via YEAR environment variable.'
    )
    parser.add_argument(
        '--quarter', '-q',
        type=str,
        default=None,
        help='Quarter of the earnings call (e.g., Q1, 1, Q2). Can also be set via QUARTER environment variable.'
    )
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip all LLM processing (entity and relationship enrichment). Faster processing using only spaCy.'
    )
    
    args = parser.parse_args()
    
    # Override SKIP_LLM if command-line argument is provided
    # Update the module-level variable so functions can see it
    if args.skip_llm:
        globals()['SKIP_LLM'] = True
    
    # Get symbol, year, quarter from args or environment
    symbol = args.symbol or SYMBOL
    year = args.year or YEAR
    quarter = args.quarter or QUARTER
    
    # Validate
    if not symbol:
        print("❌ ERROR: SYMBOL is required.", flush=True)
        print("   Set via: --symbol AAPL or export SYMBOL=AAPL", flush=True)
        exit(1)
    
    if not year:
        print("❌ ERROR: YEAR is required.", flush=True)
        print("   Set via: --year 2024 or export YEAR=2024", flush=True)
        exit(1)
    
    if not quarter:
        print("❌ ERROR: QUARTER is required.", flush=True)
        print("   Set via: --quarter Q1 or export QUARTER=Q1", flush=True)
        exit(1)
    
    symbol = symbol.upper()
    quarter_clean = quarter.upper().replace('Q', '')
    quarter_normalized = f"Q{quarter_clean}"
    
    # Process documents
    result = process_documents(symbol=symbol, year=year, quarter=quarter)
    
    if result is None:
        print("❌ Processing failed or no data found", flush=True)
        exit(1)
    
    # Optionally save to JSON for inspection
    output_file = f"extraction_results_{symbol}_{year}_{quarter_normalized}.json"
    print(f"💾 Saving results to {output_file}...", flush=True)
    
    # Prepare for JSON serialization (remove sent objects)
    json_result = {
        'symbol': symbol,
        'year': year,
        'quarter': quarter_normalized,
        'sentences': [{
            'sentence_id': s['sentence_id'],
            'text': s['text'],
            'start_char': s['start_char'],
            'end_char': s['end_char'],
            'entities': s['entities']
        } for s in result['sentences']],
        'entities': result['entities'],
        'relationships': result['relationships'],
        'stats': result['stats']
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_result, f, indent=2)
    
    print(f"✅ Results saved to {output_file}\n", flush=True)

