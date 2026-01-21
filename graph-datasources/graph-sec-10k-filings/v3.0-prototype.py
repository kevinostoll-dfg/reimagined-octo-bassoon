#!/usr/bin/env python3
"""
Entity & Relationship Extraction Pipeline v3.0 (SEC 10-K Filings)
Extracts entities and relationships from SEC 10-K filings using spaCy and LLM enrichment
Enhanced with Qwen3-Max via Novita.ai for deep semantic understanding
Outputs Python data structures for knowledge graph construction

v3.0 MIGRATION (Earnings Calls → SEC 10-K Filings):
- Multi-threaded section processing (each 10-K section processed in parallel)
- Section-aware entity extraction with SECTION nodes
- 10-K specific entity types: RISK, REGULATION, GEOGRAPHY, SEGMENT, COMPETITOR
- Updated relationship types: EXPOSES_TO_RISK, OPERATES_IN, COMPETES_WITH, SUPPLIES
- Annual financial metrics (vs quarterly earnings metrics)
- Cross-section relationship detection
- Removed: Speaker attribution, STATEMENT nodes, conversational patterns
"""

import os
import json
import time
import re
import logging
import argparse
import threading
import fcntl
import pickle
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from contextlib import contextmanager
from pathlib import PurePosixPath, Path
from datetime import datetime, timezone
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
from openai import OpenAI
from gqlalchemy import Memgraph, Node, Relationship
from google.cloud import storage

# Load environment variables
load_dotenv()

# Initialize Novita client (OpenAI-compatible API) - only if API key is set
novita_api_key = os.getenv("NOVITA_API_KEY")
if novita_api_key:
    novita_client = OpenAI(
        api_key=novita_api_key,
        base_url="https://api.novita.ai/v3/openai"
    )
else:
    novita_client = None
    print("ℹ️  NOVITA_API_KEY not set - LLM enrichment will be skipped (spaCy-only mode)\n", flush=True)

# Reduce noise in logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# ------------------------------------------------------------------
# 1. Configuration from Environment
# ------------------------------------------------------------------
# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "sec-10k")
SPACY_MODELS_BUCKET = os.getenv("SPACY_MODELS_BUCKET", "blacksmith-sec-filings")
SPACY_MODELS_GCS_PATH = os.getenv("SPACY_MODELS_GCS_PATH", "spacy-models")

# Checkpoint configuration (matches batch_process_10k.py)
CHECKPOINT_GCS_PATH = os.getenv("CHECKPOINT_GCS_PATH", "checkpoints/10k_processing_checkpoint.pkl")
CHECKPOINT_ENABLED = os.getenv("CHECKPOINT_ENABLED", "true").lower() == "true"

# Fine-tuned model configuration (optional)
FINE_TUNED_MODEL_GCS_PATH = os.getenv("FINE_TUNED_MODEL_GCS_PATH", "")
FINE_TUNED_MODEL_NAME = os.getenv("FINE_TUNED_MODEL_NAME", "")

# Symbol and Year - REQUIRED for processing
SYMBOL = os.getenv("SYMBOL", "").upper()
YEAR = os.getenv("YEAR", "")

# ------------------------------------------------------------------
# 1.1 Input Validation & Safe Paths
# ------------------------------------------------------------------
import re
def validate_symbol(symbol: str) -> str:
    """Validate stock ticker symbol (1-5 uppercase letters)."""
    cleaned = str(symbol).strip().upper()
    if not re.match(r'^[A-Z]{1,5}$', cleaned):
        raise ValueError(f"Invalid symbol format: {symbol!r}. Expected 1-5 uppercase letters.")
    return cleaned

def validate_year(year: str) -> int:
    """Validate year within reasonable SEC filing bounds."""
    try:
        y = int(str(year).strip())
        if not (1995 <= y <= 2030):
            raise ValueError(f"Year out of reasonable range: {y}")
        return y
    except ValueError:
        raise ValueError(f"Invalid year: {year!r}. Expected 4-digit number.")

def sanitize_section_id(section: str) -> str:
    """Restrict section id to alphanumerics (e.g., 1, 1A, 7A)."""
    if not section:
        return "unknown"
    cleaned = re.sub(r'[^A-Za-z0-9]', '', str(section).strip())
    return cleaned or "unknown"

def safe_gcs_section_path(symbol: str, year: str, section: str) -> str:
    """Build safe GCS path preventing traversal."""
    safe_symbol = validate_symbol(symbol)
    safe_year = str(validate_year(year))
    safe_section = sanitize_section_id(section)
    # PurePosixPath to normalize any stray separators
    safe_rel = PurePosixPath(GCS_BASE_PATH) / safe_symbol / safe_year / f"{safe_section}.json"
    return str(safe_rel)

# Initialize GCS client (uses Application Default Credentials)
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# ------------------------------------------------------------------
# GQLAlchemy relationship registrations
# ------------------------------------------------------------------
class CO_MENTIONED(Relationship):
    """
    Register CO_MENTIONED so GQLAlchemy recognizes the relationship type.
    """
    type = "CO_MENTIONED"
    count: Optional[int]
    method: Optional[str]
    section: Optional[str]
    section_name: Optional[str]
    sentence_id: Optional[int]

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

# ------------------------------------------------------------------
# 2. Metric Normalization Configuration (v3.0 - 10-K Metrics)
# ------------------------------------------------------------------
# Canonical metric mapping for SEC 10-K filings (annual metrics)
METRIC_CANONICAL_MAP = {
    # Annual Revenue & Income
    "net sales": "Net Sales (Annual)",
    "total.*revenue": "Total Revenue (Annual)",
    "operating income": "Operating Income",
    "net income": "Net Income",
    "income.*operations": "Income from Operations",
    
    # Expenses
    "research.*development": "R&D Expenses",
    "r&d": "R&D Expenses",
    "cost.*sales": "Cost of Sales",
    "selling.*general.*administrative": "SG&A Expenses",
    "operating expenses": "Operating Expenses",
    
    # Balance Sheet
    "total assets": "Total Assets",
    "total liabilities": "Total Liabilities",
    "shareholders.*equity": "Shareholders' Equity",
    "stockholders.*equity": "Shareholders' Equity",
    "cash.*equivalents": "Cash and Cash Equivalents",
    "term debt": "Term Debt",
    
    # Margins & Returns
    "gross margin": "Gross Margin",
    "operating margin": "Operating Margin",
    "effective tax rate": "Effective Tax Rate",
    
    # Market Risk Metrics
    "interest rate.*sensitivity": "Interest Rate Sensitivity",
    "foreign exchange.*exposure": "Foreign Exchange Exposure",
    "value.?at.?risk": "Value at Risk (VAR)",
    "var": "Value at Risk (VAR)",
    
    # Shares & Per-Share
    "diluted.*per.*share": "Diluted EPS",
    "basic.*per.*share": "Basic EPS",
    "earnings per share": "Diluted EPS",
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
# File lock for serialized graph writes (per symbol)
# ------------------------------------------------------------------
@contextmanager
def acquire_symbol_lock(symbol: Optional[str], timeout: Optional[float] = None, poll_interval: float = 1.0):
    """
    Serialize Memgraph writes by ticker to avoid transaction conflicts.
    By default, wait indefinitely so same-symbol jobs queue instead of failing.
    MEMGRAPH_LOCK_TIMEOUT_SECONDS can override; <=0 keeps the infinite wait.
    """
    env_timeout = os.getenv("MEMGRAPH_LOCK_TIMEOUT_SECONDS")
    if env_timeout:
        try:
            timeout_val = float(env_timeout)
            timeout = None if timeout_val <= 0 else timeout_val
        except ValueError:
            # Ignore malformed env value; fall back to provided timeout (default: None)
            pass
    
    lock_dir = Path(os.getenv("MEMGRAPH_LOCK_DIR", "/tmp/graph-sec-10k-locks"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = (symbol or "GLOBAL").upper()
    lock_file = lock_dir / f"{lock_name}.lock"
    
    start = time.time()
    last_log = 0.0
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write pid for observability (best effort)
                try:
                    os.ftruncate(fd, 0)
                    os.write(fd, str(os.getpid()).encode())
                except OSError:
                    pass
                break
            except BlockingIOError:
                waited = time.time() - start
                if timeout is not None and waited >= timeout:
                    raise TimeoutError(f"Timeout acquiring lock for {lock_name}")
                
                if waited - last_log >= 30:
                    holder_pid = None
                    try:
                        os.lseek(fd, 0, os.SEEK_SET)
                        holder_pid_raw = os.read(fd, 32).decode().strip()
                        holder_pid = holder_pid_raw or None
                    except OSError:
                        holder_pid = None
                    print(
                        f"Waiting for lock {lock_name} (held by pid {holder_pid}) for {waited:.1f}s",
                        flush=True
                    )
                    last_log = waited
                
                time.sleep(poll_interval)
        
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

# ------------------------------------------------------------------
# 3. 10-K Section Definitions (v3.0)
# ------------------------------------------------------------------
SECTION_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures about Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accounting Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}

# ------------------------------------------------------------------
# 4. Model Format Detection
# ------------------------------------------------------------------
def detect_model_format(model_dir: Path) -> str:
    """
    Determine whether a downloaded model directory is HuggingFace or spaCy.
    HuggingFace: has weights (model.safetensors or pytorch_model.bin) AND config.json
                whose JSON contains model_type or architectures.
    spaCy: has meta.json.
    Raises if neither is satisfied.
    """
    if not model_dir.exists():
        raise RuntimeError(f"Model directory does not exist: {model_dir}")
    
    meta_path = model_dir / "meta.json"
    cfg_path = model_dir / "config.json"
    weight_paths = [model_dir / "model.safetensors", model_dir / "pytorch_model.bin"]
    
    has_spacy = meta_path.exists()
    has_cfg = cfg_path.exists()
    has_weights = any(p.exists() for p in weight_paths)
    
    if has_cfg and has_weights:
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as e:
            raise RuntimeError(f"config.json is unreadable in {model_dir}: {type(e).__name__}: {e}") from e
        
        model_type = cfg.get("model_type")
        architectures = cfg.get("architectures")
        if model_type or (isinstance(architectures, list) and architectures):
            return "huggingface"
        raise RuntimeError(
            f"config.json in {model_dir} is missing required HuggingFace keys "
            f"(model_type or architectures)."
        )
    
    if has_spacy:
        return "spacy"
    
    found = []
    if has_cfg:
        found.append("config.json")
    if has_weights:
        found.append("weights")
    if found:
        found_str = ", ".join(found)
    else:
        found_str = "no expected files"
    raise RuntimeError(
        f"Could not determine model format in {model_dir}. Found: {found_str}. "
        f"Expected meta.json for spaCy or weights+config.json for HuggingFace."
    )


# ------------------------------------------------------------------
# 5. Download spaCy Model from GCS
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
    from pathlib import Path
    
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
    
    # Check if model already exists in cache
    if model_cache_path.exists():
        try:
            model_format = detect_model_format(model_cache_path)
            print(f"✅ {model_name} ({model_format}) found in cache: {model_cache_path}", flush=True)
            return str(model_cache_path)
        except Exception as e:
            print(f"⚠️  Cached model at {model_cache_path} is invalid: {e}. Re-downloading...", flush=True)
    
    # Create cache directory for this model
    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize GCS client
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(SPACY_MODELS_BUCKET)
        
        # Use fine-tuned model path if specified, otherwise use default spacy-models path
        if FINE_TUNED_MODEL_NAME and model_name == FINE_TUNED_MODEL_NAME and FINE_TUNED_MODEL_GCS_PATH:
            # For fine-tuned models, use the GCS path directly (model files are in that directory)
            gcs_prefix = f"{FINE_TUNED_MODEL_GCS_PATH}/"
        else:
            gcs_prefix = f"{SPACY_MODELS_GCS_PATH}/{model_name}/"
        
        # List all files in the model directory
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        
        if not blobs:
            raise RuntimeError(
                f"Model {model_name} not found in GCS bucket {SPACY_MODELS_BUCKET} at path {gcs_prefix}"
            )
        
        print(f"   Found {len(blobs)} files to download...", flush=True)
        
        # Download all files, preserving directory structure
        downloaded = 0
        for blob in blobs:
            # Get relative path from model root
            relative_path = blob.name[len(gcs_prefix):]
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
# 6. Initialize spaCy Model
# ------------------------------------------------------------------
def initialize_spacy():
    """
    Initialize spaCy model with required pipeline components.
    Downloads model from GCS if not available locally.
    Supports both spaCy models and HuggingFace models.
    Automatically uses GPU if available (CUDA).
    MUST SUCCEED - no fallbacks.
    """
    print("Loading spaCy model...", flush=True)
    start_time = time.time()
    
    import spacy
    from pathlib import Path
    
    # Check for GPU availability
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
            print(f"🎮 GPU detected: {gpu_name} (device: {device})", flush=True)
        else:
            device = "cpu"
            print(f"💻 Using CPU (no GPU detected)", flush=True)
    except ImportError:
        device = "cpu"
        use_gpu = False
        print(f"💻 Using CPU (torch not available)", flush=True)
    
    # Use fine-tuned model if specified, otherwise default to transformer model
    if FINE_TUNED_MODEL_NAME and FINE_TUNED_MODEL_GCS_PATH:
        model_name = FINE_TUNED_MODEL_NAME
        print(f"🎯 Using fine-tuned HuggingFace model: {model_name}", flush=True)
        
        # Download HuggingFace model from GCS
        print(f"📥 Downloading HuggingFace model from GCS...", flush=True)
        model_path = download_spacy_model_from_gcs(model_name)
        model_path_abs = str(Path(model_path).resolve())
        
        # Determine model format from the downloaded directory
        model_format = detect_model_format(Path(model_path_abs))
        
        if model_format == "huggingface":
            # Load HuggingFace model and create spaCy pipeline with transformer component
            try:
                import spacy_transformers
                from transformers import AutoTokenizer, AutoModel
                
                print(f"   Loading HuggingFace model from {model_path_abs}...", flush=True)
                
                # Verify tokenizer files exist in the model directory
                tokenizer_path = Path(model_path_abs)
                if not (tokenizer_path / "tokenizer_config.json").exists() and not (tokenizer_path / "vocab.txt").exists():
                    # Check if tokenizer files are in a subdirectory
                    potential_tokenizer_dirs = [tokenizer_path, tokenizer_path / "tokenizer"]
                    tokenizer_found = False
                    for tok_dir in potential_tokenizer_dirs:
                        if (tok_dir / "tokenizer_config.json").exists() or (tok_dir / "vocab.txt").exists():
                            tokenizer_path = tok_dir
                            tokenizer_found = True
                            break
                    if not tokenizer_found:
                        raise RuntimeError(
                            f"Tokenizer files not found in {model_path_abs}. "
                            f"Expected tokenizer_config.json or vocab.txt in the model directory."
                        )
                
                # Explicitly load tokenizer to verify it works
                try:
                    test_tokenizer = AutoTokenizer.from_pretrained(str(model_path_abs))
                    print(f"   ✅ Tokenizer loaded successfully", flush=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load tokenizer from {model_path_abs}. "
                        f"Error: {type(e).__name__}: {e}"
                    ) from e
                
                # Create a blank spaCy pipeline
                nlp = spacy.blank("en")
                
                # Add transformer component pointing to the local HuggingFace model
                # spacy-transformers will automatically load the model and tokenizer from the path
                # The "name" field should point to a directory containing both model and tokenizer files
                config = {
                    "model": {
                        "@architectures": "spacy-transformers.TransformerModel.v3",
                        "name": str(model_path_abs),  # Point directly to the local model directory
                        # tokenizer_config is optional - spacy-transformers will auto-detect tokenizer from model path
                    }
                }
                
                # Add sentencizer first (before transformer) for sentence segmentation
                nlp.add_pipe("sentencizer", first=True)
                
                # Add the transformer pipe with explicit tokenizer path
                # spacy-transformers will load model + tokenizer from the directory
                transformer = nlp.add_pipe("transformer", config=config)
                
                # Initialize components so the transformer sets up its tokenizer
                # If we get a Flax weights error, convert and retry
                try:
                    nlp.initialize(lambda: [])
                except Exception as e:
                    error_str = str(e)
                    # Check if this is the Flax weights error
                    if "Flax weights" in error_str and "from_flax=True" in error_str:
                        # Use a lock file to prevent multiple processes from converting simultaneously
                        lock_file_path = Path(model_path_abs) / ".conversion_lock"
                        lock_acquired = False
                        try:
                            # Try to acquire exclusive lock (non-blocking)
                            with open(lock_file_path, "w") as lock_file:
                                try:
                                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                                    lock_acquired = True
                                    
                                    # Double-check PyTorch weights weren't created by another process
                                    model_dir = Path(model_path_abs)
                                    has_pytorch = (
                                        (model_dir / "pytorch_model.bin").exists() or
                                        (model_dir / "model.safetensors").exists() or
                                        (model_dir / "pytorch_model.bin.index.json").exists()
                                    )
                                    
                                    if not has_pytorch:
                                        print(f"   ⚠️  Model has Flax weights but no PyTorch weights. Converting to PyTorch format...", flush=True)
                                        # Load model with from_flax=True
                                        model = AutoModel.from_pretrained(str(model_path_abs), from_flax=True)
                                        # Save in PyTorch format (this will create pytorch_model.bin or model.safetensors)
                                        model.save_pretrained(str(model_path_abs))
                                        print(f"   ✅ Converted Flax weights to PyTorch format", flush=True)
                                    else:
                                        print(f"   ℹ️  PyTorch weights already exist (created by another process)", flush=True)
                                    
                                    # Release lock
                                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                                except BlockingIOError:
                                    # Another process is converting, wait for it to finish
                                    print(f"   ⏳ Another process is converting Flax weights, waiting...", flush=True)
                                    # Wait with blocking lock
                                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                                    print(f"   ✅ Conversion completed by another process", flush=True)
                            
                            # Retry initialization after conversion
                            nlp.initialize(lambda: [])
                        except Exception as conv_e:
                            raise RuntimeError(
                                f"Failed to convert Flax weights to PyTorch format. "
                                f"Original error: {type(e).__name__}: {e}. "
                                f"Conversion error: {type(conv_e).__name__}: {conv_e}"
                            ) from conv_e
                        finally:
                            # Only clean up lock file if we acquired it
                            if lock_acquired and lock_file_path.exists():
                                try:
                                    lock_file_path.unlink()
                                except:
                                    pass
                    else:
                        raise RuntimeError(
                            f"Failed to initialize transformer component. Error: {type(e).__name__}: {e}"
                        ) from e
                
                # Quick warm-up pass to confirm tokenizer/model are callable
                print(f"   Warming up transformer component...", flush=True)
                try:
                    warmup_doc = nlp("Warm-up sentence to initialize tokenizer.")
                    if warmup_doc is None:
                        raise RuntimeError("Warm-up failed: nlp() returned None")
                    print(f"   ✅ Transformer component warmed up successfully", flush=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to warm up transformer component. "
                        f"This may indicate the tokenizer is not properly configured. "
                        f"Error: {type(e).__name__}: {e}"
                    ) from e
                
                # Do not attach tagger/parser here; noun_chunks are handled by separate chunk_nlp to avoid transformer listener collisions
                
                # Move pipeline to GPU if available
                if use_gpu:
                    try:
                        nlp.to(device)
                        print(f"✅ Loaded HuggingFace model into spaCy pipeline (on {device})", flush=True)
                    except Exception as e:
                        print(f"⚠️  Failed to move model to {device}, using CPU: {e}", flush=True)
                        device = "cpu"
                        use_gpu = False
                else:
                    print(f"✅ Loaded HuggingFace model into spaCy pipeline (on {device})", flush=True)
                
            except ImportError as e:
                raise RuntimeError(
                    f"Required packages for HuggingFace integration not installed. "
                    f"Install with: pip install transformers spacy-transformers torch. Error: {e}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load HuggingFace model from {model_path_abs}. "
                    f"Error: {type(e).__name__}: {e}"
                ) from e
        elif model_format == "spacy":
            # Try loading as spaCy model
            try:
                nlp = spacy.load(model_path_abs)
                print(f"✅ Loaded {model_name} from GCS as spaCy model", flush=True)
                
                # Move pipeline to GPU if available
                if use_gpu:
                    try:
                        nlp.to(device)
                        print(f"✅ Moved {model_name} to {device}", flush=True)
                    except Exception as e:
                        print(f"⚠️  Failed to move model to {device}, using CPU: {e}", flush=True)
                        device = "cpu"
                        use_gpu = False
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model from {model_path_abs}. "
                    f"Expected either HuggingFace format (model.safetensors/config.json) or spaCy format (meta.json). Error: {e}"
                )
        else:
            raise RuntimeError(f"Unknown model format '{model_format}' for {model_path_abs}")
    else:
        # Use default spaCy transformer model
        model_name = "en_core_web_trf"
        print(f"🎯 Using default spaCy transformer model: {model_name}", flush=True)
        
        try:
            import spacy_curated_transformers  # Required for transformer models
        except ImportError:
            raise RuntimeError(
                "spacy-curated-transformers is required for transformer models. "
                "Install with: pip install spacy-curated-transformers"
            )
        
        # Try to load model first (might already be installed via spacy download)
        try:
            nlp = spacy.load(model_name)
            print(f"✅ Using existing {model_name} installation", flush=True)
        except (OSError, IOError):
            # Model not found locally, download from GCS - MUST SUCCEED
            print(f"📥 {model_name} not found locally, downloading from GCS...", flush=True)
            model_path = download_spacy_model_from_gcs(model_name)
            model_path_abs = str(Path(model_path).resolve())
            nlp = spacy.load(model_path_abs)
            print(f"✅ Loaded {model_name} from GCS", flush=True)
        
        # Move pipeline to GPU if available (works for spacy-curated-transformers models)
        if use_gpu:
            try:
                nlp.to(device)
                print(f"✅ Moved {model_name} to {device}", flush=True)
            except Exception as e:
                print(f"⚠️  Failed to move model to {device}, using CPU: {e}", flush=True)
                device = "cpu"
                use_gpu = False
    
    # Ensure we have all needed components
    has_sentencizer = 'sentencizer' in nlp.pipe_names or 'senter' in nlp.pipe_names
    if not has_sentencizer:
        if 'ner' in nlp.pipe_names:
            nlp.add_pipe('sentencizer', before='ner')
        else:
            nlp.add_pipe('sentencizer', first=True)
        print("✅ Added 'sentencizer' component to pipeline", flush=True)
    
    # Keep all components - don't disable any
    print(f"🚀 Keeping all pipeline components", flush=True)
    
    load_time = time.time() - start_time
    print(f"✅ spaCy loaded: {model_name} (loaded in {load_time:.2f}s, device: {device})", flush=True)
    print(f"   Active pipes: {', '.join(nlp.pipe_names)}\n", flush=True)
    
    return nlp

nlp = initialize_spacy()
nlp_lock = threading.Lock()  # Lock for thread-safe access to spaCy model

# Secondary parser-only pipeline for noun_chunks (kept separate to avoid transformer listener incompatibilities)
def initialize_chunk_spacy():
    """
    Initialize a lightweight spaCy pipeline for dependency parsing/noun_chunks.
    Uses a non-transformer backbone (default en_core_web_sm) to avoid listener collisions.
    MUST SUCCEED - no fallbacks.
    """
    import spacy
    from pathlib import Path
    
    parser_backbone_name = os.getenv("SPACY_CHUNK_BACKBONE", "en_core_web_sm")
    print(f"Loading chunking spaCy model for noun_chunks: {parser_backbone_name}", flush=True)
    
    try:
        chunk_nlp = spacy.load(parser_backbone_name)
    except (OSError, IOError):
        print(f"📥 {parser_backbone_name} not found locally, downloading from GCS...", flush=True)
        parser_model_path = download_spacy_model_from_gcs(parser_backbone_name)
        chunk_nlp = spacy.load(parser_model_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load chunking model {parser_backbone_name}. Error: {type(e).__name__}: {e}"
        ) from e
    
    if 'transformer' in chunk_nlp.pipe_names:
        raise RuntimeError(
            f"Chunking model {parser_backbone_name} includes a transformer pipe. "
            f"Use a non-transformer model (e.g., en_core_web_sm or en_core_web_md) "
            f"to avoid listener incompatibility."
        )
    
    # Sanity check: noun_chunks must work
    sanity_doc = chunk_nlp("Noun chunks require a dependency parse.")
    try:
        _ = list(sanity_doc.noun_chunks)
    except Exception as e:
        raise RuntimeError(
            f"Chunking model {parser_backbone_name} failed noun_chunks sanity check. "
            f"Error: {type(e).__name__}: {e}"
        ) from e
    
    return chunk_nlp

chunk_nlp = initialize_chunk_spacy()
chunk_nlp_lock = threading.Lock()

# ------------------------------------------------------------------
# 8.1 Chunked spaCy Processing Helper
# ------------------------------------------------------------------
def generate_spacy_docs(full_text: str, section_name: str = "") -> List[Tuple[any, int]]:
    """
    Yield spaCy docs over the text in overlapping character chunks to reduce peak memory usage.
    Returns list of (doc, chunk_offset).
    """
    if not full_text:
        return []
    
    chunk_size = int(os.getenv("SPACY_CHUNK_SIZE", "12000"))
    overlap = int(os.getenv("SPACY_CHUNK_OVERLAP", "800"))
    chunk_size = max(chunk_size, 2000)  # guardrail
    overlap = max(0, min(overlap, chunk_size // 2))
    
    spans = []
    start = 0
    text_len = len(full_text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Try to break on paragraph or sentence boundary for cleaner splits
            para_break = full_text.rfind("\n\n", start + chunk_size // 2, end)
            if para_break != -1 and para_break > start + 100:
                end = para_break
            else:
                sent_break = full_text.rfind(". ", start + chunk_size // 2, end)
                if sent_break != -1 and sent_break > start + 50:
                    end = sent_break + 2
        spans.append((start, end))
        if end == text_len:
            break
        start = max(end - overlap, start + 1)
    
    docs = []
    for idx, (span_start, span_end) in enumerate(spans, 1):
        chunk_text = full_text[span_start:span_end]
        print(f"   🧩 Processing chunk {idx}/{len(spans)} ({len(chunk_text):,} chars){' ' + section_name if section_name else ''}", flush=True)
        try:
            with nlp_lock:
                doc = nlp(chunk_text)
        except Exception as e:
            raise RuntimeError(
                f"Error calling nlp() on chunk {idx}/{len(spans)} for section {section_name or 'N/A'}: "
                f"{type(e).__name__}: {e}"
            ) from e
        docs.append((doc, span_start))
    return docs

# ------------------------------------------------------------------
# 3. Entity Clustering & Normalization
# ------------------------------------------------------------------
def normalize_entity(text: str) -> str:
    """Normalize entity text for comparison"""
    return text.strip().lower()

def fuzzy_match_score(str1: str, str2: str) -> float:
    """
    Calculate fuzzy match score between two strings using SequenceMatcher
    Returns score between 0.0 and 1.0
    """
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def find_matching_entity(text: str, entities: List[Dict], threshold: float = 0.85) -> Optional[Dict]:
    """
    Find best matching entity from list using fuzzy matching
    Args:
        text: Text to match
        entities: List of entity dicts with 'text' field
        threshold: Minimum similarity score (0.0-1.0)
    Returns:
        Best matching entity dict or None
    """
    if not text or not entities:
        return None
    
    normalized_text = normalize_entity(text)
    best_match = None
    best_score = threshold
    
    for entity in entities:
        entity_text = entity.get('text', '')
        if not entity_text:
            continue
        
        normalized_entity = normalize_entity(entity_text)
        
        # Exact match (after normalization)
        if normalized_text == normalized_entity:
            return entity
        
        # Check if one is substring of other
        if normalized_text in normalized_entity or normalized_entity in normalized_text:
            score = 0.95  # High score for substring matches
        else:
            # Fuzzy match
            score = fuzzy_match_score(normalized_text, normalized_entity)
        
        if score > best_score:
            best_score = score
            best_match = entity
    
    return best_match

def resolve_entity_reference(text: str, sentence_entities: List[Dict], all_entities: List[Dict] = None) -> Dict:
    """
    Resolve an entity text reference to an actual entity
    First tries sentence entities, then all entities with fuzzy matching
    If no match found, creates a new CONCEPT entity
    
    Args:
        text: Entity text to resolve
        sentence_entities: Entities from the same sentence
        all_entities: All entities in document (optional)
    
    Returns:
        Entity dict with at minimum: {'text': str, 'label': str}
    """
    if not text:
        return {'text': '', 'label': 'UNKNOWN'}
    
    # Try exact match in sentence entities first
    for ent in sentence_entities:
        if normalize_entity(ent.get('text', '')) == normalize_entity(text):
            return ent
    
    # Try fuzzy match in sentence entities
    match = find_matching_entity(text, sentence_entities, threshold=0.85)
    if match:
        return match
    
    # Try fuzzy match in all entities if provided
    if all_entities:
        match = find_matching_entity(text, all_entities, threshold=0.90)
        if match:
            return match
    
    # No match found - create new entity as CONCEPT
    # Infer label from context if possible
    label = 'CONCEPT'
    text_lower = text.lower()
    
    # Simple heuristic-based label inference
    if any(word in text_lower for word in ['company', 'corporation', 'inc', 'llc', 'ltd']):
        label = 'ORG'
    elif any(word in text_lower for word in ['risk', 'exposure', 'uncertainty']):
        label = 'RISK'
    elif any(word in text_lower for word in ['market', 'region', 'country', 'geographic']):
        label = 'GEOGRAPHY'
    
    return {
        'text': text[:100].strip(),  # Limit length
        'label': label,
        'source': 'relationship_inference',
        'start_char': -1,
        'end_char': -1
    }

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
# 4. 10-K Specific Entity Extractors (v3.0)
# ------------------------------------------------------------------
def extract_risk_entities(text: str, section_id: str = None) -> List[Dict]:
    """
    Extract risk factors as entities (primarily from Section 1A but applicable to all sections)
    Patterns: "X risk", "risk of X", "risks related to X", "exposed to X"
    """
    risk_entities = []
    seen_risks = set()  # Deduplicate within this extraction
    
    # Enhanced risk patterns
    risk_patterns = [
        # Direct risk mentions: "credit risk", "market risk", "liquidity risk"
        r'([a-z]+(?:\s+[a-z]+){0,3})\s+risks?(?:\s|,|\.|\))',
        # Risk of X: "risk of default", "risk of loss"
        r'risks?\s+(?:of|from|related to|associated with|arising from)\s+([a-z]+(?:\s+[a-z]+){0,3})',
        # Exposed/subject to: "exposed to currency fluctuations"
        r'(?:exposed|subject|vulnerable)\s+to\s+([a-z]+(?:\s+[a-z]+){0,4})',
        # Risk that/if patterns: "risk that rates increase"
        r'risks?\s+that\s+([a-z]+(?:\s+[a-z]+){0,3})',
    ]
    
    text_lower = text.lower()
    
    for pattern in risk_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            risk_text = match.group(1).strip()
            
            # Quality filters
            if (len(risk_text) < 5 or  # Too short
                len(risk_text) > 60 or  # Too long
                risk_text in seen_risks):  # Already found
                continue
            
            # Filter out generic terms
            if risk_text in {'the', 'our', 'such', 'these', 'other', 'certain', 'any'}:
                continue
            
            seen_risks.add(risk_text)
            risk_entities.append({
                'text': risk_text.title(),  # Capitalize for consistency
                'label': 'RISK',
                'source': 'pattern_extractor',
                'section': section_id if section_id else 'unknown'
            })
    
    return risk_entities

def extract_competitor_entities(text: str) -> List[Dict]:
    """
    Extract competitor mentions
    Patterns: "competition from X", "competitors such as X", "competes with X"
    """
    competitor_entities = []
    seen_competitors = set()
    
    # Competitor patterns
    competitor_patterns = [
        # Competition from X: "competition from Samsung", "competition from other companies"
        r'compet(?:ition|itors?|ing|es?)\s+(?:from|with|including|such as)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
        # Named competitors in lists: "Samsung, Google, and Microsoft"
        r'competitors?\s+(?:include|such as|including|like)\s+([A-Z][a-z]+(?:(?:,\s+|\s+and\s+)[A-Z][a-z]+)*)',
        # Direct competition statements
        r'compete\s+with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
    ]
    
    for pattern in competitor_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            competitor_text = match.group(1).strip()
            
            # Split comma-separated lists
            competitors = re.split(r',\s*|\s+and\s+', competitor_text)
            
            for comp in competitors:
                comp = comp.strip()
                
                # Quality filters
                if (len(comp) < 2 or 
                    len(comp) > 50 or
                    comp in seen_competitors):
                    continue
                
                # Must start with capital letter (proper noun)
                if not comp[0].isupper():
                    continue
                
                seen_competitors.add(comp)
                competitor_entities.append({
                    'text': comp,
                    'label': 'COMPETITOR',
                    'source': 'pattern_extractor'
                })
    
    return competitor_entities

def extract_regulation_entities(text: str) -> List[Dict]:
    """
    Extract regulatory references: laws, acts, regulatory bodies
    Patterns: "SEC", "Exchange Act", "Dodd-Frank", "GDPR", etc.
    """
    regulation_entities = []
    seen_regulations = set()
    
    # Known regulatory bodies and acts
    known_regulations = {
        'SEC', 'Securities and Exchange Commission',
        'Exchange Act', 'Securities Exchange Act',
        'Securities Act', 'Sarbanes-Oxley Act', 'SOX',
        'Dodd-Frank', 'Dodd-Frank Act',
        'GDPR', 'General Data Protection Regulation',
        'FCPA', 'Foreign Corrupt Practices Act',
        'FTC', 'Federal Trade Commission',
        'FCC', 'Federal Communications Commission',
        'FDA', 'Food and Drug Administration',
        'EPA', 'Environmental Protection Agency',
        'FINRA', 'Financial Industry Regulatory Authority',
        'FASB', 'Financial Accounting Standards Board',
        'GAAP', 'Generally Accepted Accounting Principles',
        'IFRS', 'International Financial Reporting Standards',
        'Basel III', 'Basel II',
        'CFTC', 'Commodity Futures Trading Commission'
    }
    
    # Pattern-based extraction for acts/regulations
    regulation_patterns = [
        # Act patterns: "Sarbanes-Oxley Act", "Exchange Act of 1934"
        r'([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)\s+Act(?:\s+of\s+\d{4})?',
        # Regulation patterns: "Regulation D", "Rule 144"
        r'(?:Regulation|Rule)\s+([A-Z0-9-]+)',
        # Code patterns: "Internal Revenue Code", "U.S. Code"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Code',
    ]
    
    # First, find known regulations (exact matches)
    text_upper = text.upper()
    for regulation in known_regulations:
        # Case-insensitive search but preserve original casing
        if regulation.upper() in text_upper:
            if regulation not in seen_regulations:
                seen_regulations.add(regulation)
                regulation_entities.append({
                    'text': regulation,
                    'label': 'REGULATION',
                    'source': 'pattern_extractor',
                    'subtype': 'known_regulation'
                })
    
    # Then, find pattern-based regulations
    for pattern in regulation_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            full_match = match.group(0).strip()
            
            # Quality filters
            if (len(full_match) < 3 or 
                len(full_match) > 80 or
                full_match in seen_regulations):
                continue
            
            seen_regulations.add(full_match)
            regulation_entities.append({
                'text': full_match,
                'label': 'REGULATION',
                'source': 'pattern_extractor',
                'subtype': 'pattern_match'
            })
    
    return regulation_entities

def extract_segment_entities(text: str) -> List[Dict]:
    """
    Extract business segment mentions
    Patterns: "iPhone segment", "Services business", "reportable segments"
    """
    segment_entities = []
    seen_segments = set()
    
    # Segment patterns
    segment_patterns = [
        # Direct segment mentions: "iPhone segment", "Mac segment"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+segment',
        # Business/division mentions: "Services business", "Hardware division"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:business|division|unit)',
        # Reportable segments: "reportable segments include X"
        r'reportable segments?\s+(?:include|are|consist of)\s+([A-Z][a-z]+(?:(?:,\s+|\s+and\s+)[A-Z][a-z]+)*)',
        # Operating segments
        r'operating segments?\s+(?:include|are|consist of)\s+([A-Z][a-z]+(?:(?:,\s+|\s+and\s+)[A-Z][a-z]+)*)',
    ]
    
    for pattern in segment_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            segment_text = match.group(1).strip()
            
            # Split comma-separated lists
            segments = re.split(r',\s*|\s+and\s+', segment_text)
            
            for seg in segments:
                seg = seg.strip()
                
                # Quality filters
                if (len(seg) < 2 or 
                    len(seg) > 50 or
                    seg in seen_segments):
                    continue
                
                # Must start with capital letter
                if not seg[0].isupper():
                    continue
                
                seen_segments.add(seg)
                segment_entities.append({
                    'text': seg,
                    'label': 'SEGMENT',
                    'source': 'pattern_extractor'
                })
    
    return segment_entities

def extract_geography_entities(text: str) -> List[Dict]:
    """
    Extract geographic segments and regions beyond standard NER
    Patterns: "Americas region", "Greater China", "EMEA", geographic segments
    """
    geography_entities = []
    seen_geographies = set()
    
    # Known geographic segments (common in 10-Ks)
    known_geo_segments = {
        'Americas', 'North America', 'South America', 'Latin America',
        'EMEA', 'Europe', 'Middle East', 'Africa',
        'APAC', 'Asia Pacific', 'Greater China', 'Asia-Pacific',
        'U.S.', 'United States', 'International',
        'Domestic', 'Worldwide'
    }
    
    # Geographic segment patterns
    geo_patterns = [
        # Region mentions: "Americas region", "EMEA segment"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:region|segment|market)',
        # Geographic references in context
        r'(?:operates in|present in|located in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    
    # First, find known geographic segments
    for geo in known_geo_segments:
        # Case-sensitive search for geographic names
        if geo in text:
            if geo not in seen_geographies:
                seen_geographies.add(geo)
                geography_entities.append({
                    'text': geo,
                    'label': 'GEOGRAPHY',
                    'source': 'pattern_extractor',
                    'subtype': 'known_segment'
                })
    
    # Then, find pattern-based geographies
    for pattern in geo_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            geo_text = match.group(1).strip()
            
            # Quality filters
            if (len(geo_text) < 3 or 
                len(geo_text) > 40 or
                geo_text in seen_geographies):
                continue
            
            seen_geographies.add(geo_text)
            geography_entities.append({
                'text': geo_text,
                'label': 'GEOGRAPHY',
                'source': 'pattern_extractor',
                'subtype': 'pattern_match'
            })
    
    return geography_entities

# ------------------------------------------------------------------
# 5. 10-K Specific Relationship Patterns (v3.0)
# ------------------------------------------------------------------
def extract_risk_relationships(sent, entities: List[Dict]) -> List[Dict]:
    """
    Extract risk exposure and mitigation relationships from 10-K text
    Patterns: "exposed to [risk]", "mitigate [risk]", "hedge against [risk]"
    
    IMPROVED v3.1: Uses entity resolution with fuzzy matching
    
    Args:
        sent: spaCy Span object (sentence)
        entities: List of entity dicts from this sentence
    """
    relationships = []
    
    # Risk exposure patterns
    exposure_verbs = {'exposed', 'subject', 'vulnerable', 'face', 'facing'}
    mitigation_verbs = {'mitigate', 'hedge', 'manage', 'reduce', 'offset', 'protect'}
    
    for token in sent:
        if token.lemma_.lower() in exposure_verbs:
            # Look for subject (usually "Company" or organization)
            subject_text = None
            for child in token.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    subject_text = ' '.join([t.text for t in child.subtree]).strip()
                    break
            
            # Look for risk object
            for child in token.children:
                if child.dep_ in ['prep']:
                    for pchild in child.children:
                        if pchild.dep_ == 'pobj':
                            risk_text = ' '.join([t.text for t in pchild.subtree]).strip()
                            
                            if subject_text and len(risk_text) > 3:
                                # Resolve entities using fuzzy matching
                                subject_ent = resolve_entity_reference(subject_text, entities)
                                risk_ent = resolve_entity_reference(risk_text, entities)
                                
                                relationships.append({
                                    'type': 'EXPOSES_TO_RISK',
                                    'source': subject_ent['text'],
                                    'source_label': subject_ent['label'],
                                    'target': risk_ent['text'],
                                    'target_label': risk_ent['label'],
                                    'method': 'risk_pattern_exposure_v3.1'
                                })
        
        elif token.lemma_.lower() in mitigation_verbs:
            # Find what mitigates the risk
            subject_text = None
            for child in token.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    subject_text = ' '.join([t.text for t in child.subtree]).strip()
                    break
            
            # Find risk being mitigated
            for child in token.children:
                if child.dep_ in ['dobj', 'prep']:
                    risk_text = None
                    if child.dep_ == 'dobj':
                        risk_text = ' '.join([t.text for t in child.subtree]).strip()
                    else:
                        for pchild in child.children:
                            if pchild.dep_ == 'pobj':
                                risk_text = ' '.join([t.text for t in pchild.subtree]).strip()
                                break
                    
                    if subject_text and risk_text and len(risk_text) > 3:
                        # Resolve entities using fuzzy matching
                        subject_ent = resolve_entity_reference(subject_text, entities)
                        risk_ent = resolve_entity_reference(risk_text, entities)
                        
                        relationships.append({
                            'type': 'MITIGATES_RISK',
                            'source': subject_ent['text'][:100],
                            'source_label': subject_ent['label'],
                            'target': risk_ent['text'][:100],
                            'target_label': risk_ent['label'],
                            'method': 'risk_pattern_mitigation_v3.1'
                        })
    
    return relationships

def extract_geographic_relationships(sent, entities: List[Dict]) -> List[Dict]:
    """
    Extract geographic operation relationships
    Patterns: "operates in [geography]", "segment includes [geography]"
    
    IMPROVED v3.1: Uses entity resolution with fuzzy matching
    """
    relationships = []
    
    operation_verbs = {'operate', 'sell', 'market', 'distribute', 'serve'}
    
    for token in sent:
        if token.lemma_.lower() in operation_verbs:
            subject_text = None
            for child in token.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    subject_text = ' '.join([t.text for t in child.subtree]).strip()
                    break
            
            # Look for geographic object
            for child in token.children:
                if child.dep_ == 'prep' and child.text.lower() in ['in', 'to', 'across']:
                    for pchild in child.children:
                        if pchild.dep_ == 'pobj':
                            geo_text = ' '.join([t.text for t in pchild.subtree]).strip()
                            
                            if subject_text and len(geo_text) > 2:
                                # Resolve entities using fuzzy matching
                                subject_ent = resolve_entity_reference(subject_text, entities)
                                geo_ent = resolve_entity_reference(geo_text, entities)
                                
                                relationships.append({
                                    'type': 'OPERATES_IN',
                                    'source': subject_ent['text'],
                                    'source_label': subject_ent['label'],
                                    'target': geo_ent['text'],
                                    'target_label': geo_ent['label'],
                                    'method': 'geographic_pattern_v3.1'
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
                # Resolve entities using fuzzy matching (v3.1 improvement)
                subject_ent = resolve_entity_reference(subject[:100], entities)
                obj_ent = resolve_entity_reference(obj[:100], entities)
                
                relationships.append({
                    'type': 'SVO_TRIPLE',
                    'source': subject_ent['text'],
                    'source_label': subject_ent['label'],
                    'target': obj_ent['text'],
                    'target_label': obj_ent['label'],
                    'verb': token.lemma_,
                    'verb_text': token.text,
                    'method': 'dependency_parse_v3.1'
                })
    
    return relationships

def extract_temporal_links(sent, entities: List[Dict]) -> List[Dict]:
    """
    Link entities/events to temporal expressions in the sentence
    OPTIMIZED: Only creates links for high-value entity types to reduce noise
    """
    relationships = []
    
    # High-value entity types worth linking to temporal context
    HIGH_VALUE_TYPES = {
        'ORG', 'PERSON', 'PRODUCT',  # Core business entities
        'GPE', 'LOC',  # Geographic entities
        'RISK', 'REGULATION',  # 10-K specific entities
        'MONEY', 'PERCENT'  # Financial metrics
    }
    
    # Find time/date entities
    temporal_entities = [ent for ent in entities if ent['label'] in ['DATE', 'TIME']]
    
    if temporal_entities:
        # Link ONLY high-value entities to temporal context (not CONCEPT, CARDINAL, etc.)
        non_temporal = [ent for ent in entities 
                       if ent['label'] not in ['DATE', 'TIME'] 
                       and ent['label'] in HIGH_VALUE_TYPES]
        
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
# 7. Concept Filtering Configuration (v3.0 - 10-K Optimized)
# ------------------------------------------------------------------
# Option B: Minimum mention threshold for displaying concepts in summary
MIN_CONCEPT_MENTIONS = 2  # Only show concepts mentioned at least this many times

# Option A: Blacklist of generic/boilerplate root words to filter out
CONCEPT_ROOT_BLACKLIST = {
    # Legal/regulatory boilerplate
    'section', 'item', 'table', 'form', 'report', 'filing', 'document',
    'paragraph', 'subsection', 'exhibit', 'amendment', 'page', 'appendix',
    
    # Generic business terms (too common in 10-Ks)
    'company', 'period', 'year', 'quarter', 'fiscal', 'information',
    'matter', 'portion', 'part', 'aspect', 'level', 'basis', 'time',
    
    # Vague descriptors
    'lot', 'thing', 'way', 'bit', 'sense', 'kind', 'sort', 'type',
    'couple', 'bunch', 'number', 'side', 'end', 'point', 'fact', 'case',
    'example', 'term', 'perspective', 'standpoint', 'context', 'regard',
    'respect', 'view', 'extent', 'degree', 'amount'
}

# Option D: Whitelist of valuable business/domain-specific root words
CONCEPT_ROOT_WHITELIST = {
    # Financial metrics & statements
    'revenue', 'margin', 'profit', 'income', 'expense', 'cost', 'earnings',
    'ebitda', 'cash', 'debt', 'equity', 'asset', 'liability', 'valuation',
    'growth', 'decline', 'loss', 'gain', 'return', 'investment', 'roi',
    'capex', 'opex', 'dividend', 'share', 'stock', 'warrant', 'derivative',
    
    # Risk & compliance
    'risk', 'exposure', 'hedge', 'mitigation', 'compliance', 'regulation',
    'regulatory', 'disclosure', 'materiality', 'contingency', 'litigation',
    'lawsuit', 'claim', 'settlement', 'penalty', 'fine', 'violation',
    
    # Business operations
    'customer', 'user', 'client', 'partner', 'supplier', 'vendor', 'employee',
    'workforce', 'capacity', 'production', 'manufacturing', 'deployment',
    'acquisition', 'merger', 'divestiture', 'subsidiary', 'joint', 'venture',
    
    # Products & Services
    'product', 'service', 'platform', 'solution', 'offering', 'feature',
    'technology', 'innovation', 'patent', 'trademark', 'copyright', 'license',
    'software', 'hardware', 'application', 'system', 'infrastructure',
    
    # Markets & Geography
    'market', 'segment', 'vertical', 'industry', 'sector', 'region',
    'geography', 'territory', 'jurisdiction', 'competition', 'competitor',
    'strategy', 'opportunity', 'demand', 'supply', 'trend', 'channel',
    
    # Performance metrics
    'performance', 'quality', 'efficiency', 'productivity', 'utilization',
    'retention', 'attrition', 'turnover', 'penetration', 'concentration',
    
    # Market risks
    'interest', 'currency', 'commodity', 'credit', 'liquidity', 'volatility',
    'fluctuation', 'sensitivity', 'correlation', 'diversification'
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
def enrich_entities_with_qwen3(sentences: List[str], batch_num: int = 0, section_name: str = "") -> List[Dict]:
    """
    Use Qwen3-Max to enrich raw sentences from SEC 10-K filings with entities.
    Returns list of dicts with 'entities' key containing enriched entity data.
    
    v3.0: Updated for 10-K filings with section-aware prompting
    """
    if not sentences:
        return []
    
    # Check if API key is set and client is initialized
    if not novita_client:
        return [{"entities": []} for _ in sentences]
    
    section_context = f" (from section: {section_name})" if section_name else ""
    
    prompt = f"""You are an expert SEC filing analyst. Extract ALL important entities from the following 10-K filing sentences{section_context}.

NOTE: Pattern-based extractors have already identified RISK, COMPETITOR, REGULATION, SEGMENT, and GEOGRAPHY entities. 
Focus on other entity types and only add these specialized types if you find ones the patterns missed.

Include these labels:
- PERSON, ORG, PRODUCT, TECHNOLOGY, LOCATION, DATE (prioritize these)
- METRIC (financial/operational metrics)
- CONCEPT (multi-word business terms)
- RISK (only if pattern extractor missed obvious ones)
- REGULATION (only if pattern extractor missed obvious ones)
- GEOGRAPHY (only if pattern extractor missed obvious ones)
- SEGMENT (only if pattern extractor missed obvious ones)
- COMPETITOR (only if pattern extractor missed obvious ones)

For each sentence, return entities as a JSON array. For each entity provide:
{{
  "text": str (entity text),
  "label": str (one of above),
  "description": str (4-8 word explanation),
  "confidence": float (0.0-1.0)
}}

Return a JSON object with "sentences" array, where each item has "entities" array.

Example format:
{{
  "sentences": [
    {{"entities": [{{"text": "foreign exchange rate risk", "label": "RISK", "description": "currency fluctuation exposure", "confidence": 0.92}}]}},
    {{"entities": [{{"text": "Americas", "label": "GEOGRAPHY", "description": "North and South America region", "confidence": 0.95}}]}}
  ]
}}

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
            timeout=120.0,  # 120 second timeout
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
def enrich_relationships_with_qwen3(sentences_data: List[Dict], entity_clusters: Dict, section_name: str = "") -> List[Dict]:
    """
    Extract rich relationships from SEC 10-K filings using LLM.
    Uses Qwen3-Max to find semantic relationships beyond syntactic patterns.
    
    v3.0 MIGRATION:
    - Updated for 10-K filings (formal regulatory text vs earnings calls)
    - New relationship types: EXPOSES_TO_RISK, OPERATES_IN, COMPETES_WITH, SUPPLIES
    - Section-aware prompting for better context
    """
    # Check if API key is set and client is initialized
    if not novita_client:
        print("⚠️  LLM relationship enrichment disabled - skipping", flush=True)
        return []
    
    # Sample 30 most entity-rich sentences
    rich_sentences = sorted(
        sentences_data,
        key=lambda x: len(x['entities']),
        reverse=True
    )[:30]
    
    if not rich_sentences:
        return []
    
    texts = [s['text'] for s in rich_sentences]
    section_context = f" from section '{section_name}'" if section_name else ""
    
    print(f"   Analyzing {len(texts)} entity-rich sentences{section_context}...", flush=True)
    
    # v3.0 PROMPT - SEC 10-K optimized
    prompt = f"""You are an expert SEC filing analyst extracting relationships from 10-K documents{section_context}.

Extract ALL directed relationships in this exact JSON format:

{{
  "relationships": [
    {{
      "type": "EXPOSES_TO_RISK|MITIGATES_RISK|OPERATES_IN|COMPETES_WITH|SUPPLIES|USES_TECHNOLOGY|CAUSES|DRIVES|RESULTS_IN|TARGETS|REGULATES",
      "source": "exact entity/phrase (subject)",
      "target": "exact entity/phrase (object)",
      "verb": "single verb used",
      "quote": "exact snippet proving the relationship (max 120 chars)",
      "sentiment": "positive|negative|neutral",
      "confidence": 0.XX
    }}
  ]
}}

Relationship Types Guide (10-K Specific):
- EXPOSES_TO_RISK: Company exposed to specific risk (e.g., "Company is exposed to foreign exchange risk")
- MITIGATES_RISK: Strategy/action mitigates risk (e.g., "hedging strategy mitigates interest rate risk")
- OPERATES_IN: Geographic/segment operations (e.g., "Company operates in Americas")
- COMPETES_WITH: Competitive relationships (e.g., "Company faces competition from Samsung")
- SUPPLIES: Supply chain (e.g., "suppliers provide components")
- USES_TECHNOLOGY: Technology utilization (e.g., "products use machine learning")
- CAUSES/DRIVES: Direct causation
- RESULTS_IN: Outcome relationship
- TARGETS: Forward-looking guidance
- REGULATES: Regulatory oversight (e.g., "SEC regulates disclosures")

Rules:
- Only output valid JSON
- Use exact phrases from text as source/target
- Do NOT hallucinate entities
- Confidence >= 0.65 required (higher threshold for formal text)
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
            timeout=120.0,  # 120 second timeout
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Clean markdown code blocks
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        
        data = json.loads(raw)
        rels = data.get("relationships", [])
        
        # Convert to standard format with v3.0 enhancements
        enriched = []
        for r in rels:
            confidence = r.get("confidence", 0.7)
            
            # v3.0: Threshold 0.65 for formal 10-K text (higher than earnings calls)
            if confidence >= 0.65:
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
                    "method": "qwen3_novita_v3.0"
                })
        
        print(f"   ✅ Extracted {len(enriched)} relationships (confidence >= 0.65)", flush=True)
        return enriched
        
    except Exception as e:
        print(f"   ⚠️  LLM relationship enrichment error: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return []

# ------------------------------------------------------------------
# 9. Single-Pass Sentence & Entity Extraction
# ------------------------------------------------------------------
def extract_sentences_and_entities(full_text: str, section_name: str = "") -> Tuple[List[Dict], any]:
    """
    Extract sentences AND entities in a single pass through the document
    Enhanced with LLM-based entity enrichment using Qwen3-Max
    Returns sentences with entities (spaCy NER + noun chunks + LLM enrichment)
    
    v3.0: Updated for 10-K filings with section-aware processing
    
    Concept Quality Filtering:
    - Option A: Blacklist generic/boilerplate root words (e.g., 'section', 'item', 'form')
    - Option D: Whitelist valuable business/domain root words (overrides blacklist)
    - Multi-word phrases only (2-8 tokens)
    - No overlap with NER entities
    - Note: Option B (minimum mentions) is applied during display, not extraction
    """
    global nlp, nlp_lock
    
    if nlp is None:
        raise RuntimeError("spaCy model (nlp) is not initialized. Call initialize_spacy() first.")
    
    if not callable(nlp):
        raise RuntimeError(f"spaCy model (nlp) is not callable. Type: {type(nlp)}, Value: {nlp}")
    
    section_context = f" ({section_name})" if section_name else ""
    print(f"📊 Pass 1a: Processing document with spaCy{section_context}...", flush=True)
    start_time = time.time()
    
    docs_with_offsets = generate_spacy_docs(full_text, section_name)
    
    processing_time = time.time() - start_time
    print(f"✅ spaCy processing completed in {processing_time:.2f}s\n", flush=True)
    
    # Extract sentences with their spaCy entities
    print("📊 Pass 1b: Extracting spaCy entities and concepts (chunked)...", flush=True)
    sentences = []
    concept_stats = {
        'total_chunks': 0,
        'filtered_chunks': 0,
        'filtered_by_quality': 0,  # Filtered by blacklist/whitelist
        'filtered_by_overlap': 0,   # Filtered by NER overlap
        'added_concepts': 0
    }
    
    sentence_texts = []  # Collect for batch LLM processing
    
    # First collect sentences and NER entities (no noun_chunks yet), across chunks
    last_sentence_end = -1
    for doc, chunk_offset in docs_with_offsets:
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            if not sentence_text:
                continue
            
            abs_start = chunk_offset + sent.start_char
            abs_end = chunk_offset + sent.end_char
            
            # Skip duplicate/overlapping sentence already captured from previous chunk overlap
            if abs_end <= last_sentence_end:
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
                    ent_abs_start = chunk_offset + ent.start_char
                    ent_abs_end = chunk_offset + ent.end_char
                    spacy_entities.append({
                        "text": entity_text,
                        "label": ent.label_,
                        "start_char": ent_abs_start,
                        "end_char": ent_abs_end,
                        "source": "spacy"
                    })
                    entity_spans.add((ent_abs_start, ent_abs_end))
            
            sentences.append({
                "sentence_id": len(sentences),
                "text": sentence_text,
                "start_char": abs_start,
                "end_char": abs_end,
                "spacy_entities": spacy_entities,
                "entities": [],  # Will be populated after LLM enrichment
                "sent_obj": sent,  # Store sentence span (has full parse info)
                "_entity_spans": entity_spans  # temp for noun chunk overlap checks
            })
            
            last_sentence_end = abs_end
    
    # Run noun-chunk parser in separate pipeline to avoid transformer listener conflicts
    chunk_doc = None
    try:
        with chunk_nlp_lock:
            chunk_doc = chunk_nlp(full_text)
    except Exception as e:
        raise RuntimeError(
            f"Error calling chunk_nlp() for noun_chunks: {type(e).__name__}: {e}"
        ) from e
    
    if chunk_doc is None or not chunk_doc.has_annotation("DEP"):
        raise RuntimeError("Chunking pipeline did not produce dependency annotations; noun_chunks unavailable.")
    
    # Map noun chunks to sentences by character span
    for chunk in chunk_doc.noun_chunks:
        concept_stats['total_chunks'] += 1
        chunk_text = chunk.text.strip()
        
        # Find sentence containing this chunk
        target_sentence = None
        for sent_data in sentences:
            if chunk.start_char >= sent_data["start_char"] and chunk.end_char <= sent_data["end_char"]:
                target_sentence = sent_data
                break
        if target_sentence is None:
            concept_stats['filtered_chunks'] += 1
            continue
        
        entity_spans = target_sentence["_entity_spans"]
        
        # Filter criteria
        if (len(chunk) >= 2 and 
            len(chunk) <= 8 and
            should_keep_concept(chunk)):
            
            # Check: not already covered by NER or overlapping
            if (chunk.start_char, chunk.end_char) in entity_spans:
                concept_stats['filtered_by_overlap'] += 1
                continue
            
            overlaps = False
            for (start, end) in entity_spans:
                if not (chunk.end_char <= start or chunk.start_char >= end):
                    overlaps = True
                    break
            
            if not overlaps:
                target_sentence["spacy_entities"].append({
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
            if len(chunk) < 2 or len(chunk) > 8:
                concept_stats['filtered_chunks'] += 1
            else:
                concept_stats['filtered_by_quality'] += 1
    
    # Cleanup temporary span storage
    for sent_data in sentences:
        sent_data.pop("_entity_spans", None)
    
    # Report concept extraction statistics
    total_filtered = (concept_stats['filtered_chunks'] + 
                     concept_stats['filtered_by_quality'] + 
                     concept_stats['filtered_by_overlap'])
    
    print(f"   Total noun chunks found: {concept_stats['total_chunks']}", flush=True)
    print(f"   ✅ Added as concepts: {concept_stats['added_concepts']}", flush=True)
    print(f"   ❌ Filtered out: {total_filtered}\n", flush=True)
    
    # 10-K Specific Pattern-Based Entity Extraction
    print("📊 Pass 1.5: Extracting 10-K specific entities (RISK, COMPETITOR, REGULATION, SEGMENT, GEOGRAPHY)...", flush=True)
    pattern_start = time.time()
    
    # Extract section ID if available (for section-aware extraction)
    # This will be passed from the calling function
    
    # Run specialized extractors on full text
    risk_entities = extract_risk_entities(full_text, section_name)
    competitor_entities = extract_competitor_entities(full_text)
    regulation_entities = extract_regulation_entities(full_text)
    segment_entities = extract_segment_entities(full_text)
    geography_entities = extract_geography_entities(full_text)
    
    # Combine all pattern-extracted entities
    pattern_entities = (
        risk_entities + 
        competitor_entities + 
        regulation_entities + 
        segment_entities + 
        geography_entities
    )
    
    # Add pattern-extracted entities to the first sentence (they're document-level, not sentence-specific)
    # Better approach: distribute them across sentences where they actually appear
    pattern_entity_map = defaultdict(list)  # Maps sentence text to entities found in it
    
    for entity in pattern_entities:
        entity_text = entity['text']
        # Find which sentences contain this entity
        for sent_data in sentences:
            if entity_text.lower() in sent_data['text'].lower():
                # Add to this sentence's entities
                pattern_entity_map[sent_data['sentence_id']].append(entity)
    
    # Merge pattern entities into sentence data
    pattern_entity_stats = {
        'RISK': 0,
        'COMPETITOR': 0,
        'REGULATION': 0,
        'SEGMENT': 0,
        'GEOGRAPHY': 0
    }
    
    for sent_data in sentences:
        sent_id = sent_data['sentence_id']
        if sent_id in pattern_entity_map:
            for pattern_ent in pattern_entity_map[sent_id]:
                # Add to spaCy entities (will be merged with LLM later)
                sent_data['spacy_entities'].append(pattern_ent)
                pattern_entity_stats[pattern_ent['label']] += 1
    
    pattern_time = time.time() - pattern_start
    print(f"✅ Pattern-based extraction completed in {pattern_time:.2f}s", flush=True)
    print(f"   RISK entities: {pattern_entity_stats['RISK']}", flush=True)
    print(f"   COMPETITOR entities: {pattern_entity_stats['COMPETITOR']}", flush=True)
    print(f"   REGULATION entities: {pattern_entity_stats['REGULATION']}", flush=True)
    print(f"   SEGMENT entities: {pattern_entity_stats['SEGMENT']}", flush=True)
    print(f"   GEOGRAPHY entities: {pattern_entity_stats['GEOGRAPHY']}", flush=True)
    print()
    
    # LLM Enrichment: Process in batches
    print("📊 Pass 1c: Enriching entities with Qwen3-Max...", flush=True)
    
    # Check if API key is available
    if not novita_client:
        print("⚠️  LLM enrichment disabled - using spaCy only", flush=True)
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
                enriched_batch = enrich_entities_with_qwen3(batch_texts, batch_num, section_name)
                
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
def extract_all_relationships(sentences_data: List[Dict], section_metadata: Dict = None) -> Dict:
    """
    Extract all relationships using hybrid approach
    
    v3.0: Removed speaker attribution, added 10-K specific relationships
    """
    print("📊 Pass 2: Extracting rule-based relationships...", flush=True)
    start_time = time.time()
    
    all_relationships = {
        'risk': [],
        'geographic': [],
        'temporal': [],
        'quantity': []
    }
    
    # Rule-based extraction per sentence (10-K specific)
    for sent_data in sentences_data:
        sent_text = sent_data['text']
        entities = sent_data['entities']
        sent = sent_data['sent_obj']
        
        # 10-K specific relationship extraction
        risk_rels = extract_risk_relationships(sent, entities)
        geo_rels = extract_geographic_relationships(sent, entities)
        temporal_rels = extract_temporal_links(sent, entities)
        quantity_rels = extract_quantity_links(sent, entities)
        
        # Store with sentence reference
        for rel in risk_rels:
            rel['sentence_id'] = sent_data['sentence_id']
            if section_metadata:
                rel['section'] = section_metadata.get('section', '')
                rel['section_name'] = section_metadata.get('section_name', '')
            all_relationships['risk'].append(rel)
        
        for rel in geo_rels:
            rel['sentence_id'] = sent_data['sentence_id']
            if section_metadata:
                rel['section'] = section_metadata.get('section', '')
                rel['section_name'] = section_metadata.get('section_name', '')
            all_relationships['geographic'].append(rel)
        
        for rel in temporal_rels:
            rel['sentence_id'] = sent_data['sentence_id']
            if section_metadata:
                rel['section'] = section_metadata.get('section', '')
                rel['section_name'] = section_metadata.get('section_name', '')
            all_relationships['temporal'].append(rel)
        
        for rel in quantity_rels:
            rel['sentence_id'] = sent_data['sentence_id']
            if section_metadata:
                rel['section'] = section_metadata.get('section', '')
                rel['section_name'] = section_metadata.get('section_name', '')
            all_relationships['quantity'].append(rel)
    
    processing_time = time.time() - start_time
    print(f"✅ Rule-based relationships extracted in {processing_time:.2f}s\n", flush=True)
    
    # Pass 3: Dependency-based extraction
    print("📊 Pass 3: Extracting dependency-based relationships...", flush=True)
    start_time = time.time()
    
    all_relationships['svo_triples'] = []
    verb_stats = {'total_verbs': 0, 'business_verbs': 0, 'complete_svo': 0}
    
    # Business verbs for 10-K filings
    business_verbs = {
        'operate', 'sell', 'market', 'distribute', 'manufacture', 'produce',
        'acquire', 'divest', 'license', 'supply', 'compete', 'expose', 'mitigate',
        'hedge', 'regulate', 'require', 'disclose', 'report', 'file', 'comply',
        'invest', 'expand', 'reduce', 'increase', 'decrease', 'generate', 'incur',
        'recognize', 'record', 'measure', 'estimate', 'assess', 'evaluate'
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
            if section_metadata:
                rel['section'] = section_metadata.get('section', '')
                rel['section_name'] = section_metadata.get('section_name', '')
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
    
    return all_relationships

# ------------------------------------------------------------------
# 10.5. Checkpoint Management (for standalone execution)
# ------------------------------------------------------------------
def load_checkpoint() -> Dict:
    """
    Load checkpoint from GCS PKL file.
    Returns dict with processed filings information.
    Compatible with batch_process_10k.py checkpoint format.
    """
    if not CHECKPOINT_ENABLED:
        return {'processed': {}, 'last_updated': None}
    
    try:
        blob = bucket.blob(CHECKPOINT_GCS_PATH)
        if blob.exists():
            checkpoint_bytes = blob.download_as_bytes()
            checkpoint_data = pickle.loads(checkpoint_bytes)
            print(f"✅ Loaded checkpoint from gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH}", flush=True)
            print(f"   Found {len(checkpoint_data.get('processed', {}))} processed filing(s)", flush=True)
            return checkpoint_data
        else:
            print(f"ℹ️  No checkpoint found, starting fresh", flush=True)
            return {'processed': {}, 'last_updated': None}
    except Exception as e:
        print(f"⚠️  Error loading checkpoint: {e}, starting fresh", flush=True)
        return {'processed': {}, 'last_updated': None}


def save_checkpoint(checkpoint_data: Dict):
    """
    Save checkpoint to GCS PKL file.
    Compatible with batch_process_10k.py checkpoint format.
    """
    if not CHECKPOINT_ENABLED:
        return
    
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
        print(f"💾 Checkpoint saved to gs://{GCS_BUCKET_NAME}/{CHECKPOINT_GCS_PATH} ({processed_count} filing(s), {len(checkpoint_bytes)} bytes)", flush=True)
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)


def update_checkpoint_entry(checkpoint_data: Dict, symbol: str, year: str, status: str, 
                            duration: float = None, error: str = None):
    """
    Update checkpoint entry for a specific filing.
    Compatible with batch_process_10k.py checkpoint format.
    Always reloads checkpoint from GCS to avoid race conditions with concurrent updates.
    NOTE: This function does NOT use file locking, so concurrent updates from different
    processes may still conflict. For best results, use batch_process_10k.py which handles
    checkpoint updates with proper locking.
    """
    if not CHECKPOINT_ENABLED:
        return
    
    # CRITICAL: Reload checkpoint from GCS to get latest state (prevents race conditions)
    # This ensures we don't overwrite concurrent updates from batch_process_10k.py or other instances
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
        print(f"⚠️  Failed to reload checkpoint from GCS, using passed data: {e}", flush=True)
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


# ------------------------------------------------------------------
# 11. Graph Database Persistence
# ------------------------------------------------------------------
def connect_to_memgraph(host='localhost', port=7687):
    """Connect to MemgraphDB (default port 7687 for Bolt protocol)"""
    print(f"Connecting to MemgraphDB at {host}:{port}...", flush=True)
    try:
        db = Memgraph(host=host, port=port)
        # Test connection
        db.execute("MATCH (n) RETURN count(n) LIMIT 1;")
        print("✅ Connected to MemgraphDB\n", flush=True)
        return db
    except Exception as e:
        print(f"❌ Failed to connect to MemgraphDB: {e}", flush=True)
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

def create_section_node(db: Memgraph, section: Dict, filing_year: Optional[str] = None):
    """Create a SECTION node (v3.0)"""
    query = """
    MERGE (s:SECTION {section_key: $section_key})
    SET s.section_id = $section_id,
        s.section_name = $section_name,
        s.ticker = $ticker,
        s.filing_year = $filing_year,
        s.company_name = $company_name,
        s.filing_date = $filing_date,
        s.period_of_report = $period_of_report,
        s.sentence_count = $sentence_count,
        s.char_count = $char_count
    RETURN s
    """
    
    metadata = section['metadata']
    year_val = filing_year or metadata.get('year') or metadata.get('period_of_report') or 'UNKNOWN'
    section_key = f"{metadata.get('ticker', 'UNKNOWN')}_{year_val}_{section['section_id']}"
    db.execute(query, {
        'section_key': section_key,
        'section_id': section['section_id'],
        'section_name': section['section_name'],
        'ticker': metadata.get('ticker', 'UNKNOWN'),
        'filing_year': year_val,
        'company_name': metadata.get('company_name', 'Unknown'),
        'filing_date': metadata.get('filed_at', ''),
        'period_of_report': metadata.get('period_of_report', ''),
        'sentence_count': section.get('sentence_count', 0),
        'char_count': section.get('char_count', 0)
    })

def auto_create_entity_if_missing(db: Memgraph, entity_name: str, entity_label: str, 
                                   entity_text_to_canonical: Dict) -> str:
    """
    Auto-create an entity if it doesn't exist in the graph
    Returns the canonical name (which is the same as entity_name for new entities)
    
    v3.1: New function to support auto-entity creation from relationships
    """
    normalized = normalize_entity(entity_name)
    
    # Check if already exists in mapping
    if normalized in entity_text_to_canonical:
        return entity_text_to_canonical[normalized]
    
    # Create new entity
    canonical_name = entity_name.strip()
    safe_label = entity_label.replace(' ', '_').replace('-', '_')
    
    try:
        query = f"""
        MERGE (e:{safe_label} {{canonical_name: $canonical_name}})
        SET e.entity_id = $entity_id,
            e.mention_count = 1,
            e.variants = [$canonical_name],
            e.inferred = true
        RETURN e
        """
        
        db.execute(query, {
            'canonical_name': canonical_name,
            'entity_id': f'INFERRED_{normalized[:50]}'
        })
        
        # Add to mapping
        entity_text_to_canonical[normalized] = canonical_name
        
        return canonical_name
        
    except Exception as e:
        # If creation fails, return original name (relationship will use it anyway)
        return canonical_name

def create_relationship(db: Memgraph, rel: Dict):
    """Create a relationship in the graph (v3.0 - 10-K enhanced)"""
    source = rel['source']
    target = rel['target']
    rel_type = rel['type'].upper().replace(' ', '_').replace('-', '_')
    
    # Build properties dict (only include non-None values)
    props = {}
    if 'verb' in rel and rel['verb']:
        props['verb'] = str(rel['verb'])[:100]
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
        props['quote'] = str(rel['quote'])[:200]
    # v3.0: Add section metadata
    if 'section' in rel and rel['section']:
        props['section'] = str(rel['section'])[:10]
    if 'section_name' in rel and rel['section_name']:
        props['section_name'] = str(rel['section_name'])[:100]
    
    # Create Cypher query - use CREATE instead of MERGE for better performance
    if props:
        props_str = ', '.join([f'{k}: ${k}' for k in props.keys()])
        props_clause = f"{{{props_str}}}"
    else:
        props_clause = ""
    
    # Standard relationship by canonical_name
    query = f"""
    MATCH (a {{canonical_name: $source}})
    MATCH (b {{canonical_name: $target}})
    CREATE (a)-[r:{rel_type} {props_clause}]->(b)
    RETURN r
    """
    
    result = db.execute_and_fetch(query, {
        'source': source,
        'target': target,
        **props
    })
    
    # Return True if relationship was created
    return list(result) is not None

def persist_to_memgraph(entities: Dict, relationships: Dict, sections: List[Dict],
                        host='localhost', port=7687, clear_existing=True, symbol=None, year=None):
    """
    Persist extracted entities and relationships to MemgraphDB
    
    v3.0: Added sections parameter for SECTION nodes (10-K filings)
          Fixed entity mapping for relationship creation
    v3.2: Added symbol/year parameters for selective clearing to support concurrent processing
    
    Args:
        entities: Dict of entity clusters from cluster_entities()
        relationships: Dict of relationship lists by type
        sections: List of section metadata (v3.0)
        host: MemgraphDB host
        port: MemgraphDB port
        clear_existing: If True, clear existing nodes related to this filing before inserting
        symbol: Stock symbol (e.g., AAPL) - used for selective clearing when concurrent processing
        year: Filing year (e.g., 2024) - used for selective clearing when concurrent processing
    """
    print("="*80)
    print("PERSISTING TO MEMGRAPHDB")
    print("="*80)
    print()
    
    lock_label = ((symbol and year and f"{symbol}_{year}") or (symbol or "GLOBAL")).upper()
    print(f"🔒 Acquiring Memgraph lock for {lock_label}...", flush=True)
    with acquire_symbol_lock(lock_label):
        print(f"✅ Lock acquired for {lock_label}", flush=True)
        
        db = connect_to_memgraph(host, port)
        if not db:
            print("❌ Skipping graph persistence - no database connection", flush=True)
            return False
        
        # Clear existing data (optional)
        # v3.2: Only clear nodes related to this specific filing to avoid transaction conflicts in concurrent processing
        if clear_existing:
            max_retries = 5
            retry_delay = 0.5  # Start with 0.5 seconds
            
            for attempt in range(max_retries):
                try:
                    if symbol:
                        # Selective clearing: only delete SECTION nodes for this filing (ticker + year)
                        print(f"🗑️  Clearing SECTION nodes for {symbol} {year} (attempt {attempt + 1}/{max_retries})...", flush=True)
                        query = """
                        MATCH (s:SECTION {ticker: $ticker, filing_year: $year})
                        DETACH DELETE s
                        """
                        db.execute(query, {'ticker': symbol, 'year': str(year)})
                    else:
                        # Fallback: clear all nodes if symbol not provided (original behavior)
                        print("🗑️  Clearing existing graph...", flush=True)
                        db.execute("MATCH (n) DETACH DELETE n;")
                    
                    print("✅ Graph cleared\n", flush=True)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if "conflicting transactions" in str(e).lower() or "retry" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            print(f"   ⚠️  Transaction conflict, retrying in {wait_time:.2f}s...", flush=True)
                            time.sleep(wait_time)
                        else:
                            print(f"   ❌ Failed to clear graph after {max_retries} attempts: {e}", flush=True)
                            raise
                    else:
                        # Non-retryable error
                        print(f"   ❌ Error clearing graph: {e}", flush=True)
                        raise
        
        # Build entity text → canonical name mapping for relationship creation
        print("📊 Building entity mapping (raw text → canonical names)...", flush=True)
        entity_text_to_canonical = {}
        
        for entity_id, entity in entities.items():
            canonical = entity['canonical_name']
            # Map canonical name to itself
            entity_text_to_canonical[normalize_entity(canonical)] = canonical
            
            # Map all variants to canonical name
            for variant in entity['variants']:
                entity_text_to_canonical[normalize_entity(variant)] = canonical
        
        print(f"✅ Built mapping for {len(entity_text_to_canonical)} entity variants\n", flush=True)
        
        # Create SECTION nodes (v3.0)
        if sections:
            print(f"📊 Creating {len(sections)} SECTION nodes...", flush=True)
            start_time = time.time()
            
            created_sections = 0
            for section in sections:
                try:
                    create_section_node(db, section, filing_year=str(year) if year else None)
                    created_sections += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to create section {section['section_id']}: {e}", flush=True)
            
            section_time = time.time() - start_time
            print(f"✅ Created {created_sections} SECTION nodes in {section_time:.2f}s\n", flush=True)
        
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
        
        # Create relationships with entity mapping (v3.1: improved with fuzzy matching + auto-creation)
        print(f"📊 Creating relationships (v3.1: fuzzy matching + auto-entity creation)...", flush=True)
        start_time = time.time()
        
        created_relationships = 0
        failed_relationships = 0
        auto_created_entities = 0
        fuzzy_matched = 0
        
        # Build fuzzy matching helper
        def find_best_match(text: str, mapping: Dict) -> Optional[str]:
            """Find best fuzzy match for entity text"""
            normalized = normalize_entity(text)
            
            # Try exact match first
            if normalized in mapping:
                return mapping[normalized]
            
            # Try fuzzy matching
            best_match = None
            best_score = 0.85  # Threshold
            
            for key, value in mapping.items():
                score = fuzzy_match_score(normalized, key)
                if score > best_score:
                    best_score = score
                    best_match = value
            
            return best_match
        
        # Process each relationship type
        for rel_type, rels in relationships.items():
            type_start = time.time()
            type_created = 0
            type_failed = 0
            type_auto_created = 0
            type_fuzzy_matched = 0
            
            print(f"   Processing {len(rels)} {rel_type} relationships...", flush=True)
            
            for rel in rels:
                source_text = rel['source']
                target_text = rel['target']
                
                # Try to find canonical names (exact match or fuzzy match)
                source_canonical = find_best_match(source_text, entity_text_to_canonical)
                target_canonical = find_best_match(target_text, entity_text_to_canonical)
                
                # Track fuzzy matching
                if source_canonical and normalize_entity(source_text) != normalize_entity(source_canonical):
                    type_fuzzy_matched += 1
                    fuzzy_matched += 1
                if target_canonical and normalize_entity(target_text) != normalize_entity(target_canonical):
                    type_fuzzy_matched += 1
                    fuzzy_matched += 1
                
                # Auto-create missing entities
                if not source_canonical:
                    source_label = rel.get('source_label', 'CONCEPT')
                    source_canonical = auto_create_entity_if_missing(
                        db, source_text, source_label, entity_text_to_canonical
                    )
                    type_auto_created += 1
                    auto_created_entities += 1
                
                if not target_canonical:
                    target_label = rel.get('target_label', 'CONCEPT')
                    target_canonical = auto_create_entity_if_missing(
                        db, target_text, target_label, entity_text_to_canonical
                    )
                    type_auto_created += 1
                    auto_created_entities += 1
                
                # Update relationship with canonical names
                rel_mapped = rel.copy()
                rel_mapped['source'] = source_canonical
                rel_mapped['target'] = target_canonical
                
                try:
                    success = create_relationship(db, rel_mapped)
                    if success:
                        created_relationships += 1
                        type_created += 1
                    else:
                        failed_relationships += 1
                        type_failed += 1
                except Exception as e:
                    failed_relationships += 1
                    type_failed += 1
            
            type_time = time.time() - type_start
            status_msg = f"      ✅ {type_created} created in {type_time:.2f}s"
            if type_auto_created > 0:
                status_msg += f" ({type_auto_created} entities auto-created)"
            if type_fuzzy_matched > 0:
                status_msg += f" ({type_fuzzy_matched} fuzzy matched)"
            if type_failed > 0:
                status_msg += f" ({type_failed} failed)"
            print(status_msg, flush=True)
        
        rel_time = time.time() - start_time
        print(f"\n✅ Total: {created_relationships} relationships created in {rel_time:.2f}s", flush=True)
        if auto_created_entities > 0:
            print(f"   🆕 {auto_created_entities} entities auto-created from relationships", flush=True)
        if fuzzy_matched > 0:
            print(f"   🎯 {fuzzy_matched} entities matched using fuzzy matching", flush=True)
        if failed_relationships > 0:
            print(f"   ⚠️  {failed_relationships} relationships failed (other errors)", flush=True)
        
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
        
        print(f"🔓 Released Memgraph lock for {lock_label}", flush=True)
        return True

# ------------------------------------------------------------------
# 12. Main Processing Pipeline
# ------------------------------------------------------------------
def process_10k_section(section_data: Dict, section_num: int, total_sections: int) -> Optional[Dict]:
    """
    Process a single 10-K section data (from GCS)
    Returns dict with section data, entities, and relationships
    """
    try:
        metadata = section_data.get('metadata', {})
        content = section_data.get('content', '')
        
        section_id = metadata.get('section', 'unknown')
        section_name = metadata.get('section_name', SECTION_NAMES.get(section_id, 'Unknown'))
        ticker = metadata.get('ticker', 'UNKNOWN')
        
        print(f"\n{'='*80}")
        print(f"PROCESSING SECTION {section_num}/{total_sections}: {ticker} - Section {section_id} ({section_name})")
        print(f"{'='*80}")
        print(f"   📄 Section: {section_id}.json")
        print(f"   📏 Length: {len(content):,} characters")
        print()
        
        # Extract sentences and entities
        sentences_data, full_doc = extract_sentences_and_entities(content, section_name)
        print(f"   Found {len(sentences_data)} sentences\n", flush=True)
        
        if not sentences_data:
            print(f"   ⚠️  No sentences found in section, skipping...\n", flush=True)
            return None
        
        # Extract relationships
        relationships = extract_all_relationships(sentences_data, metadata)
        
        return {
            'metadata': metadata,
            'section_id': section_id,
            'section_name': section_name,
            'sentences': sentences_data,
            'relationships': relationships,
            'char_count': len(content)
        }
        
    except Exception as e:
        print(f"   ❌ Error processing section {section_id}: {e}\n", flush=True)
        return None

def load_section_from_gcs(symbol: str, year: str, section: str) -> Optional[Dict]:
    """
    Load a single section JSON file from GCS
    Returns parsed JSON data or None if not found
    """
    gcs_path = safe_gcs_section_path(symbol, year, section)
    
    try:
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            return None
        
        # Download and parse JSON
        json_str = blob.download_as_text()
        section_data = json.loads(json_str)
        return section_data
        
    except Exception as e:
        print(f"   ⚠️  Error loading {gcs_path}: {e}", flush=True)
        return None

def list_sections_from_gcs(symbol: str, year: str) -> List[str]:
    """
    List all section files available in GCS for a given symbol and year
    Returns list of section IDs (e.g., ["1", "1A", "2", ...])
    """
    safe_symbol = validate_symbol(symbol)
    safe_year = str(validate_year(year))
    prefix = f"{GCS_BASE_PATH}/{safe_symbol}/{safe_year}/"
    sections = []
    
    try:
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith('.json'):
                # Extract section ID from filename (e.g., "1.json" -> "1")
                filename = os.path.basename(blob.name)
                section_id = sanitize_section_id(filename.replace('.json', ''))
                sections.append(section_id)
        
        # Sort sections in a logical order
        def section_sort_key(s: str) -> tuple:
            """Sort sections: 1, 1A, 1B, 1C, 2, 3, ..., 9, 9A, 9B, 9C, 10, ..."""
            if s.isdigit():
                return (int(s), '')
            else:
                # Split into number and letter (e.g., "1A" -> (1, "A"))
                match = re.match(r'(\d+)([A-Z]*)', s)
                if match:
                    num = int(match.group(1))
                    letter = match.group(2) or ''
                    return (num, letter)
                return (999, s)  # Put unknown formats at end
        
        sections.sort(key=section_sort_key)
        return sections
        
    except Exception as e:
        print(f"❌ Error listing sections from GCS: {e}", flush=True)
        return []

def process_documents(symbol: str = None, year: str = None):
    """
    Main pipeline: Extract entities and relationships from SEC 10-K filings
    
    v3.0: Reads from GCS bucket, processes one document (symbol + year) at a time
    """
    overall_start = time.time()
    
    # Get symbol and year from parameters or environment
    if symbol is None:
        symbol = SYMBOL
    if year is None:
        year = YEAR
    
    # Validate required parameters
    if not symbol:
        print("❌ ERROR: SYMBOL is required. Set SYMBOL environment variable or pass as argument.", flush=True)
        print("   Example: export SYMBOL=AAPL", flush=True)
        return None
    
    if not year:
        print("❌ ERROR: YEAR is required. Set YEAR environment variable or pass as argument.", flush=True)
        print("   Example: export YEAR=2024", flush=True)
        return None
    
    try:
        symbol = validate_symbol(symbol)
        year_int = validate_year(year)
        year = str(year_int)
    except ValueError as ve:
        print(f"❌ ERROR: {ve}", flush=True)
        return None
    
    print("="*80)
    print("ENTITY & RELATIONSHIP EXTRACTION PIPELINE (SEC 10-K v3.0)")
    print("="*80)
    print(f"📊 Processing: {symbol} - {year}")
    print(f"☁️  GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"📁 GCS Path: {GCS_BASE_PATH}/{symbol}/{year}/")
    print()
    
    # List available sections from GCS
    print(f"📋 Listing sections from GCS...", flush=True)
    load_start = time.time()
    
    section_ids = list_sections_from_gcs(symbol, year)
    
    if not section_ids:
        print(f"❌ No sections found in GCS for {symbol}/{year}", flush=True)
        print(f"   Path: {GCS_BASE_PATH}/{symbol}/{year}/", flush=True)
        return None
    
    load_time = time.time() - load_start
    print(f"✅ Found {len(section_ids)} section(s) in {load_time:.2f}s", flush=True)
    print(f"   Sections: {', '.join(section_ids)}\n", flush=True)
    
    # Load all sections from GCS
    print(f"📥 Loading sections from GCS...", flush=True)
    load_start = time.time()
    
    section_data_list = []
    for section_id in section_ids:
        section_data = load_section_from_gcs(symbol, year, section_id)
        if section_data:
            section_data_list.append(section_data)
        else:
            print(f"   ⚠️  Section {section_id} not found or failed to load", flush=True)
    
    if not section_data_list:
        print(f"❌ No sections were successfully loaded from GCS", flush=True)
        return None
    
    load_time = time.time() - load_start
    print(f"✅ Loaded {len(section_data_list)} section(s) in {load_time:.2f}s\n", flush=True)
    
    # Get parallel processing setting from environment
    use_parallel = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
    _CPU_COUNT = os.cpu_count() or 1
    _DEFAULT_MAX_WORKERS = max(_CPU_COUNT - 1, 1)
    max_workers = int(os.getenv("MAX_WORKERS", str(_DEFAULT_MAX_WORKERS)))
    
    concurrency_source = "env MAX_WORKERS" if os.getenv("MAX_WORKERS") else "auto (CPU-1)"
    if use_parallel and len(section_data_list) > 1:
        print(f"🔄 Processing sections in parallel (max {max_workers} workers) [{concurrency_source}]...\n", flush=True)
        
        section_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_section = {
                executor.submit(process_10k_section, section_data, i+1, len(section_data_list)): section_data
                for i, section_data in enumerate(section_data_list)
            }
            
            for future in as_completed(future_to_section):
                result = future.result()
                if result:
                    section_results.append(result)
    else:
        print(f"🔄 Processing sections sequentially...\n", flush=True)
        section_results = []
        for i, section_data in enumerate(section_data_list):
            result = process_10k_section(section_data, i+1, len(section_data_list))
            if result:
                section_results.append(result)
    
    if not section_results:
        print("❌ No sections were successfully processed", flush=True)
        return
    
    print(f"\n{'='*80}")
    print(f"MERGING RESULTS FROM {len(section_results)} SECTIONS")
    print(f"{'='*80}\n")
    
    # Merge all section data
    all_sentences = []
    all_relationships = {
        'risk': [],
        'geographic': [],
        'temporal': [],
        'quantity': [],
        'svo_triples': [],
        'comention': []
    }
    sections_metadata = []
    
    for section_result in section_results:
        # Track section metadata
        sections_metadata.append({
            'section_id': section_result['section_id'],
            'section_name': section_result['section_name'],
            'metadata': section_result['metadata'],
            'sentence_count': len(section_result['sentences']),
            'char_count': section_result['char_count']
        })
        
        # Merge sentences
        all_sentences.extend(section_result['sentences'])
        
        # Merge relationships
        for rel_type, rels in section_result['relationships'].items():
            if rel_type in all_relationships:
                all_relationships[rel_type].extend(rels)
    
    # Entity clustering and normalization
    print("📊 Clustering and normalizing entities across all sections...", flush=True)
    start_time = time.time()
    
    all_entities = []
    for sent_data in all_sentences:
        all_entities.extend(sent_data['entities'])
    
    entity_clusters = cluster_entities(all_entities)
    
    processing_time = time.time() - start_time
    print(f"✅ Entity clustering completed in {processing_time:.2f}s\n", flush=True)
    
    # LLM-based relationship enrichment (Pass 4)
    # Process each section's entity-rich sentences
    print("📊 Pass 4: Enriching relationships with Qwen3-Max...", flush=True)
    start_time = time.time()
    
    all_llm_relationships = []
    for section_result in section_results:
        section_name = section_result['section_name']
        llm_rels = enrich_relationships_with_qwen3(
            section_result['sentences'], 
            entity_clusters,
            section_name
        )
        all_llm_relationships.extend(llm_rels)
    
    all_relationships['llm_enriched'] = all_llm_relationships
    
    processing_time = time.time() - start_time
    print(f"✅ LLM relationship enrichment completed in {processing_time:.2f}s", flush=True)
    print(f"   Added {len(all_llm_relationships)} LLM-enriched relationships\n", flush=True)
    
    # Display summary statistics
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    
    # Section summary
    print(f"\n📊 SECTIONS PROCESSED:")
    print(f"   Total sections: {len(section_results)}")
    for section_meta in sections_metadata:
        print(f"      Section {section_meta['section_id']}: {section_meta['section_name']} "
              f"({section_meta['sentence_count']} sentences, {section_meta['char_count']:,} chars)")
    
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
    print(f"   Risk relationships: {len(all_relationships['risk'])}")
    print(f"   Geographic relationships: {len(all_relationships['geographic'])}")
    print(f"   SVO triples: {len(all_relationships['svo_triples'])}")
    print(f"   Temporal links: {len(all_relationships['temporal'])} (high-value entities only: ORG, PERSON, PRODUCT, GPE, RISK, REGULATION, MONEY, PERCENT)")
    print(f"   Quantity links: {len(all_relationships['quantity'])}")
    print(f"   Co-mention pairs: {len(all_relationships['comention'])}")
    print(f"   🤖 LLM-enriched (10-K specific): {len(all_relationships['llm_enriched'])}")
    
    total_relationships = sum(len(rels) for rels in all_relationships.values())
    print(f"   Total relationships: {total_relationships}")
    
    # Performance metrics
    overall_time = time.time() - overall_start
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Total execution time: {overall_time:.2f}s")
    print(f"   Processing speed: {len(all_sentences)/overall_time:.1f} sentences/sec")
    print(f"   Sections processed: {len(section_results)}")
    
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
    
    print(f"\n   Risk Relationships (showing first 15):")
    if all_relationships['risk']:
        for rel in all_relationships['risk'][:15]:
            section_info = f" [Section {rel.get('section', '?')}]" if 'section' in rel else ""
            print(f"      [{rel['source'][:40]}] --{rel['type']}--> [{rel['target'][:40]}]{section_info}")
    else:
        print("      (None found)")
    
    print(f"\n   Geographic Relationships (showing first 15):")
    if all_relationships['geographic']:
        for rel in all_relationships['geographic'][:15]:
            section_info = f" [Section {rel.get('section', '?')}]" if 'section' in rel else ""
            print(f"      [{rel['source']}] --{rel['type']}--> [{rel['target']}]{section_info}")
    else:
        print("      (None found)")
    
    print(f"\n   SVO Triples (showing first 20):")
    if all_relationships['svo_triples']:
        for rel in all_relationships['svo_triples'][:20]:
            section_info = f" [Sec {rel.get('section', '?')}]" if 'section' in rel else ""
            print(f"      [{rel['source'][:45]}] --{rel['verb_text']}--> [{rel['target'][:45]}]{section_info}")
    else:
        print("      (None found)")
    
    print(f"\n   Quantity Links (showing first 15):")
    if all_relationships['quantity']:
        for rel in all_relationships['quantity'][:15]:
            print(f"      [{rel['source']}] --> [{rel['target']}] ({rel['type']})")
    else:
        print("      (None found)")
    
    print(f"\n   Co-mentions (top 15 by frequency):")
    sorted_comentions = sorted(all_relationships['comention'], key=lambda x: x['count'], reverse=True)
    for rel in sorted_comentions[:15]:
        print(f"      [{rel['source'][:40]}] <--{rel['count']}x--> [{rel['target'][:40]}]")
    
    print(f"\n   🤖 LLM-Enriched Relationships (10-K specific, showing up to 25):")
    if all_relationships['llm_enriched']:
        for rel in all_relationships['llm_enriched'][:25]:
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
    graph_persistence_success = False
    if enable_graph_persistence:
        graph_persistence_success = persist_to_memgraph(
            entities=entity_clusters,
            relationships=all_relationships,
            sections=sections_metadata,  # v3.0: pass section metadata
            host=os.getenv('MEMGRAPH_HOST', 'localhost'),
            port=int(os.getenv('MEMGRAPH_PORT', '7687')),  # Default port 7687 for Bolt protocol
            clear_existing=os.getenv('MEMGRAPH_CLEAR_EXISTING', 'true').lower() == 'true',
            symbol=symbol,  # v3.2: pass symbol for selective clearing
            year=year  # v3.2: pass year for selective clearing
        )
    else:
        print("ℹ️  Graph persistence disabled (set ENABLE_GRAPH_PERSISTENCE=true to enable)\n", flush=True)
    
    # Update checkpoint if processing was successful
    # Only update if run standalone (not via batch_process_10k.py which handles its own checkpoint)
    # Check if we're being run by batch_process_10k.py by looking for a flag or parent process
    update_checkpoint = CHECKPOINT_ENABLED and graph_persistence_success
    
    # If batch_process_10k.py is running us, it will handle checkpoint updates
    # We only update checkpoint if run standalone
    # Check if SKIP_CHECKPOINT_UPDATE is set (batch_process_10k.py can set this)
    skip_checkpoint_update = os.getenv("SKIP_CHECKPOINT_UPDATE", "false").lower() == "true"
    
    if update_checkpoint and not skip_checkpoint_update:
        print("\n💾 Updating checkpoint...", flush=True)
        checkpoint_data = load_checkpoint()
        update_checkpoint_entry(
            checkpoint_data, 
            symbol, 
            year, 
            status='completed',
            duration=overall_time,
            error=None
        )
        print("✅ Checkpoint updated\n", flush=True)
    elif skip_checkpoint_update:
        print("ℹ️  Checkpoint update skipped (handled by batch processor)\n", flush=True)
    
    # Return data structures
    return {
        'sentences': all_sentences,
        'entities': entity_clusters,
        'relationships': all_relationships,
        'sections': sections_metadata,
        'stats': {
            'total_sections': len(section_results),
            'total_sentences': len(all_sentences),
            'total_entity_mentions': len(all_entities),
            'unique_entities': len(entity_clusters),
            'total_relationships': total_relationships,
            'processing_time_seconds': overall_time
        }
    }

# ------------------------------------------------------------------
# 13. Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract entities and relationships from SEC 10-K filings stored in GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variables:
  export SYMBOL=AAPL YEAR=2024
  python v3.0-prototype.py

  # Using command-line arguments:
  python v3.0-prototype.py --symbol AAPL --year 2024

  # Override environment variables:
  export SYMBOL=AAPL YEAR=2023
  python v3.0-prototype.py --year 2024  # Will use AAPL from env, 2024 from arg
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
        help='Year of the 10-K filing (e.g., 2024). Can also be set via YEAR environment variable.'
    )
    
    args = parser.parse_args()
    
    # Get symbol and year (command-line args override environment variables)
    symbol = args.symbol or SYMBOL
    year = args.year or YEAR
    
    # Validate
    if not symbol:
        print("❌ ERROR: SYMBOL is required.", flush=True)
        print("   Set via: --symbol AAPL or export SYMBOL=AAPL", flush=True)
        exit(1)
    
    if not year:
        print("❌ ERROR: YEAR is required.", flush=True)
        print("   Set via: --year 2024 or export YEAR=2024", flush=True)
        exit(1)
    
    # Load checkpoint to check if already processed (optional check)
    checkpoint_data = load_checkpoint()
    key = f"{symbol}_{year}"
    existing_entry = checkpoint_data.get('processed', {}).get(key)
    
    if existing_entry and existing_entry.get('status') == 'completed':
        print(f"ℹ️  Filing {symbol}-{year} is already marked as completed in checkpoint", flush=True)
        print(f"   Processed at: {existing_entry.get('processed_at', 'unknown')}", flush=True)
        response = input(f"   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled by user.", flush=True)
            exit(0)
        print()
    
    # Process documents
    result = process_documents(symbol=symbol, year=year)
    
    if result is None:
        print("❌ Processing failed or no data found", flush=True)
        
        # Update checkpoint with failure status if enabled
        if CHECKPOINT_ENABLED and not os.getenv("SKIP_CHECKPOINT_UPDATE", "false").lower() == "true":
            checkpoint_data = load_checkpoint()
            update_checkpoint_entry(
                checkpoint_data,
                symbol,
                year,
                status='failed',
                duration=None,
                error="Processing failed or no data found"
            )
        
        exit(1)
    
    # Optionally save to JSON for inspection
    output_file = f"extraction_results_{symbol}_{year}.json"
    print(f"💾 Saving results to {output_file}...", flush=True)
    
    # Prepare for JSON serialization (remove sent objects)
    json_result = {
        'symbol': symbol,
        'year': year,
        'sections': result['sections'],
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

