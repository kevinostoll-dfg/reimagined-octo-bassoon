#!/usr/bin/env python3
"""
Process a single random Form 4 filing from GCS using spaCy for entity extraction.
Multi-step workflow: Fetch → Parse → Extract → Relate → Persist → Output
Persists data to MemgraphDB and outputs in Cypher format and human-readable text.
"""

import os
import sys
import json
import random
import spacy
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import urllib3
import logging
from google.cloud import storage
from dotenv import load_dotenv
from gqlalchemy import Memgraph
import re

# Suppress urllib3 connection pool warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

# Load environment variables
load_dotenv('env')

# Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "blacksmith-sec-filings")
GCS_BASE_PATH = os.getenv("GCS_BASE_PATH", "form-4-filings")

# Fine-tuned model configuration (optional)
FINE_TUNED_MODEL_GCS_PATH = os.getenv("FINE_TUNED_MODEL_GCS_PATH", "")
FINE_TUNED_MODEL_NAME = os.getenv("FINE_TUNED_MODEL_NAME", "")

# Use fine-tuned model if specified, otherwise default to transformer model
if FINE_TUNED_MODEL_NAME:
    SPACY_MODEL = FINE_TUNED_MODEL_NAME
else:
    SPACY_MODEL = "en_core_web_trf"

# spaCy Model GCS Configuration
SPACY_MODELS_BUCKET = os.getenv("SPACY_MODELS_BUCKET", "blacksmith-sec-filings")
SPACY_MODELS_GCS_PATH = os.getenv("SPACY_MODELS_GCS_PATH", "spacy-models")

# Memgraph Configuration
MEMGRAPH_HOST = os.getenv("MEMGRAPH_HOST", "localhost")
MEMGRAPH_PORT = int(os.getenv("MEMGRAPH_PORT", "7687"))
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "memgraph")
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "memgraph")

# Initialize GCS client
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    print(f"✅ GCS client initialized")
except Exception as e:
    print(f"❌ Failed to initialize GCS client: {e}")
    sys.exit(1)

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
        FINE_TUNED_MODEL_GCS_PATH, detected_model_name = model_info
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
                            SPACY_MODEL = potential_model_name
                            print(f"✅ Auto-detected model name: {FINE_TUNED_MODEL_NAME}", flush=True)
                            break
            except Exception as e:
                print(f"⚠️  Could not auto-detect model name: {e}", flush=True)
        else:
            # Update SPACY_MODEL if FINE_TUNED_MODEL_NAME was set via env
            if FINE_TUNED_MODEL_NAME:
                SPACY_MODEL = FINE_TUNED_MODEL_NAME

# Initialize Memgraph connection (will connect when needed)
# Use thread-local storage for thread-safe connections
import threading
memgraph_db_local = threading.local()

# ============================================================================
# MODEL DOWNLOAD - Download spaCy model from GCS
# ============================================================================

def download_spacy_model_from_gcs(model_name: str) -> str:
    """
    Download spaCy model from GCS bucket to persistent cache directory.
    Returns path to downloaded model directory.
    
    Args:
        model_name: Model name (e.g., "en_core_web_trf", "en_core_web_sm")
        
    Returns:
        Path to model directory
        
    Raises:
        RuntimeError: If download fails
    """
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
    if model_cache_path.exists() and (model_cache_path / "meta.json").exists():
        print(f"✅ {model_name} found in cache: {model_cache_path}", flush=True)
        return str(model_cache_path)
    
    # Create cache directory for this model
    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize GCS client
        gcs_client = storage.Client()
        model_bucket = gcs_client.bucket(SPACY_MODELS_BUCKET)
        
        # Use fine-tuned model path if specified, otherwise use default spacy-models path
        if FINE_TUNED_MODEL_NAME and model_name == FINE_TUNED_MODEL_NAME and FINE_TUNED_MODEL_GCS_PATH:
            gcs_prefix = f"{FINE_TUNED_MODEL_GCS_PATH}/{model_name}/"
        else:
            gcs_prefix = f"{SPACY_MODELS_GCS_PATH}/{model_name}/"
        
        # List all files in the model directory
        blobs = list(model_bucket.list_blobs(prefix=gcs_prefix))
        
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


def initialize_spacy_model():
    """
    Initialize spaCy model with required pipeline components.
    Downloads model from GCS if not available locally.
    MUST SUCCEED - no fallbacks.
    """
    print("Loading spaCy model...", flush=True)
    start_time = time.time()
    
    # Print fine-tuned model info if using one
    if FINE_TUNED_MODEL_NAME:
        print(f"🎯 Using fine-tuned model: {SPACY_MODEL}", flush=True)
    
    # Import spacy and spacy-curated-transformers (required for transformer models)
    # Note: en_core_web_trf uses spacy-curated-transformers, not spacy-transformers
    import spacy
    try:
        import spacy_curated_transformers  # Required for transformer models - registers factories
    except ImportError:
        raise RuntimeError(
            "spacy-curated-transformers is required for transformer models. "
            "Install with: pip install spacy-curated-transformers"
        )
    
    # Try to load model first (might already be installed via spacy download)
    try:
        nlp = spacy.load(SPACY_MODEL)
        print(f"✅ Using existing {SPACY_MODEL} installation", flush=True)
    except (OSError, IOError):
        # Model not found locally, download from GCS - MUST SUCCEED
        print(f"📥 {SPACY_MODEL} not found locally, downloading from GCS...", flush=True)
        model_path = download_spacy_model_from_gcs(SPACY_MODEL)
        
        # Load model from downloaded path using spacy.load()
        # Must use absolute path
        model_path_abs = str(Path(model_path).resolve())
        
        # Try loading - if it fails due to architecture issues, the error will be raised
        # (no fallbacks per user requirement)
        nlp = spacy.load(model_path_abs)
        print(f"✅ Loaded {SPACY_MODEL} from GCS", flush=True)
    
    # Ensure we have all needed components
    has_sentencizer = 'sentencizer' in nlp.pipe_names or 'senter' in nlp.pipe_names
    if not has_sentencizer:
        if 'ner' in nlp.pipe_names:
            nlp.add_pipe('sentencizer', before='ner')
        else:
            nlp.add_pipe('sentencizer', first=True)
        print("✅ Added 'sentencizer' component to pipeline", flush=True)
    
    # Keep all components - don't disable any
    # Note: In transformer models, disabling attribute_ruler or lemmatizer breaks POS tagging
    # This is because the transformer pipeline has dependencies between components
    print(f"🚀 Keeping all pipeline components (required for POS tagging in transformer models)", flush=True)
    
    load_time = time.time() - start_time
    print(f"✅ spaCy loaded: {SPACY_MODEL} (loaded in {load_time:.2f}s)", flush=True)
    print(f"   Active pipes: {', '.join(nlp.pipe_names)}\n", flush=True)
    
    return nlp

# Load spaCy model (will download from GCS if needed)
try:
    nlp = initialize_spacy_model()
except Exception as e:
    print(f"❌ Failed to load spaCy model: {e}")
    sys.exit(1)


# ============================================================================
# STEP 1: FETCH - Get random document from GCS
# ============================================================================

def list_all_filings() -> List[str]:
    """List all filing paths in GCS."""
    print(f"\n{'='*80}")
    print("STEP 1: FETCH - Listing all filings...")
    print(f"{'='*80}")
    
    prefix = f"{GCS_BASE_PATH}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Filter for JSON files only
    json_files = [blob.name for blob in blobs if blob.name.endswith('.json')]
    
    print(f"   Found {len(json_files)} total filings")
    return json_files


def fetch_random_document(filing_paths: List[str]) -> Tuple[Dict, str]:
    """Fetch a random document from GCS."""
    if not filing_paths:
        raise ValueError("No filings found in GCS")
    
    # Select random file
    random_path = random.choice(filing_paths)
    print(f"   🎲 Selected random file: {random_path}")
    
    # Download from GCS
    blob = bucket.blob(random_path)
    content = blob.download_as_text()
    filing_data = json.loads(content)
    
    print(f"   ✅ Document fetched successfully")
    return filing_data, random_path


def fetch_document_by_path(file_path: str) -> Tuple[Dict, str]:
    """Fetch a specific document from GCS by path."""
    # Download from GCS
    blob = bucket.blob(file_path)
    if not blob.exists():
        raise ValueError(f"File not found in GCS: {file_path}")
    
    content = blob.download_as_text()
    filing_data = json.loads(content)
    
    return filing_data, file_path


# ============================================================================
# STEP 2: PARSE - Extract structured data from JSON
# ============================================================================

def parse_filing(filing: Dict, output_enabled: bool = True) -> Dict:
    """Parse filing JSON into structured components."""
    if output_enabled:
        print(f"\n{'='*80}")
        print("STEP 2: PARSE - Extracting structured data...")
        print(f"{'='*80}")
    
    parsed = {
        "metadata": {
            "accession_no": filing.get("accessionNo", ""),
            "filed_at": filing.get("filedAt", ""),
            "period_of_report": filing.get("periodOfReport", ""),
            "document_type": filing.get("documentType", ""),
        },
        "issuer": filing.get("issuer", {}),
        "reporting_owner": filing.get("reportingOwner", {}),
        "transactions": [],
        "derivative_transactions": [],
        "footnotes": filing.get("footnotes", []),
        "remarks": filing.get("remarks", ""),
    }
    
    # Parse non-derivative transactions
    non_deriv = filing.get("nonDerivativeTable", {})
    if non_deriv:
        parsed["transactions"] = non_deriv.get("transactions", [])
        parsed["holdings"] = non_deriv.get("holdings", [])
    
    # Parse derivative transactions
    deriv = filing.get("derivativeTable", {})
    if deriv:
        parsed["derivative_transactions"] = deriv.get("transactions", [])
        parsed["derivative_holdings"] = deriv.get("holdings", [])
    
    if output_enabled:
        print(f"   ✅ Parsed:")
        print(f"      Metadata: {parsed['metadata']['accession_no']}")
        print(f"      Transactions: {len(parsed['transactions'])}")
        print(f"      Derivative transactions: {len(parsed['derivative_transactions'])}")
        print(f"      Footnotes: {len(parsed['footnotes'])}")
    
    return parsed


# ============================================================================
# STEP 3: EXTRACT - Use spaCy to extract entities
# ============================================================================

def extract_text_from_filing(parsed: Dict) -> str:
    """Extract all text content from parsed filing for NLP processing."""
    text_parts = []
    
    # Issuer info
    issuer = parsed.get("issuer", {})
    if issuer.get("name"):
        text_parts.append(f"Issuer: {issuer['name']}")
    if issuer.get("tradingSymbol"):
        text_parts.append(f"Trading Symbol: {issuer['tradingSymbol']}")
    
    # Reporting owner info
    owner = parsed.get("reporting_owner", {})
    if owner.get("name"):
        text_parts.append(f"Reporting Owner: {owner['name']}")
    
    # Relationship info
    relationship = owner.get("relationship", {})
    if relationship.get("officerTitle"):
        text_parts.append(f"Title: {relationship['officerTitle']}")
    if relationship.get("isDirector"):
        text_parts.append("Position: Director")
    if relationship.get("isOfficer"):
        text_parts.append("Position: Officer")
    if relationship.get("isTenPercentOwner"):
        text_parts.append("Position: 10% Owner")
    
    # Transactions (non-derivative)
    for txn in parsed.get("transactions", []):
        if txn.get("securityTitle"):
            text_parts.append(f"Security: {txn['securityTitle']}")
        if txn.get("transactionDate"):
            text_parts.append(f"Transaction Date: {txn['transactionDate']}")
        if txn.get("amounts"):
            amounts = txn["amounts"]
            if amounts.get("shares"):
                text_parts.append(f"Shares: {amounts['shares']}")
            if amounts.get("pricePerShare"):
                text_parts.append(f"Price per Share: ${amounts['pricePerShare']}")
        if txn.get("coding", {}).get("code"):
            text_parts.append(f"Transaction Code: {txn['coding']['code']}")
    
    # Transactions (derivative)
    for txn in parsed.get("derivative_transactions", []):
        if txn.get("securityTitle"):
            text_parts.append(f"Derivative Security: {txn['securityTitle']}")
        if txn.get("transactionDate"):
            text_parts.append(f"Derivative Transaction Date: {txn['transactionDate']}")
        if txn.get("amounts"):
            amounts = txn["amounts"]
            if amounts.get("shares"):
                text_parts.append(f"Derivative Shares: {amounts['shares']}")
            if amounts.get("pricePerShare"):
                text_parts.append(f"Derivative Price per Share: ${amounts['pricePerShare']}")
        if txn.get("coding", {}).get("code"):
            text_parts.append(f"Derivative Transaction Code: {txn['coding']['code']}")
        if txn.get("underlyingSecurity"):
            underlying = txn["underlyingSecurity"]
            if underlying.get("title"):
                text_parts.append(f"Underlying Security: {underlying['title']}")
    
    # Footnotes
    for footnote in parsed.get("footnotes", []):
        if footnote.get("text"):
            text_parts.append(f"Footnote: {footnote['text']}")
    
    # Remarks
    if parsed.get("remarks"):
        text_parts.append(f"Remarks: {parsed['remarks']}")
    
    return " ".join(text_parts)


def extract_entities(text: str, parsed: Dict, output_enabled: bool = True) -> Dict:
    """Extract entities using spaCy."""
    if output_enabled:
        print(f"\n{'='*80}")
        print("STEP 3: EXTRACT - Using spaCy for entity extraction...")
        print(f"{'='*80}")
    
    doc = nlp(text)
    
    entities = {
        "persons": [],
        "organizations": [],
        "dates": [],
        "money": [],
        "quantities": [],
        "transaction_codes": [],
        "positions": [],
    }
    
    # Extract person name from structured data (more reliable than NLP)
    owner = parsed.get("reporting_owner", {})
    owner_name = owner.get("name", "")
    if owner_name:
        # Process name with spaCy to extract person entity
        name_doc = nlp(owner_name)
        for ent in name_doc.ents:
            if ent.label_ == "PERSON":
                entities["persons"].append({
                    "text": ent.text,
                    "start": 0,
                    "end": len(ent.text),
                    "label": ent.label_,
                    "source": "structured_data"
                })
        # If spaCy didn't catch it, add it manually
        if not any(p["text"] == owner_name for p in entities["persons"]):
            entities["persons"].append({
                "text": owner_name,
                "start": 0,
                "end": len(owner_name),
                "label": "PERSON",
                "source": "structured_data"
            })
    
    # Extract standard spaCy entities from text
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Avoid duplicates
            if not any(p["text"] == ent.text for p in entities["persons"]):
                entities["persons"].append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_,
                    "source": "nlp_extraction"
                })
        elif ent.label_ == "ORG":
            entities["organizations"].append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
        elif ent.label_ == "DATE":
            entities["dates"].append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
        elif ent.label_ == "MONEY":
            entities["money"].append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
    
    # Extract domain-specific entities from structured data
    # Transaction codes (non-derivative)
    for txn in parsed.get("transactions", []):
        code = txn.get("coding", {}).get("code")
        if code:
            entities["transaction_codes"].append({
                "code": code,
                "transaction_date": txn.get("transactionDate", ""),
                "security": txn.get("securityTitle", ""),
                "type": "non_derivative"
            })
    
    # Transaction codes (derivative)
    for txn in parsed.get("derivative_transactions", []):
        code = txn.get("coding", {}).get("code")
        if code:
            entities["transaction_codes"].append({
                "code": code,
                "transaction_date": txn.get("transactionDate", ""),
                "security": txn.get("securityTitle", ""),
                "type": "derivative"
            })
    
    # Positions/titles
    relationship = parsed.get("reporting_owner", {}).get("relationship", {})
    if relationship.get("officerTitle"):
        entities["positions"].append({
            "title": relationship["officerTitle"],
            "is_director": relationship.get("isDirector", False),
            "is_officer": relationship.get("isOfficer", False),
            "is_ten_percent": relationship.get("isTenPercentOwner", False)
        })
    
    # Extract share quantities and prices from structured data (non-derivative)
    for txn in parsed.get("transactions", []):
        amounts = txn.get("amounts", {})
        if amounts.get("shares"):
            entities["quantities"].append({
                "shares": amounts["shares"],
                "transaction_date": txn.get("transactionDate", ""),
                "security": txn.get("securityTitle", ""),
                "type": "non_derivative"
            })
        if amounts.get("pricePerShare"):
            entities["money"].append({
                "text": f"${amounts['pricePerShare']}",
                "type": "price_per_share",
                "transaction_date": txn.get("transactionDate", ""),
                "security": txn.get("securityTitle", ""),
                "transaction_type": "non_derivative"
            })
    
    # Extract share quantities and prices from structured data (derivative)
    for txn in parsed.get("derivative_transactions", []):
        amounts = txn.get("amounts", {})
        underlying = txn.get("underlyingSecurity", {})
        if amounts.get("shares"):
            entities["quantities"].append({
                "shares": amounts["shares"],
                "transaction_date": txn.get("transactionDate", ""),
                "security": txn.get("securityTitle", ""),
                "underlying_security": underlying.get("title", ""),
                "type": "derivative"
            })
        if amounts.get("pricePerShare"):
            entities["money"].append({
                "text": f"${amounts['pricePerShare']}",
                "type": "price_per_share",
                "transaction_date": txn.get("transactionDate", ""),
                "security": txn.get("securityTitle", ""),
                "transaction_type": "derivative"
            })
    
    if output_enabled:
        print(f"   ✅ Extracted entities:")
        print(f"      Persons: {len(entities['persons'])}")
        print(f"      Organizations: {len(entities['organizations'])}")
        print(f"      Dates: {len(entities['dates'])}")
        print(f"      Money amounts: {len(entities['money'])}")
        print(f"      Quantities: {len(entities['quantities'])}")
        print(f"      Transaction codes: {len(entities['transaction_codes'])}")
        print(f"      Positions: {len(entities['positions'])}")
    
    return entities


# ============================================================================
# STEP 4: RELATE - Build relationships for MemgraphDB
# ============================================================================

def build_relationships(parsed: Dict, entities: Dict, output_enabled: bool = True) -> List[Dict]:
    """Build relationships in MemgraphDB/Cypher format."""
    if output_enabled:
        print(f"\n{'='*80}")
        print("STEP 4: RELATE - Building relationships for MemgraphDB...")
        print(f"{'='*80}")
    
    relationships = []
    
    # Get core entities
    issuer = parsed.get("issuer", {})
    owner = parsed.get("reporting_owner", {})
    owner_name = owner.get("name", "")
    relationship_info = owner.get("relationship", {})
    accession_no = parsed["metadata"]["accession_no"]
    
    # Relationship: Insider - FILED -> Transaction (non-derivative)
    for i, txn in enumerate(parsed.get("transactions", [])):
        txn_date = txn.get("transactionDate", "")
        code = txn.get("coding", {}).get("code", "")
        security = txn.get("securityTitle", "")
        amounts = txn.get("amounts", {})
        
        relationships.append({
            "type": "FILED",
            "from": {
                "type": "Insider",
                "properties": {
                    "cik": owner.get("cik", ""),
                    "name": owner_name,
                    "normalized_name": owner_name.replace(",", "").title()
                }
            },
            "to": {
                "type": "Transaction",
                "properties": {
                    "accession_no": accession_no,
                    "transaction_date": txn_date,
                    "code": code,
                    "security_type": security,
                    "shares": amounts.get("shares", 0),
                    "price_per_share": amounts.get("pricePerShare", 0),
                    "acquired_disposed": amounts.get("acquiredDisposedCode", ""),
                    "transaction_type": "non_derivative"
                }
            }
        })
        
        # Relationship: Transaction - INVOLVES -> Company
        relationships.append({
            "type": "INVOLVES",
            "from": {
                "type": "Transaction",
                "properties": {
                    "accession_no": accession_no,
                    "transaction_date": txn_date,
                    "transaction_type": "non_derivative"
                }
            },
            "to": {
                "type": "Company",
                "properties": {
                    "cik": issuer.get("cik", ""),
                    "symbol": issuer.get("tradingSymbol", ""),
                    "name": issuer.get("name", "")
                }
            }
        })
    
    # Relationship: Insider - FILED -> Transaction (derivative)
    for i, txn in enumerate(parsed.get("derivative_transactions", [])):
        txn_date = txn.get("transactionDate", "")
        code = txn.get("coding", {}).get("code", "")
        security = txn.get("securityTitle", "")
        amounts = txn.get("amounts", {})
        underlying = txn.get("underlyingSecurity", {})
        
        relationships.append({
            "type": "FILED",
            "from": {
                "type": "Insider",
                "properties": {
                    "cik": owner.get("cik", ""),
                    "name": owner_name,
                    "normalized_name": owner_name.replace(",", "").title()
                }
            },
            "to": {
                "type": "Transaction",
                "properties": {
                    "accession_no": accession_no,
                    "transaction_date": txn_date,
                    "code": code,
                    "security_type": security,
                    "shares": amounts.get("shares", 0),
                    "price_per_share": amounts.get("pricePerShare", 0),
                    "acquired_disposed": amounts.get("acquiredDisposedCode", ""),
                    "transaction_type": "derivative",
                    "underlying_security": underlying.get("title", ""),
                    "underlying_shares": underlying.get("shares", 0)
                }
            }
        })
        
        # Relationship: Transaction - INVOLVES -> Company
        relationships.append({
            "type": "INVOLVES",
            "from": {
                "type": "Transaction",
                "properties": {
                    "accession_no": accession_no,
                    "transaction_date": txn_date,
                    "transaction_type": "derivative"
                }
            },
            "to": {
                "type": "Company",
                "properties": {
                    "cik": issuer.get("cik", ""),
                    "symbol": issuer.get("tradingSymbol", ""),
                    "name": issuer.get("name", "")
                }
            }
        })
    
    # Relationship: Insider - HOLDS_POSITION -> Company
    # Always create this if we have relationship info (even without transactions)
    if relationship_info.get("isDirector") or relationship_info.get("isOfficer") or relationship_info.get("isTenPercentOwner"):
        # Check if we already have this relationship (avoid duplicates)
        position_exists = any(
            rel["type"] == "HOLDS_POSITION" and
            rel["from"]["properties"]["cik"] == owner.get("cik", "") and
            rel["to"]["properties"]["cik"] == issuer.get("cik", "")
            for rel in relationships
        )
        
        if not position_exists:
            relationships.append({
                "type": "HOLDS_POSITION",
                "from": {
                    "type": "Insider",
                    "properties": {
                        "cik": owner.get("cik", ""),
                        "name": owner_name,
                        "normalized_name": owner_name.replace(",", "").title()
                    }
                },
                "to": {
                    "type": "Company",
                    "properties": {
                        "cik": issuer.get("cik", ""),
                        "symbol": issuer.get("tradingSymbol", ""),
                        "name": issuer.get("name", "")
                    }
                },
                "properties": {
                    "is_director": relationship_info.get("isDirector", False),
                    "is_officer": relationship_info.get("isOfficer", False),
                    "is_ten_percent_owner": relationship_info.get("isTenPercentOwner", False),
                    "officer_title": relationship_info.get("officerTitle", "")
                }
            })
    
    if output_enabled:
        print(f"   ✅ Built {len(relationships)} relationships")
    elif len(relationships) == 0:
        # Log warning if no relationships found (even when output disabled)
        import logging
        logging.getLogger(__name__).warning(f"   ⚠️  No relationships built for filing")
    
    return relationships


def escape_cypher_value(value):
    """Escape value for Cypher query."""
    if isinstance(value, str):
        # Escape quotes and wrap in quotes
        escaped = value.replace("'", "\\'").replace('"', '\\"')
        return f"'{escaped}'"
    elif isinstance(value, bool):
        return str(value).lower()
    elif value is None:
        return "null"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return str(value)


def generate_cypher_queries(relationships: List[Dict]) -> List[str]:
    """Generate Cypher queries from relationships."""
    cypher_queries = []
    
    for rel in relationships:
        from_node = rel["from"]
        to_node = rel["to"]
        rel_type = rel["type"]
        rel_props = rel.get("properties", {})
        
        # Build FROM node properties string with actual values
        from_props_list = []
        for k, v in from_node["properties"].items():
            if v is not None and v != "":  # Include non-None, non-empty values (0 is valid)
                from_props_list.append(f"{k}: {escape_cypher_value(v)}")
        from_props = ", ".join(from_props_list)
        from_var = from_node['type'].lower()
        from_match = f"({from_var}:{from_node['type']} {{{from_props}}})"
        
        # Build TO node properties string with actual values
        to_props_list = []
        for k, v in to_node["properties"].items():
            if v is not None and v != "":  # Include non-None, non-empty values (0 is valid)
                to_props_list.append(f"{k}: {escape_cypher_value(v)}")
        to_props = ", ".join(to_props_list)
        to_var = to_node['type'].lower()
        to_match = f"({to_var}:{to_node['type']} {{{to_props}}})"
        
        # Build relationship properties string
        rel_props_str = ""
        if rel_props:
            rel_props_list = []
            for k, v in rel_props.items():
                if v is not None and v != "":  # Include non-None, non-empty values (False is valid)
                    rel_props_list.append(f"{k}: {escape_cypher_value(v)}")
            if rel_props_list:
                rel_props_str = " {" + ", ".join(rel_props_list) + "}"
        
        # Generate MERGE query
        query = f"""
MERGE {from_match}
MERGE {to_match}
MERGE ({from_var})-[:{rel_type}{rel_props_str}]->({to_var})
"""
        cypher_queries.append(query.strip())
    
    return cypher_queries


# ============================================================================
# STEP 5: PERSIST - Save to MemgraphDB
# ============================================================================

def connect_to_memgraph() -> Memgraph:
    """Connect to Memgraph database. Uses thread-local storage for thread safety."""
    # Use thread-local storage so each thread has its own connection
    if not hasattr(memgraph_db_local, 'db') or memgraph_db_local.db is None:
        try:
            # Only print connection message once per thread
            if not hasattr(memgraph_db_local, 'connected'):
                print(f"\n{'='*80}")
                print("Connecting to MemgraphDB...")
                print(f"{'='*80}")
                print(f"   Host: {MEMGRAPH_HOST}")
                print(f"   Port: {MEMGRAPH_PORT}")
            
            memgraph_db_local.db = Memgraph(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT)
            
            # Test connection
            memgraph_db_local.db.execute("MATCH (n) RETURN count(n) LIMIT 1;")
            
            if not hasattr(memgraph_db_local, 'connected'):
                print(f"   ✅ Connected successfully")
                memgraph_db_local.connected = True
            
        except Exception as e:
            memgraph_db_local.db = None
            print(f"   ❌ Failed to connect to MemgraphDB: {e}")
            print(f"\n   Make sure MemgraphDB is running:")
            print(f"     cd graph-datasources/memgraph_docker")
            print(f"     docker-compose up -d")
            raise
    
    return memgraph_db_local.db


def persist_to_memgraph(cypher_queries: List[str], relationships: List[Dict], output_enabled: bool = True) -> Dict:
    """Persist relationships to MemgraphDB."""
    if output_enabled:
        print(f"\n{'='*80}")
        print("STEP 5: PERSIST - Saving to MemgraphDB...")
        print(f"{'='*80}")
    
    stats = {
        "executed": 0,
        "failed": 0,
        "errors": []
    }
    
    if not cypher_queries:
        return stats
    
    try:
        db = connect_to_memgraph()
    except Exception as e:
        error_msg = f"Failed to connect to MemgraphDB: {str(e)}"
        stats["errors"].append(error_msg)
        stats["failed"] = len(cypher_queries)
        if output_enabled:
            print(f"   ❌ {error_msg}")
        return stats
    
    if output_enabled:
        print(f"   Executing {len(cypher_queries)} Cypher queries...")
    
    # Limit number of queries to prevent hanging on large batches
    max_queries = 50  # Process max 50 queries at a time to prevent hangs
    queries_to_process = cypher_queries[:max_queries]
    
    for i, query in enumerate(queries_to_process, 1):
        try:
            db.execute(query)
            stats["executed"] += 1
            if output_enabled and (i % 10 == 0 or i == len(queries_to_process)):
                print(f"      [{i}/{len(queries_to_process)}] Executed successfully")
        except Exception as e:
            stats["failed"] += 1
            error_msg = f"Query {i} failed: {str(e)}"
            stats["errors"].append(error_msg)
            if output_enabled:
                print(f"      ⚠️  {error_msg}")
            # If we get connection errors, stop trying
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                error_msg = f"Connection issue detected, stopping persistence after {i} queries"
                stats["errors"].append(error_msg)
                break
    
    if len(cypher_queries) > max_queries:
        stats["errors"].append(f"Only processed {max_queries} of {len(cypher_queries)} queries (limit to prevent hanging)")
    
    if output_enabled:
        print(f"\n   ✅ Persistence complete:")
        print(f"      Executed: {stats['executed']}")
        print(f"      Failed: {stats['failed']}")
        
        if stats["errors"]:
            print(f"\n   ⚠️  Errors encountered:")
            for error in stats["errors"][:5]:  # Show first 5 errors
                print(f"      - {error}")
            if len(stats["errors"]) > 5:
                print(f"      ... and {len(stats['errors']) - 5} more")
    
    return stats


# ============================================================================
# STEP 6: OUTPUT - Format and display results
# ============================================================================

def output_results(parsed: Dict, entities: Dict, relationships: List[Dict], file_path: str, persist_stats: Optional[Dict] = None):
    """Output results in both JSON and human-readable format."""
    print(f"\n{'='*80}")
    print("STEP 6: OUTPUT - Results")
    print(f"{'='*80}")
    
    # Generate Cypher queries for output display
    cypher_queries = generate_cypher_queries(relationships)
    
    # Build comprehensive output
    output = {
        "source_file": file_path,
        "metadata": parsed["metadata"],
        "issuer": parsed["issuer"],
        "reporting_owner": parsed["reporting_owner"],
        "entities": entities,
        "relationships": relationships,
        "cypher_queries": cypher_queries,
        "transactions": parsed["transactions"],
        "derivative_transactions": parsed["derivative_transactions"],
        "footnotes": parsed["footnotes"],
        "remarks": parsed.get("remarks", ""),
        "persist_stats": persist_stats
    }
    
    # Human-readable output
    print("\n" + "="*80)
    print("HUMAN-READABLE OUTPUT")
    print("="*80)
    
    print(f"\n📄 Document: {file_path}")
    print(f"   Accession No: {parsed['metadata']['accession_no']}")
    print(f"   Filed At: {parsed['metadata']['filed_at']}")
    print(f"   Period of Report: {parsed['metadata']['period_of_report']}")
    
    issuer = parsed["issuer"]
    print(f"\n🏢 Company:")
    print(f"   Name: {issuer.get('name', 'N/A')}")
    print(f"   Symbol: {issuer.get('tradingSymbol', 'N/A')}")
    print(f"   CIK: {issuer.get('cik', 'N/A')}")
    
    owner = parsed["reporting_owner"]
    relationship = owner.get("relationship", {})
    print(f"\n👤 Insider:")
    print(f"   Name: {owner.get('name', 'N/A')}")
    print(f"   CIK: {owner.get('cik', 'N/A')}")
    if relationship.get("officerTitle"):
        print(f"   Title: {relationship['officerTitle']}")
    print(f"   Director: {relationship.get('isDirector', False)}")
    print(f"   Officer: {relationship.get('isOfficer', False)}")
    print(f"   10% Owner: {relationship.get('isTenPercentOwner', False)}")
    
    print(f"\n📊 Transactions: {len(parsed['transactions'])} (non-derivative)")
    for i, txn in enumerate(parsed["transactions"][:5], 1):  # Show first 5
        amounts = txn.get("amounts", {})
        print(f"   {i}. {txn.get('securityTitle', 'N/A')}")
        print(f"      Date: {txn.get('transactionDate', 'N/A')}")
        print(f"      Code: {txn.get('coding', {}).get('code', 'N/A')}")
        print(f"      Shares: {amounts.get('shares', 'N/A')}")
        print(f"      Price: ${amounts.get('pricePerShare', 'N/A')}")
    if len(parsed["transactions"]) > 5:
        print(f"   ... and {len(parsed['transactions']) - 5} more")
    
    print(f"\n📊 Derivative Transactions: {len(parsed['derivative_transactions'])}")
    for i, txn in enumerate(parsed["derivative_transactions"][:5], 1):  # Show first 5
        amounts = txn.get("amounts", {})
        underlying = txn.get("underlyingSecurity", {})
        print(f"   {i}. {txn.get('securityTitle', 'N/A')}")
        print(f"      Date: {txn.get('transactionDate', 'N/A')}")
        print(f"      Code: {txn.get('coding', {}).get('code', 'N/A')}")
        print(f"      Shares: {amounts.get('shares', 'N/A')}")
        print(f"      Price: ${amounts.get('pricePerShare', 'N/A')}")
        if underlying.get("title"):
            print(f"      Underlying: {underlying['title']}")
    if len(parsed["derivative_transactions"]) > 5:
        print(f"   ... and {len(parsed['derivative_transactions']) - 5} more")
    
    print(f"\n🔍 Extracted Entities:")
    print(f"   Persons: {len(entities['persons'])}")
    for person in entities['persons'][:3]:
        print(f"      - {person['text']}")
    print(f"   Organizations: {len(entities['organizations'])}")
    for org in entities['organizations'][:3]:
        print(f"      - {org['text']}")
    print(f"   Dates: {len(entities['dates'])}")
    print(f"   Money amounts: {len(entities['money'])}")
    print(f"   Share quantities: {len(entities['quantities'])}")
    
    print(f"\n🔗 Relationships: {len(relationships)}")
    for i, rel in enumerate(relationships[:5], 1):  # Show first 5
        from_type = rel["from"]["type"]
        to_type = rel["to"]["type"]
        rel_type = rel["type"]
        print(f"   {i}. {from_type} -[{rel_type}]-> {to_type}")
    if len(relationships) > 5:
        print(f"   ... and {len(relationships) - 5} more")
    
    if parsed["footnotes"]:
        print(f"\n📝 Footnotes: {len(parsed['footnotes'])}")
        for i, footnote in enumerate(parsed["footnotes"][:3], 1):
            text = footnote.get("text", "")[:100]
            print(f"   {i}. {text}...")
    
    if persist_stats:
        print(f"\n💾 MemgraphDB Persistence:")
        print(f"   Queries executed: {persist_stats['executed']}")
        print(f"   Queries failed: {persist_stats['failed']}")
        if persist_stats['executed'] > 0:
            print(f"   ✅ Data successfully persisted to MemgraphDB")
    
    # JSON output
    print("\n" + "="*80)
    print("JSON OUTPUT")
    print("="*80)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    
    # Cypher queries output
    print("\n" + "="*80)
    print("CYPHER QUERIES (for MemgraphDB)")
    print("="*80)
    for i, query in enumerate(cypher_queries[:10], 1):  # Show first 10
        print(f"\n-- Query {i}")
        print(query)
    if len(cypher_queries) > 10:
        print(f"\n... and {len(cypher_queries) - 10} more queries")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def process_single_document(file_path: str, output_enabled: bool = True) -> Dict:
    """
    Process a single Form 4 document by file path.
    Returns dict with processing results.
    
    Args:
        file_path: GCS path to the filing JSON file
        output_enabled: Whether to print output (default: True)
        
    Returns:
        Dict with keys: status, file_path, duration, error, relationships_count, etc.
    """
    start_time = time.time()
    result = {
        'file_path': file_path,
        'status': 'running',
        'start_time': start_time,
        'duration': None,
        'error': None,
        'relationships_count': 0,
        'entities_count': 0,
        'persist_stats': None
    }
    
    try:
        step_start = time.time()
        # Step 1: Fetch
        filing_data, _ = fetch_document_by_path(file_path)
        fetch_time = time.time() - step_start
        if not output_enabled:
            import logging
            logging.getLogger(__name__).debug(f"  ⏱️  Fetch: {fetch_time:.2f}s")
        
        step_start = time.time()
        # Step 2: Parse
        parsed = parse_filing(filing_data, output_enabled=output_enabled)
        parse_time = time.time() - step_start
        if not output_enabled:
            import logging
            logging.getLogger(__name__).debug(f"  ⏱️  Parse: {parse_time:.2f}s")
        
        step_start = time.time()
        # Step 3: Extract
        text = extract_text_from_filing(parsed)
        entities = extract_entities(text, parsed, output_enabled=output_enabled)
        result['entities_count'] = sum(len(v) if isinstance(v, list) else 0 for v in entities.values())
        extract_time = time.time() - step_start
        if not output_enabled:
            import logging
            logging.getLogger(__name__).debug(f"  ⏱️  Extract: {extract_time:.2f}s")
        
        step_start = time.time()
        # Step 4: Relate
        relationships = build_relationships(parsed, entities, output_enabled=output_enabled)
        result['relationships_count'] = len(relationships)
        relate_time = time.time() - step_start
        if not output_enabled:
            import logging
            logging.getLogger(__name__).debug(f"  ⏱️  Relate: {relate_time:.2f}s")
        
        step_start = time.time()
        # Step 5: Persist to MemgraphDB
        cypher_queries = generate_cypher_queries(relationships)
        persist_stats = None
        if cypher_queries:
            try:
                persist_stats = persist_to_memgraph(cypher_queries, relationships, output_enabled=output_enabled)
                result['persist_stats'] = persist_stats
            except Exception as e:
                if output_enabled:
                    print(f"\n⚠️  Warning: Could not persist to MemgraphDB: {e}")
                result['persist_error'] = str(e)
        persist_time = time.time() - step_start
        if not output_enabled:
            import logging
            logging.getLogger(__name__).debug(f"  ⏱️  Persist: {persist_time:.2f}s")
        
        # Step 6: Output (if enabled)
        if output_enabled:
            output_results(parsed, entities, relationships, file_path, persist_stats)
        
        result['status'] = 'completed'
        result['duration'] = time.time() - start_time
        
        return result
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['duration'] = time.time() - start_time
        if output_enabled:
            print(f"\n❌ Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
        return result


def main():
    """Main workflow execution."""
    print("="*80)
    print("Form 4 Filing Processor - spaCy Entity Extraction")
    print("="*80)
    print(f"GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"Base Path: {GCS_BASE_PATH}")
    print(f"spaCy Model: {SPACY_MODEL}")
    print()
    
    try:
        # Step 1: Fetch
        filing_paths = list_all_filings()
        filing_data, file_path = fetch_random_document(filing_paths)
        
        # Step 2: Parse
        parsed = parse_filing(filing_data)
        
        # Step 3: Extract
        text = extract_text_from_filing(parsed)
        entities = extract_entities(text, parsed)
        
        # Step 4: Relate
        relationships = build_relationships(parsed, entities)
        
        # Step 5: Persist to MemgraphDB
        cypher_queries = generate_cypher_queries(relationships)
        persist_stats = None
        if cypher_queries:
            try:
                persist_stats = persist_to_memgraph(cypher_queries, relationships)
            except Exception as e:
                print(f"\n⚠️  Warning: Could not persist to MemgraphDB: {e}")
                print(f"   Continuing with output only...")
        
        # Step 6: Output
        output_results(parsed, entities, relationships, file_path, persist_stats)
        
        print("\n" + "="*80)
        print("✅ Processing complete!")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
