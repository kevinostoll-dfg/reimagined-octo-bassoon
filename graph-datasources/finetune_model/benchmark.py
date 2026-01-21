"""
Comprehensive benchmark and validation script for fine-tuned SEC-BERT model.

This script provides detailed evaluation metrics, robustness testing, and
performance analysis to validate model quality before deployment.

Usage:
    python benchmark.py --model_path ./fine-tuned-model --test_csv dev_labeled.csv --num_labels 2788
    python benchmark.py --model_path ./fine-tuned-model --test_csv dev_labeled.csv --num_labels 2788 --output_dir ./benchmark_results
"""

import os
import sys
import argparse
import json
import time
import traceback
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import statistics
import spacy
import copy
import csv

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
)
from transformers import BertModel, BertTokenizer

from classifier import SECClassifier
from dataset import sec_bert_shape_preprocess
from get_num_labels import get_num_labels

# GCS imports
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    # Overall metrics
    accuracy: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    
    # Top-k accuracy
    top_3_accuracy: Optional[float] = None
    top_5_accuracy: Optional[float] = None
    
    # Per-class metrics
    per_class_metrics: Optional[Dict[int, Dict[str, float]]] = None
    
    # Confidence metrics
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    confidence_std: float = 0.0
    
    # Performance metrics
    inference_time_per_sample: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    # Data statistics
    num_samples: int = 0
    num_labels: int = 0
    label_distribution: Optional[Dict[int, int]] = None


def get_latest_model_from_gcs(
    bucket_name: str = "blacksmith-sec-filings",
    metadata_path: str = "fine_tuning/models_metadata.json",
    gcp_project: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get the latest model metadata from GCS models_metadata.json.
    
    Prefers models with "latest:stable" label, otherwise returns most recent by date_created.
    
    Args:
        bucket_name: GCS bucket name
        metadata_path: Path to models_metadata.json file in bucket
        gcp_project: Optional GCP project ID
        
    Returns:
        Model metadata dictionary with 'path', 'date_created', 'label', or None if not found
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
    
    # Read metadata file
    blob = bucket.blob(metadata_path)
    
    if not blob.exists():
        raise FileNotFoundError(f"Metadata file not found: gs://{bucket_name}/{metadata_path}")
    
    try:
        content = blob.download_as_text()
        metadata = json.loads(content)
        if not isinstance(metadata, list):
            raise ValueError(f"Invalid metadata format: expected list, got {type(metadata)}")
    except (json.JSONDecodeError, Exception) as e:
        raise RuntimeError(f"Failed to read metadata file: {e}") from e
    
    if not metadata:
        return None
    
    # First, try to find model with "latest:stable" label
    for model in metadata:
        if model.get("label") == "latest:stable":
            return model
    
    # If no "latest:stable" label, find most recent by date_created
    latest_model = None
    latest_date = None
    
    for model in metadata:
        date_str = model.get("date_created")
        if date_str:
            try:
                from datetime import datetime
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                if latest_date is None or date_obj > latest_date:
                    latest_date = date_obj
                    latest_model = model
            except (ValueError, TypeError):
                continue
    
    return latest_model


def download_model_from_gcs(
    gcs_model_path: str,
    local_model_dir: str,
    bucket_name: str = "blacksmith-sec-filings",
    gcp_project: Optional[str] = None
) -> str:
    """
    Download model files from GCS to local directory.
    
    Args:
        gcs_model_path: GCS path to model (e.g., "gs://bucket/path/to/model/" or "path/to/model/")
        local_model_dir: Local directory to download model files to
        bucket_name: GCS bucket name (if not in gcs_model_path)
        gcp_project: Optional GCP project ID
        
    Returns:
        Path to local model directory
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install with: pip install google-cloud-storage")
    
    # Parse GCS path
    if gcs_model_path.startswith("gs://"):
        # Extract bucket and path from gs:// URL
        parts = gcs_model_path[5:].split("/", 1)
        if len(parts) == 2:
            bucket_name = parts[0]
            model_path_prefix = parts[1]
        else:
            bucket_name = parts[0]
            model_path_prefix = ""
    else:
        model_path_prefix = gcs_model_path
    
    # Remove trailing slash if present
    if model_path_prefix.endswith("/"):
        model_path_prefix = model_path_prefix[:-1]
    
    # Initialize GCS client
    if gcp_project:
        client = storage.Client(project=gcp_project)
    else:
        client = storage.Client()
    
    bucket = client.bucket(bucket_name)
    
    if not bucket.exists():
        raise ValueError(f"Bucket '{bucket_name}' does not exist or is not accessible")
    
    # Create local directory
    os.makedirs(local_model_dir, exist_ok=True)
    
    # List all blobs with the model prefix
    print(f"📥 Downloading model from gs://{bucket_name}/{model_path_prefix}/")
    print(f"   Local directory: {local_model_dir}")
    
    blobs = bucket.list_blobs(prefix=model_path_prefix + "/")
    files_downloaded = 0
    
    for blob in blobs:
        # Skip directories (they end with /)
        if blob.name.endswith("/"):
            continue
        
        # Calculate relative path from model root
        relative_path = blob.name[len(model_path_prefix) + 1:] if model_path_prefix else blob.name
        
        # Skip checkpoint directories (optional - can be enabled later)
        if 'checkpoint-' in relative_path:
            continue
        
        # Create local file path
        local_file_path = os.path.join(local_model_dir, relative_path)
        local_file_dir = os.path.dirname(local_file_path)
        
        # Create directory if needed
        if local_file_dir:
            os.makedirs(local_file_dir, exist_ok=True)
        
        # Download file
        blob.download_to_filename(local_file_path)
        files_downloaded += 1
        
        if files_downloaded % 10 == 0:
            print(f"   Downloaded {files_downloaded} files...", end='\r')
    
    print(f"   Downloaded {files_downloaded} files")
    print(f"✅ Model downloaded to: {local_model_dir}")
    
    return local_model_dir


def download_spacy_model_from_gcs(
    model_name: str,
    bucket_name: str = "blacksmith-sec-filings",
    gcs_prefix: str = "spacy-models",
    gcp_project: Optional[str] = None
) -> str:
    """
    Download spaCy model from GCS bucket to persistent cache directory.
    Returns path to downloaded model directory.
    
    Args:
        model_name: Model name (e.g., "en_core_web_sm", "en_core_web_lg")
        bucket_name: GCS bucket name
        gcs_prefix: Prefix path in bucket for spaCy models
        gcp_project: Optional GCP project ID
        
    Returns:
        Path to model directory
    """
    import shutil
    import spacy
    from pathlib import Path
    
    print(f"📥 Checking for spaCy model: {model_name}...", flush=True)
    
    # Check if model already exists locally using spaCy's find function
    try:
        model_path = spacy.util.find(model_name)
        if model_path and Path(model_path).exists():
            print(f"✅ {model_name} already exists locally: {model_path}", flush=True)
            return model_path
    except Exception:
        pass
    
    # Use persistent cache directory (user's home directory)
    cache_base = Path.home() / ".cache" / "spacy_models_gcs"
    cache_base.mkdir(parents=True, exist_ok=True)
    model_cache_path = cache_base / model_name
    
    # Check if model already exists in cache
    if model_cache_path.exists() and (model_cache_path / "meta.json").exists():
        print(f"✅ {model_name} found in cache: {model_cache_path}", flush=True)
        return str(model_cache_path)
    
    # Need to download from GCS
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install with: pip install google-cloud-storage")
    
    print(f"📥 Downloading {model_name} from GCS...", flush=True)
    
    # Create cache directory for this model
    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize GCS client
        if gcp_project:
            storage_client = storage.Client(project=gcp_project)
        else:
            storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        
        if not bucket.exists():
            raise ValueError(f"Bucket '{bucket_name}' does not exist or is not accessible")
        
        # GCS path to model
        gcs_model_prefix = f"{gcs_prefix}/{model_name}/"
        
        # List all blobs with the model prefix
        blobs = list(bucket.list_blobs(prefix=gcs_model_prefix))
        
        if not blobs:
            raise FileNotFoundError(
                f"No files found in gs://{bucket_name}/{gcs_model_prefix}. "
                f"Model {model_name} may not exist in GCS."
            )
        
        # Download all files
        files_downloaded = 0
        for blob in blobs:
            # Skip directories (they end with /)
            if blob.name.endswith("/"):
                continue
            
            # Calculate relative path from model root
            relative_path = blob.name[len(gcs_model_prefix):]
            
            # Create local file path
            local_file_path = model_cache_path / relative_path
            local_file_dir = local_file_path.parent
            
            # Create directory if needed
            local_file_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file
            blob.download_to_filename(str(local_file_path))
            files_downloaded += 1
            
            if files_downloaded % 50 == 0:
                print(f"   Downloaded {files_downloaded} files...", end='\r', flush=True)
        
        print(f"   Downloaded {files_downloaded} files", flush=True)
        
        # Verify model was downloaded correctly (should have meta.json)
        if not (model_cache_path / "meta.json").exists():
            raise RuntimeError(f"Model download incomplete: meta.json not found in {model_cache_path}")
        
        print(f"✅ {model_name} downloaded to: {model_cache_path}", flush=True)
        return str(model_cache_path)
        
    except Exception as e:
        # Clean up on error
        if model_cache_path.exists():
            shutil.rmtree(model_cache_path)
        raise RuntimeError(
            f"Failed to download {model_name} from GCS bucket {bucket_name}: {str(e)}"
        ) from e


def download_test_csv_from_gcs(
    csv_filename: str,
    bucket_name: str = "blacksmith-sec-filings",
    gcs_prefix: str = "fine_tuning",
    gcp_project: Optional[str] = None,
    local_path: Optional[str] = None
) -> str:
    """
    Download test CSV file from GCS if not found locally.
    
    Args:
        csv_filename: Name of CSV file (e.g., "dev_labeled.csv")
        bucket_name: GCS bucket name
        gcs_prefix: Prefix path in bucket
        gcp_project: Optional GCP project ID
        local_path: Local path to save file (default: same as csv_filename)
        
    Returns:
        Local path to downloaded file
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install with: pip install google-cloud-storage")
    
    # Determine local path
    if local_path is None:
        local_path = csv_filename
    
    # Check if file already exists locally
    if os.path.exists(local_path):
        print(f"✅ Test CSV already exists locally: {local_path}")
        return local_path
    
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
    
    print(f"📥 Downloading test CSV from gs://{bucket_name}/{blob_name} to {local_path}")
    blob.download_to_filename(local_path)
    print(f"✅ Test CSV downloaded to: {local_path}")
    
    return local_path


def load_test_data(test_csv: str) -> Tuple[List[str], List[int]]:
    """
    Load test data from CSV.
    
    Args:
        test_csv: Path to test CSV file with 'text' and 'label' columns
        
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(test_csv)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"CSV must contain 'text' and 'label' columns. Found: {df.columns.tolist()}")
    
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    
    return texts, labels


def load_label_map(label_map_path: str) -> Dict[str, int]:
    """
    Load a label mapping from string labels to integer IDs.
    """
    with open(label_map_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Label map must be a JSON object of {label_str: int_id}")
    mapped = {}
    for k, v in data.items():
        if not isinstance(v, int):
            raise ValueError(f"Label id for '{k}' must be int, got {type(v)}")
        mapped[str(k)] = v
    if not mapped:
        raise ValueError("Label map is empty")
    return mapped


class SpaCyTextCatClassifier:
    """
    Wrapper to use a spaCy textcat/textcat_multilabel pipeline with the benchmark.
    """
    def __init__(
        self,
        model_name_or_path: str,
        label_map: Dict[str, int],
        device: str = "cpu",
        batch_size: int = 64
    ):
        if not label_map:
            raise ValueError("label_map is required for spaCy textcat benchmarking")
        self.label_map = label_map
        self.batch_size = batch_size
        
        if device.lower() == "cuda":
            # Fail fast if GPU requested but not available
            if not spacy.prefer_gpu():
                raise RuntimeError("CUDA requested for spaCy, but no GPU is available")
        self.nlp = spacy.load(model_name_or_path)
        if "textcat" in self.nlp.pipe_names:
            self.cat_component = "textcat"
        elif "textcat_multilabel" in self.nlp.pipe_names:
            self.cat_component = "textcat_multilabel"
        else:
            raise ValueError("spaCy model has no textcat/textcat_multilabel component")
    
    def _map_cats(self, cats: Dict[str, float]) -> Dict[int, float]:
        if not cats:
            raise ValueError("spaCy model returned empty categories for a sample")
        mapped = {}
        for label, score in cats.items():
            if label not in self.label_map:
                raise ValueError(f"Label '{label}' not found in provided label_map")
            mapped[self.label_map[label]] = float(score)
        return mapped
    
    def classify_batch(
        self,
        texts: List[str],
        return_probabilities: bool = True,
        batch_size: int = 64
    ) -> List[Tuple[int, Dict[int, float]]]:
        results = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            prob_dict = self._map_cats(doc.cats)
            predicted_label = max(prob_dict, key=prob_dict.get)
            if return_probabilities:
                results.append((predicted_label, prob_dict))
            else:
                results.append(predicted_label)
        return results
    
    def classify(
        self,
        text: str,
        return_probabilities: bool = False,
        top_k: Optional[int] = None
    ):
        # Not used in the benchmark pipeline; included for interface parity.
        result = self.classify_batch([text], return_probabilities=True, batch_size=1)[0]
        if return_probabilities:
            pred_label, prob_dict = result
            if top_k is not None:
                sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
                prob_dict = dict(sorted_probs)
            return pred_label, prob_dict
        return result[0]


def evaluate_model(
    classifier: SECClassifier,
    texts: List[str],
    labels: List[int],
    batch_size: int = 64,
    compute_top_k: bool = True,
    top_k_values: List[int] = [3, 5],
    max_samples: Optional[int] = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[float], Dict[int, Dict[int, float]]]:
    """
    Evaluate model on test data.
    
    Args:
        classifier: Loaded SECClassifier instance
        texts: List of test texts
        labels: List of true labels
        batch_size: Batch size for inference
        compute_top_k: Whether to compute top-k accuracies (requires storing all probabilities)
        top_k_values: List of k values for top-k accuracy
        max_samples: Optional limit on number of samples to evaluate (for faster testing)
        show_progress: Whether to show progress during evaluation
        
    Returns:
        Tuple of (predictions, true_labels, confidences, all_probabilities_dict)
        where all_probabilities_dict maps sample_idx -> {label: prob} (empty if compute_top_k=False)
    """
    # Limit samples if requested (for faster iteration)
    if max_samples and max_samples < len(texts):
        texts = texts[:max_samples]
        labels = labels[:max_samples]
        print(f"Limiting evaluation to {max_samples} samples for faster benchmarking...")
    
    predictions = []
    confidences = []
    all_probabilities = {} if compute_top_k else None
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"Evaluating on {len(texts)} samples in {num_batches} batches...")
    if show_progress:
        print(f"  Progress updates every {max(1, min(10, num_batches // 20))} batches...", flush=True)
    
    # Process in batches
    batch_num = 0
    for batch_idx in range(0, len(texts), batch_size):
        batch_num += 1
        
        # Show progress every 10 batches, or every 1% of batches, whichever is more frequent
        progress_interval = max(1, min(10, num_batches // 100))
        if show_progress and (batch_num % progress_interval == 0 or batch_num == 1):
            progress = (batch_num / num_batches) * 100
            samples_processed = min(batch_idx + batch_size, len(texts))
            print(f"  Progress: {progress:.1f}% (batch {batch_num}/{num_batches}, {samples_processed}/{len(texts)} samples)", flush=True)
        
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_labels = labels[batch_idx:batch_idx + batch_size]
        
        # Always get probabilities (needed for confidence), but only store full distributions if computing top-k
        batch_results = classifier.classify_batch(
            batch_texts, 
            return_probabilities=True,  # Always needed for confidence scores
            batch_size=batch_size
        )
        
        for j, result in enumerate(batch_results):
            pred_label, prob_dict = result
            predictions.append(pred_label)
            confidences.append(prob_dict[pred_label])
            
            # Only store full probability distributions if computing top-k (saves memory)
            if compute_top_k:
                sample_idx = batch_idx + j
                all_probabilities[sample_idx] = prob_dict
    
    if show_progress:
        print(f"  ✅ Completed: {batch_num}/{num_batches} batches, {len(texts)}/{len(texts)} samples processed", flush=True)
    
    predictions = np.array(predictions)
    true_labels = np.array(labels)
    confidences = np.array(confidences)
    
    return predictions, true_labels, confidences, all_probabilities or {}


def compute_detailed_metrics(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    confidences: np.ndarray,
    all_probabilities: Dict[int, Dict[int, float]],
    top_k_values: List[int] = [3, 5],
    compute_per_class: bool = True
) -> BenchmarkMetrics:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Array of predicted labels
        true_labels: Array of true labels
        confidences: Array of confidence scores
        all_probabilities: Dictionary mapping sample_idx -> {label: prob}
        top_k_values: List of k values for top-k accuracy
        
    Returns:
        BenchmarkMetrics object with all computed metrics
    """
    # Overall accuracy
    overall_accuracy = accuracy_score(true_labels, predictions)
    
    # Weighted, macro, and micro averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='micro', zero_division=0
    )
    
    # Top-k accuracy (only compute if probabilities were collected)
    top_k_accuracies = {}
    if all_probabilities and len(all_probabilities) > 0:
        try:
            # Optimized: Build probability matrix more efficiently
            num_samples = len(true_labels)
            # Get max label index efficiently
            max_label = max(
                max(prob_dict.keys()) if prob_dict else 0 
                for prob_dict in all_probabilities.values()
            )
            num_labels = max_label + 1
            
            # Pre-allocate matrix and fill more efficiently
            prob_matrix = np.zeros((num_samples, num_labels), dtype=np.float32)
            
            for sample_idx, prob_dict in all_probabilities.items():
                if sample_idx < num_samples and prob_dict:
                    for label, prob in prob_dict.items():
                        if label < num_labels:
                            prob_matrix[sample_idx, label] = prob
            
            # Compute top-k accuracy
            for k in top_k_values:
                try:
                    # Use sklearn's top_k_accuracy_score if available
                    top_k_acc = top_k_accuracy_score(true_labels, prob_matrix, k=k)
                    top_k_accuracies[f"top_{k}_accuracy"] = top_k_acc
                except Exception as e:
                    # Fallback: manual computation (more memory efficient)
                    try:
                        # Get top-k indices per sample
                        top_k_indices = np.argsort(prob_matrix, axis=1)[:, -k:]
                        # Check if true label is in top-k for each sample
                        correct_top_k = np.sum(
                            np.any(top_k_indices == true_labels.reshape(-1, 1), axis=1)
                        )
                        top_k_accuracies[f"top_{k}_accuracy"] = float(correct_top_k) / len(true_labels)
                    except Exception as e2:
                        print(f"Warning: Could not compute top-{k} accuracy: {e2}")
        except Exception as e:
            print(f"Warning: Error computing top-k accuracy: {e}")
    
    # Confidence statistics
    mean_confidence = float(np.mean(confidences))
    median_confidence = float(np.median(confidences))
    confidence_std = float(np.std(confidences))
    
    # Per-class metrics (for classes that appear in test set)
    # Optimized: Only compute if needed (can be expensive for many classes)
    per_class_metrics = {}
    if compute_per_class:
        unique_labels = np.unique(true_labels)
        
        # Use vectorized operations where possible for better performance
        if len(unique_labels) <= 1000:  # Only compute per-class if reasonable number of classes
            for label in unique_labels:
                label_mask = true_labels == label
                label_count = np.sum(label_mask)
                if label_count == 0:
                    continue
                
                label_predictions = predictions[label_mask]
                label_true = true_labels[label_mask]
                
                # Optimized: Use vectorized operations
                binary_true = (label_true == label).astype(int)
                binary_pred = (label_predictions == label).astype(int)
                
                # Compute metrics more efficiently
                tp = np.sum(binary_true & binary_pred)
                fp = np.sum((1 - binary_true) & binary_pred)
                fn = np.sum(binary_true & (1 - binary_pred))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                label_accuracy = float(np.mean(binary_true == binary_pred))
                
                per_class_metrics[int(label)] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'support': int(label_count),
                    'accuracy': label_accuracy
                }
    else:
        per_class_metrics = None
    
    # Label distribution
    label_counts = Counter(int(label) for label in true_labels)
    label_distribution = dict(label_counts)
    
    metrics = BenchmarkMetrics(
        accuracy=float(overall_accuracy),
        weighted_precision=float(weighted_precision),
        weighted_recall=float(weighted_recall),
        weighted_f1=float(weighted_f1),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        micro_precision=float(micro_precision),
        micro_recall=float(micro_recall),
        micro_f1=float(micro_f1),
        top_3_accuracy=top_k_accuracies.get("top_3_accuracy"),
        top_5_accuracy=top_k_accuracies.get("top_5_accuracy"),
        per_class_metrics=per_class_metrics,
        mean_confidence=mean_confidence,
        median_confidence=median_confidence,
        confidence_std=confidence_std,
        num_samples=len(true_labels),
        num_labels=len(unique_labels),
        label_distribution=label_distribution
    )
    
    return metrics


def test_inference_performance(
    classifier: SECClassifier,
    texts: List[str],
    batch_size: int = 64,
    num_warmup: int = 10,
    num_test_samples: int = 100
) -> Tuple[float, float]:
    """
    Measure inference performance.
    
    Args:
        classifier: Loaded SECClassifier instance
        texts: List of test texts
        batch_size: Batch size for inference
        num_warmup: Number of warmup samples
        
    Returns:
        Tuple of (average_time_per_sample_seconds, throughput_samples_per_second)
    """
    # Warmup
    if len(texts) > num_warmup:
        warmup_texts = texts[:num_warmup]
        _ = classifier.classify_batch(warmup_texts, batch_size=batch_size)
    
    # Actual timing (limit to reasonable number for faster benchmarking)
    end_idx = min(num_warmup + num_test_samples, len(texts))
    test_texts = texts[num_warmup:end_idx]
    
    start_time = time.time()
    _ = classifier.classify_batch(test_texts, batch_size=batch_size)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    num_samples = len(test_texts)
    
    avg_time_per_sample = elapsed_time / num_samples
    throughput = num_samples / elapsed_time
    
    return avg_time_per_sample, throughput


def analyze_errors(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    texts: List[str],
    confidences: np.ndarray,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Analyze classification errors.
    
    Args:
        predictions: Array of predicted labels
        true_labels: Array of true labels
        texts: List of input texts
        confidences: Array of confidence scores
        top_n: Number of top errors to return
        
    Returns:
        Dictionary with error analysis
    """
    errors = []
    correct = []
    
    for i, (pred, true, text, conf) in enumerate(zip(predictions, true_labels, texts, confidences)):
        if pred != true:
            errors.append({
                'index': i,
                'true_label': int(true),
                'predicted_label': int(pred),
                'confidence': float(conf),
                'text_preview': text[:100] if len(text) > 100 else text
            })
        else:
            correct.append({
                'index': i,
                'label': int(true),
                'confidence': float(conf)
            })
    
    # Sort errors by confidence (low confidence errors are more interesting)
    errors.sort(key=lambda x: x['confidence'])
    
    # Error statistics
    error_confidences = [e['confidence'] for e in errors]
    correct_confidences = [c['confidence'] for c in correct]
    
    error_analysis = {
        'total_errors': len(errors),
        'total_correct': len(correct),
        'error_rate': len(errors) / len(predictions),
        'mean_error_confidence': float(np.mean(error_confidences)) if error_confidences else 0.0,
        'mean_correct_confidence': float(np.mean(correct_confidences)) if correct_confidences else 0.0,
        'top_errors': errors[:top_n],
        'most_confident_errors': sorted(errors, key=lambda x: x['confidence'], reverse=True)[:top_n]
    }
    
    return error_analysis


def test_robustness(
    classifier: SECClassifier,
    sample_texts: List[str],
    sample_labels: List[int]
) -> Dict[str, Any]:
    """
    Test model robustness on edge cases.
    
    Args:
        classifier: Loaded SECClassifier instance
        sample_texts: Sample texts from test set
        sample_labels: Corresponding labels
        
    Returns:
        Dictionary with robustness test results
    """
    robustness_results = {}
    
    # Test with very short texts
    short_texts = [t[:50] for t in sample_texts[:10] if len(t) > 50]
    if short_texts:
        try:
            short_preds = classifier.classify_batch(short_texts)
            robustness_results['short_texts'] = {
                'num_tested': len(short_texts),
                'predictions': [int(p) for p in short_preds]
            }
        except Exception as e:
            robustness_results['short_texts'] = {'error': str(e)}
    
    # Test with very long texts (will be truncated)
    long_texts = [t * 10 for t in sample_texts[:10]]  # Repeat text to make longer
    if long_texts:
        try:
            long_preds = classifier.classify_batch(long_texts)
            robustness_results['long_texts'] = {
                'num_tested': len(long_texts),
                'predictions': [int(p) for p in long_preds]
            }
        except Exception as e:
            robustness_results['long_texts'] = {'error': str(e)}
    
    # Test with special characters
    special_texts = [f"[SPECIAL] {t} [SPECIAL]" for t in sample_texts[:10]]
    try:
        special_preds = classifier.classify_batch(special_texts)
        robustness_results['special_characters'] = {
            'num_tested': len(special_texts),
            'predictions': [int(p) for p in special_preds]
        }
    except Exception as e:
        robustness_results['special_characters'] = {'error': str(e)}
    
    return robustness_results


def generate_confusion_matrix(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    output_path: Optional[str] = None,
    max_classes: int = 50
) -> np.ndarray:
    """
    Generate and optionally save confusion matrix.
    
    Args:
        predictions: Array of predicted labels
        true_labels: Array of true labels
        output_path: Optional path to save confusion matrix
        max_classes: Maximum number of classes to show (for very large label spaces)
        
    Returns:
        Confusion matrix as numpy array
    """
    unique_labels = np.unique(np.concatenate([true_labels, predictions]))
    
    # Limit classes for visualization if too many
    if len(unique_labels) > max_classes:
        # Get most frequent classes in true labels
        label_counts = Counter(int(l) for l in true_labels)
        top_classes = [label for label, _ in label_counts.most_common(max_classes)]
        mask = np.isin(true_labels, top_classes) | np.isin(predictions, top_classes)
        filtered_true = true_labels[mask]
        filtered_pred = predictions[mask]
        
        # Remap labels to 0-max_classes for visualization
        label_map = {label: i for i, label in enumerate(top_classes)}
        filtered_true_mapped = np.array([label_map.get(l, -1) for l in filtered_true])
        filtered_pred_mapped = np.array([label_map.get(l, -1) for l in filtered_pred])
        
        # Filter out unmapped labels
        valid_mask = (filtered_true_mapped >= 0) & (filtered_pred_mapped >= 0)
        cm = confusion_matrix(
            filtered_true_mapped[valid_mask],
            filtered_pred_mapped[valid_mask],
            labels=list(range(len(top_classes)))
        )
        
        print(f"Warning: Showing confusion matrix for top {max_classes} most frequent classes only")
        print(f"  (out of {len(unique_labels)} total classes)")
    else:
        cm = confusion_matrix(true_labels, predictions, labels=list(unique_labels))
    
    if output_path:
        try:
            np.save(output_path, cm)
            print(f"Confusion matrix saved to {output_path}")
        except Exception as e:
            print(f"Warning: Could not save confusion matrix: {e}")
    
    return cm


def save_benchmark_report(
    metrics: BenchmarkMetrics,
    error_analysis: Dict[str, Any],
    robustness_results: Dict[str, Any],
    performance_metrics: Tuple[float, float],
    output_dir: str,
    confusion_matrix_path: Optional[str] = None
):
    """
    Save comprehensive benchmark report.
    
    Args:
        metrics: BenchmarkMetrics object
        error_analysis: Error analysis dictionary
        robustness_results: Robustness test results
        performance_metrics: Tuple of (avg_time_per_sample, throughput)
        output_dir: Directory to save report
        confusion_matrix_path: Optional path to confusion matrix file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert metrics to dictionary
    report = {
        'overall_metrics': {
            'accuracy': metrics.accuracy,
            'weighted_precision': metrics.weighted_precision,
            'weighted_recall': metrics.weighted_recall,
            'weighted_f1': metrics.weighted_f1,
            'macro_precision': metrics.macro_precision,
            'macro_recall': metrics.macro_recall,
            'macro_f1': metrics.macro_f1,
            'micro_precision': metrics.micro_precision,
            'micro_recall': metrics.micro_recall,
            'micro_f1': metrics.micro_f1,
        },
        'top_k_accuracy': {
            'top_3_accuracy': metrics.top_3_accuracy,
            'top_5_accuracy': metrics.top_5_accuracy,
        },
        'confidence_statistics': {
            'mean': metrics.mean_confidence,
            'median': metrics.median_confidence,
            'std': metrics.confidence_std,
        },
        'performance_metrics': {
            'inference_time_per_sample_seconds': performance_metrics[0],
            'throughput_samples_per_second': performance_metrics[1],
        },
        'data_statistics': {
            'num_samples': metrics.num_samples,
            'num_unique_labels': metrics.num_labels,
            'total_unique_labels_in_model': len(metrics.label_distribution) if metrics.label_distribution else 0,
        },
        'error_analysis': {
            'total_errors': error_analysis['total_errors'],
            'error_rate': error_analysis['error_rate'],
            'mean_error_confidence': error_analysis['mean_error_confidence'],
            'mean_correct_confidence': error_analysis['mean_correct_confidence'],
        },
        'robustness_tests': robustness_results,
    }
    
    # Add per-class metrics summary (only include top/bottom performers)
    if metrics.per_class_metrics:
        sorted_classes = sorted(
            metrics.per_class_metrics.items(),
            key=lambda x: x[1]['f1'],
            reverse=True
        )
        report['per_class_metrics_summary'] = {
            'top_10_classes': {
                int(k): v for k, v in sorted_classes[:10]
            },
            'bottom_10_classes': {
                int(k): v for k, v in sorted_classes[-10:]
            },
        }
    
    # Save JSON report
    json_path = os.path.join(output_dir, 'benchmark_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save human-readable report
    txt_path = os.path.join(output_dir, 'benchmark_report.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:                {metrics.accuracy:.4f}\n")
        f.write(f"Weighted Precision:      {metrics.weighted_precision:.4f}\n")
        f.write(f"Weighted Recall:         {metrics.weighted_recall:.4f}\n")
        f.write(f"Weighted F1:             {metrics.weighted_f1:.4f}\n")
        f.write(f"Macro Precision:         {metrics.macro_precision:.4f}\n")
        f.write(f"Macro Recall:            {metrics.macro_recall:.4f}\n")
        f.write(f"Macro F1:                {metrics.macro_f1:.4f}\n")
        f.write(f"Micro Precision:         {metrics.micro_precision:.4f}\n")
        f.write(f"Micro Recall:            {metrics.micro_recall:.4f}\n")
        f.write(f"Micro F1:                {metrics.micro_f1:.4f}\n\n")
        
        if metrics.top_3_accuracy:
            f.write("TOP-K ACCURACY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Top-3 Accuracy:          {metrics.top_3_accuracy:.4f}\n")
            if metrics.top_5_accuracy:
                f.write(f"Top-5 Accuracy:          {metrics.top_5_accuracy:.4f}\n")
            f.write("\n")
        
        f.write("CONFIDENCE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Confidence:         {metrics.mean_confidence:.4f}\n")
        f.write(f"Median Confidence:       {metrics.median_confidence:.4f}\n")
        f.write(f"Confidence Std Dev:      {metrics.confidence_std:.4f}\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Inference Time/Sample:   {performance_metrics[0]:.4f} seconds\n")
        f.write(f"Throughput:              {performance_metrics[1]:.2f} samples/second\n\n")
        
        f.write("DATA STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of Samples:       {metrics.num_samples}\n")
        f.write(f"Number of Unique Labels: {metrics.num_labels}\n\n")
        
        f.write("ERROR ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Errors:            {error_analysis['total_errors']}\n")
        f.write(f"Error Rate:              {error_analysis['error_rate']:.4f}\n")
        f.write(f"Mean Error Confidence:   {error_analysis['mean_error_confidence']:.4f}\n")
        f.write(f"Mean Correct Confidence: {error_analysis['mean_correct_confidence']:.4f}\n\n")
        
        if confusion_matrix_path:
            f.write(f"Confusion Matrix:        Saved to {confusion_matrix_path}\n\n")
    
    print(f"\nBenchmark report saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")


def upload_benchmark_results_to_gcs(
    output_dir: str,
    gcs_model_path: str,
    bucket_name: str = "blacksmith-sec-filings",
    gcp_project: Optional[str] = None
) -> List[str]:
    """
    Upload benchmark results to the GCS model directory.
    
    Args:
        output_dir: Local directory containing benchmark results
        gcs_model_path: GCS path to model directory (e.g., "gs://bucket/path/to/model/" or "path/to/model/")
        bucket_name: GCS bucket name (if not in gcs_model_path)
        gcp_project: Optional GCP project ID
        
    Returns:
        List of GCS paths to uploaded files
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install with: pip install google-cloud-storage")
    
    # Parse GCS path
    if gcs_model_path.startswith("gs://"):
        # Extract bucket and path from gs:// URL
        parts = gcs_model_path[5:].split("/", 1)
        if len(parts) == 2:
            bucket_name = parts[0]
            model_path_prefix = parts[1]
        else:
            bucket_name = parts[0]
            model_path_prefix = ""
    else:
        model_path_prefix = gcs_model_path
    
    # Remove trailing slash if present
    if model_path_prefix.endswith("/"):
        model_path_prefix = model_path_prefix[:-1]
    
    # Initialize GCS client
    if gcp_project:
        client = storage.Client(project=gcp_project)
    else:
        client = storage.Client()
    
    bucket = client.bucket(bucket_name)
    
    if not bucket.exists():
        raise ValueError(f"Bucket '{bucket_name}' does not exist or is not accessible")
    
    # Files to upload
    files_to_upload = [
        ("benchmark_report.json", "application/json"),
        ("benchmark_report.txt", "text/plain"),
    ]
    
    # Add confusion matrix if it exists
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.npy")
    if os.path.exists(confusion_matrix_path):
        files_to_upload.append(("confusion_matrix.npy", "application/octet-stream"))
    
    uploaded_files = []
    
    print(f"\n📤 Uploading benchmark results to gs://{bucket_name}/{model_path_prefix}/benchmark_results/")
    
    for filename, content_type in files_to_upload:
        local_file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(local_file_path):
            print(f"   ⚠️  Skipping {filename} (file not found)")
            continue
        
        # Construct GCS path: model_path/benchmark_results/filename
        gcs_blob_name = f"{model_path_prefix}/benchmark_results/{filename}"
        
        try:
            blob = bucket.blob(gcs_blob_name)
            blob.upload_from_filename(local_file_path, content_type=content_type)
            gcs_path = f"gs://{bucket_name}/{gcs_blob_name}"
            uploaded_files.append(gcs_path)
            print(f"   ✅ Uploaded {filename}")
        except Exception as e:
            print(f"   ❌ Failed to upload {filename}: {e}")
    
    if uploaded_files:
        print(f"\n✅ Benchmark results uploaded to GCS:")
        for gcs_path in uploaded_files:
            print(f"   {gcs_path}")
    
    return uploaded_files


def run_benchmark_core(
    args,
    texts: List[str],
    labels: List[int],
    adapter: str = "sec_bert",
    label_map_path: Optional[str] = None,
    run_name: Optional[str] = None
):
    """
    Execute a single benchmark run with the provided adapter and arguments.
    """
    gcs_model_path = None
    
    # Determine model for SEC-BERT runs
    if adapter == "sec_bert":
        should_download = args.model_path is None or args.download_from_gcs
        if should_download:
            print("\n[0/7] Downloading latest model from GCS...")
            # Get latest model metadata
            latest_model = get_latest_model_from_gcs(
                bucket_name=args.gcs_bucket,
                metadata_path=args.gcs_metadata_path,
                gcp_project=args.gcp_project
            )
            
            if not latest_model:
                raise ValueError("No models found in metadata file")
            
            gcs_path = latest_model["path"]
            date_created = latest_model.get("date_created", "unknown")
            label = latest_model.get("label", "none")
            
            print(f"   Latest model found:")
            print(f"     Path: {gcs_path}")
            print(f"     Created: {date_created}")
            print(f"     Label: {label}")
            
            # Store GCS path for uploading results later
            gcs_model_path = gcs_path
            
            # Determine local download directory
            if args.local_model_dir:
                local_model_dir = args.local_model_dir
            else:
                local_model_dir = "./downloaded-model"
            
            # Download model
            downloaded_path = download_model_from_gcs(
                gcs_model_path=gcs_path,
                local_model_dir=local_model_dir,
                bucket_name=args.gcs_bucket,
                gcp_project=args.gcp_project
            )
            
            args.model_path = downloaded_path
        else:
            print(f"\n[0/7] Using local model path: {args.model_path}")
            if args.model_path and args.model_path.startswith("gs://"):
                gcs_model_path = args.model_path
    else:
        print("\n[0/7] Using spaCy textcat model")
    
    # Determine number of labels
    if adapter == "sec_bert":
        if args.num_labels is None:
            print("\n[1/7] Detecting number of labels from model...")
            num_labels = get_num_labels(args.model_path)
            print(f"      Detected {num_labels} labels")
        else:
            num_labels = args.num_labels
            print(f"\n[1/7] Using {num_labels} labels (from argument)")
    else:
        num_labels = len(set(labels))
        print(f"\n[1/7] Using {num_labels} labels from dataset for spaCy textcat")
    
    # Download spaCy model for preprocessing or as the classifier
    print("\n[3/7] Ensuring spaCy model is available...")
    spacy_model_to_use = args.spacy_model
    spacy_model_path = download_spacy_model_from_gcs(
        model_name=spacy_model_to_use,
        bucket_name=args.gcs_bucket,
        gcp_project=args.gcp_project
    )
    if adapter == "sec_bert":
        # For SEC-BERT we might use the installed shortcut name if it resolves,
        # otherwise use the downloaded path.
        import spacy as _sp
        try:
            standard_path = _sp.util.find(spacy_model_to_use)
            if spacy_model_path != standard_path:
                spacy_model_to_use = spacy_model_path
        except Exception:
            spacy_model_to_use = spacy_model_path
        print(f"      spaCy model ready: {spacy_model_to_use}")
    else:
        spacy_model_to_use = spacy_model_path
        print(f"      spaCy model ready: {spacy_model_to_use}")
    
    # Initialize classifier
    print("\n[4/7] Loading model...")
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.lower() == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        try:
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else "Unknown"
            print(f"      ✅ CUDA available: {cuda_device_count} GPU(s) detected")
            print(f"      GPU: {cuda_device_name}")
        except Exception as e:
            raise RuntimeError(f"CUDA detected but device query failed: {e}")
    
    if adapter == "sec_bert":
        classifier = SECClassifier(
            model_path=args.model_path,
            num_labels=num_labels,
            device=device,
            spacy_model=spacy_model_to_use
        )
        print(f"      Model loaded on {device}")
    elif adapter == "spacy_textcat":
        if not label_map_path:
            raise ValueError("label_map_path is required for spaCy textcat benchmarking")
        label_map = load_label_map(label_map_path)
        classifier = SpaCyTextCatClassifier(
            model_name_or_path=spacy_model_to_use,
            label_map=label_map,
            device=device,
            batch_size=args.batch_size
        )
        print(f"      spaCy textcat model loaded on {device}")
    else:
        raise ValueError(f"Unsupported adapter: {adapter}")
    
    # Evaluate model
    print("\n[5/7] Evaluating model...")
    predictions, true_labels, confidences, all_probabilities = evaluate_model(
        classifier, 
        texts, 
        labels, 
        batch_size=args.batch_size,
        compute_top_k=not args.skip_top_k,
        max_samples=args.max_samples,
        show_progress=True
    )
    
    # Compute metrics
    print("\n[6/7] Computing detailed metrics...")
    skip_per_class_computation = args.skip_per_class
    metrics = compute_detailed_metrics(
        predictions, true_labels, confidences, all_probabilities if not args.skip_top_k else {}
    )
    if skip_per_class_computation:
        metrics.per_class_metrics = None
    
    # Measure performance
    print("      Measuring inference performance...")
    perf_metrics = test_inference_performance(classifier, texts, batch_size=args.batch_size)
    metrics.inference_time_per_sample = perf_metrics[0]
    metrics.throughput_samples_per_sec = perf_metrics[1]
    
    # Error analysis
    print("\n[7/7] Analyzing errors...")
    error_analysis = analyze_errors(predictions, true_labels, texts, confidences)
    
    # Robustness tests
    robustness_results = {}
    if not args.skip_robustness:
        print("\n      Running robustness tests...")
        sample_size = min(50, len(texts))
        sample_indices = np.random.choice(len(texts), sample_size, replace=False)
        sample_texts = [texts[i] for i in sample_indices]
        sample_labels = [labels[i] for i in sample_indices]
        robustness_results = test_robustness(classifier, sample_texts, sample_labels)
    
    # Generate confusion matrix
    confusion_matrix_path = None
    if not args.skip_confusion_matrix:
        os.makedirs(args.output_dir, exist_ok=True)
        print("\n      Generating confusion matrix...")
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.npy')
        generate_confusion_matrix(predictions, true_labels, cm_path)
        confusion_matrix_path = cm_path
    
    # Save report
    print("\n      Saving benchmark report...")
    save_benchmark_report(
        metrics, error_analysis, robustness_results, perf_metrics,
        args.output_dir, confusion_matrix_path
    )
    
    # Upload benchmark results to GCS model directory if we have GCS path
    if gcs_model_path:
        print("\n      Uploading benchmark results to GCS...")
        upload_benchmark_results_to_gcs(
            output_dir=args.output_dir,
            gcs_model_path=gcs_model_path,
            bucket_name=args.gcs_bucket,
            gcp_project=args.gcp_project
        )
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY{f' - {run_name}' if run_name else ''}")
    print("=" * 80)
    print(f"Accuracy:       {metrics.accuracy:.4f}")
    print(f"Weighted F1:    {metrics.weighted_f1:.4f}")
    print(f"Macro F1:       {metrics.macro_f1:.4f}")
    if metrics.top_3_accuracy:
        print(f"Top-3 Accuracy: {metrics.top_3_accuracy:.4f}")
    print(f"Throughput:     {metrics.throughput_samples_per_sec:.2f} samples/sec")
    print(f"\nFull report saved to: {args.output_dir}")
    print("=" * 80)
    
    return {
        "name": run_name or adapter,
        "adapter": adapter,
        "metrics": metrics,
        "error_analysis": error_analysis,
        "performance": perf_metrics,
        "output_dir": args.output_dir,
        "gcs_model_path": gcs_model_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark and validation for fine-tuned SEC-BERT model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model directory (if not provided, will download latest from GCS)"
    )
    parser.add_argument(
        "--download_from_gcs",
        action="store_true",
        help="Download latest model from GCS before benchmarking"
    )
    parser.add_argument(
        "--gcs_bucket",
        type=str,
        default="blacksmith-sec-filings",
        help="GCS bucket name (default: blacksmith-sec-filings)"
    )
    parser.add_argument(
        "--gcs_metadata_path",
        type=str,
        default="fine_tuning/models_metadata.json",
        help="Path to models_metadata.json in GCS (default: fine_tuning/models_metadata.json)"
    )
    parser.add_argument(
        "--gcp_project",
        type=str,
        default=None,
        help="GCP project ID (optional, uses default credentials if not provided)"
    )
    parser.add_argument(
        "--local_model_dir",
        type=str,
        default=None,
        help="Local directory to download model to (default: ./downloaded-model)"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test CSV file with 'text' and 'label' columns (if not provided, downloads dev_labeled.csv from GCS)"
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=None,
        help="Number of classification labels (auto-detected if not provided)"
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to use (SEC-BERT preprocessing or spaCy textcat adapter)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results (default: ./benchmark_results)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cpu' or 'cuda'). Auto-detects if not provided"
    )
    parser.add_argument(
        "--skip_robustness",
        action="store_true",
        help="Skip robustness tests"
    )
    parser.add_argument(
        "--skip_confusion_matrix",
        action="store_true",
        help="Skip confusion matrix generation"
    )
    parser.add_argument(
        "--skip_top_k",
        action="store_true",
        help="Skip top-k accuracy computation (much faster, uses less memory)"
    )
    parser.add_argument(
        "--skip_per_class",
        action="store_true",
        help="Skip per-class metrics computation (faster for many classes)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for faster testing/iteration)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: skip top-k, per-class metrics, robustness tests, and confusion matrix"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of CPU workers for parallel processing (default: CPU count - 1)"
    )
    parser.add_argument(
        "--compare_configs",
        type=str,
        default=None,
        help="Path to JSON config file describing multiple benchmark runs"
    )
    
    args = parser.parse_args()
    
    # Determine number of workers
    if args.num_workers is None:
        try:
            cpu_count = multiprocessing.cpu_count()
            num_workers = max(1, cpu_count - 1)  # Use CPU-1 workers
        except Exception:
            num_workers = 4  # Fallback default
    else:
        num_workers = max(1, args.num_workers)
    
    # Store in args for easy access
    args.num_workers = num_workers
    
    print("=" * 80)
    print("SEC-BERT MODEL BENCHMARK & VALIDATION")
    print("=" * 80)
    
    # Display CPU and worker information
    try:
        total_cpus = multiprocessing.cpu_count()
        print(f"\nSystem Information:")
        print(f"  Total CPUs: {total_cpus}")
        print(f"  Workers configured: {args.num_workers} (CPU-1)")
    except Exception:
        pass
    
    # Determine number of labels
    # Determine test CSV path (download from GCS if needed)
    test_csv_path = args.test_csv
    if test_csv_path is None:
        csv_filename = "dev_labeled.csv"
        print(f"\n[2/7] No test CSV specified, downloading default from GCS: {csv_filename}")
        test_csv_path = download_test_csv_from_gcs(
            csv_filename=csv_filename,
            bucket_name=args.gcs_bucket,
            gcs_prefix="fine_tuning",
            gcp_project=args.gcp_project
        )
    elif not os.path.exists(test_csv_path):
        print(f"\n[2/7] Test CSV not found locally, attempting to download from GCS...")
        csv_filename = os.path.basename(test_csv_path)
        test_csv_path = download_test_csv_from_gcs(
            csv_filename=csv_filename,
            bucket_name=args.gcs_bucket,
            gcs_prefix="fine_tuning",
            gcp_project=args.gcp_project,
            local_path=test_csv_path
        )
    else:
        print(f"\n[2/7] Using local test CSV: {test_csv_path}")
    
    # Load test data once
    print("      Loading test data...")
    texts, labels = load_test_data(test_csv_path)
    print(f"      Loaded {len(texts)} samples")
    
    # Apply fast mode settings
    if args.fast:
        args.skip_top_k = True
        args.skip_per_class = True
        args.skip_robustness = True
        args.skip_confusion_matrix = True
        print("Fast mode enabled: skipping optional computations")
    
    # Multi-run comparison
    if args.compare_configs:
        with open(args.compare_configs, "r") as f:
            cfg_list = json.load(f)
        if not isinstance(cfg_list, list):
            raise ValueError("--compare_configs must point to a JSON array of run configs")
        
        summary_rows = []
        base_output_dir = args.output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        
        for run_cfg in cfg_list:
            if not isinstance(run_cfg, dict):
                raise ValueError("Each compare config entry must be a JSON object")
            run_name = run_cfg.get("name") or run_cfg.get("adapter") or "run"
            adapter = run_cfg.get("adapter", "sec_bert")
            label_map_path = run_cfg.get("label_map_path")
            
            run_args = copy.deepcopy(args)
            # Apply overrides
            for field in [
                "model_path", "download_from_gcs", "local_model_dir", "spacy_model",
                "batch_size", "device", "num_labels", "skip_top_k", "skip_per_class",
                "skip_robustness", "skip_confusion_matrix", "max_samples",
                "gcs_bucket", "gcs_metadata_path", "gcp_project"
            ]:
                if field in run_cfg:
                    setattr(run_args, field, run_cfg[field])
            
            run_args.output_dir = os.path.join(base_output_dir, run_name)
            os.makedirs(run_args.output_dir, exist_ok=True)
            
            result = run_benchmark_core(
                run_args,
                texts,
                labels,
                adapter=adapter,
                label_map_path=label_map_path,
                run_name=run_name
            )
            metrics = result["metrics"]
            error_analysis = result["error_analysis"]
            summary_rows.append({
                "name": run_name,
                "adapter": adapter,
                "accuracy": metrics.accuracy,
                "weighted_f1": metrics.weighted_f1,
                "macro_f1": metrics.macro_f1,
                "top_3_accuracy": metrics.top_3_accuracy,
                "throughput": metrics.throughput_samples_per_sec,
                "mean_confidence": metrics.mean_confidence,
                "error_rate": error_analysis["error_rate"],
                "output_dir": result["output_dir"]
            })
        
        # Save summary
        summary_json_path = os.path.join(base_output_dir, "compare_summary.json")
        with open(summary_json_path, "w") as f:
            json.dump(summary_rows, f, indent=2)
        
        summary_csv_path = os.path.join(base_output_dir, "compare_summary.csv")
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name", "adapter", "accuracy", "weighted_f1", "macro_f1",
                    "top_3_accuracy", "throughput", "mean_confidence",
                    "error_rate", "output_dir"
                ]
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        
        print("\nComparison summary saved to:")
        print(f"  JSON: {summary_json_path}")
        print(f"  CSV:  {summary_csv_path}")
    else:
        run_benchmark_core(
            args,
            texts,
            labels,
            adapter="sec_bert",
            label_map_path=None,
            run_name="sec_bert"
        )


if __name__ == "__main__":
    main()

