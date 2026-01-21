"""
Fine-tuning script for SEC-BERT encoder model.
Downloads labeled train/dev CSVs from GCS, then fine-tunes.
"""

import os
import sys
import argparse
import tempfile
from google.cloud import storage

# Prevent TensorFlow from being imported (we use PyTorch)
# This must be set before any transformers imports
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

# Note: We don't create a stub here because it interferes with PyTorch's import system.
# Instead, we rely on the environment variables to prevent TensorFlow from being used.
# If TensorFlow is imported and fails, transformers should handle it gracefully.

import spacy
import torch
import multiprocessing
from transformers import (
    BertModel,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from classification import BertForClassification
from dataset import load_and_prepare_dataset
from llm_classify import download_csv_from_gcs


# Configuration
BASE_MODEL_ID = "nlpaueb/sec-bert-base"
BERT_MODEL = BertModel
MAX_SEQ_LENGTH = 512

# Training hyperparameters
# 
# RECOMMENDED ADJUSTMENTS BY DATASET SIZE:
# 
# Small dataset (< 1K samples):
#   LEARNING_RATE = 5e-5  (higher learning rate for small datasets)
#   BATCH_SIZE = 8-16     (smaller batches to increase effective training steps)
#   NUM_EPOCHS = 5-10     (more epochs needed)
#   WARMUP_STEPS = 100    (10-20% of total steps)
#
# Medium dataset (1K-10K samples):
#   LEARNING_RATE = 2e-5  (standard BERT fine-tuning rate)
#   BATCH_SIZE = 16-32    (can use larger batches)
#   NUM_EPOCHS = 3-5      (standard fine-tuning)
#   WARMUP_STEPS = 500    (10-20% of total steps)
#
# Large dataset (> 10K samples):
#   LEARNING_RATE = 1e-5  (lower learning rate, more stable)
#   BATCH_SIZE = 32-64    (larger batches for efficiency)
#   NUM_EPOCHS = 2-3      (may converge faster)
#   WARMUP_STEPS = 1000+  (10-20% of total steps)
#
# GENERAL GUIDELINES:
# - Learning rate: Start with 2e-5, adjust based on loss curve
#   - If loss decreases too slowly: increase to 3e-5 or 5e-5
#   - If loss is unstable/noisy: decrease to 1e-5
# - Batch size: Limited by GPU memory (use gradient_accumulation_steps for larger effective batches)
# - Epochs: Watch validation loss - stop early if it starts increasing (overfitting)
# - Weight decay: 0.01 is good default (regularization)

LEARNING_RATE = 2e-5  # Standard BERT fine-tuning rate (good starting point)
BATCH_SIZE = 16       # Adjust based on GPU memory (8 for limited memory, 32+ if you have it)
NUM_EPOCHS = 3        # Start with 3, increase if model hasn't converged
WEIGHT_DECAY = 0.01   # L2 regularization (0.01 is good default, try 0.0-0.1 range)
WARMUP_STEPS = 500    # Linear warmup steps (10-20% of total training steps)
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients over N steps (effective batch = BATCH_SIZE * N)
DATALOADER_NUM_WORKERS = 4  # Number of worker processes for data loading (set to CPU count - 1 or 4-8)
OUTPUT_DIR = "./fine-tuned-model"
LOGGING_DIR = "./logs"

DEFAULT_BUCKET = "blacksmith-sec-filings"
DEFAULT_GCS_PREFIX = "fine_tuning"
DEFAULT_TRAIN_CSV = "train_labeled.csv"
DEFAULT_VAL_CSV = "dev_labeled.csv"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
# Optional override: set SPACY_MODEL_GCS_URI env or pass --spacy_model_gcs_uri
# Example: gs://blacksmith-sec-filings/spacy_models/en_core_web_sm-3.7.1-py3-none-any.whl
DEFAULT_SPACY_MODEL_GCS_URI = os.environ.get(
    "SPACY_MODEL_GCS_URI",
    "gs://blacksmith-sec-filings/spacy-models/en_core_web_sm",
)
def _download_gcs_prefix(gcs_uri: str, dest_dir: str):
    """
    Download all objects under a GCS prefix into dest_dir using google-cloud-storage.
    """
    if not gcs_uri.startswith("gs://"):
        raise RuntimeError(f"GCS URI must start with gs://, got: {gcs_uri}")
    path = gcs_uri[len("gs://") :]
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise RuntimeError(f"GCS URI must include bucket and prefix: {gcs_uri}")
    bucket_name, prefix = parts

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        raise RuntimeError(f"No objects found at {gcs_uri}")

    for blob in blobs:
        rel_path = blob.name[len(prefix) :].lstrip("/")
        target_path = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        blob.download_to_filename(target_path)


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Tuple of predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune SEC-BERT using labeled CSVs from GCS"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=DEFAULT_BUCKET,
        help=f"GCS bucket containing labeled CSVs (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--gcs_prefix",
        type=str,
        default=DEFAULT_GCS_PREFIX,
        help=f"GCS prefix path (default: {DEFAULT_GCS_PREFIX})",
    )
    parser.add_argument(
        "--gcp_project",
        type=str,
        default=None,
        help="Optional GCP project ID (uses default credentials if omitted)",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=DEFAULT_TRAIN_CSV,
        help=f"Train CSV filename in GCS (default: {DEFAULT_TRAIN_CSV})",
    )
    parser.add_argument(
        "--dev_csv",
        type=str,
        default=DEFAULT_VAL_CSV,
        help=f"Dev CSV filename in GCS (default: {DEFAULT_VAL_CSV})",
    )
    parser.add_argument(
        "--local_train_path",
        type=str,
        default=None,
        help="Optional local path to save train CSV (default: same as train_csv)",
    )
    parser.add_argument(
        "--local_dev_path",
        type=str,
        default=None,
        help="Optional local path to save dev CSV (default: same as dev_csv)",
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default=DEFAULT_SPACY_MODEL,
        help=f"spaCy model to load (default: {DEFAULT_SPACY_MODEL})",
    )
    parser.add_argument(
        "--spacy_model_gcs_uri",
        type=str,
        default=DEFAULT_SPACY_MODEL_GCS_URI,
        help=(
            "GCS URI to spaCy model wheel/tar.gz to install if the model is missing "
            f"(default env SPACY_MODEL_GCS_URI or {DEFAULT_SPACY_MODEL_GCS_URI})"
        ),
    )
    return parser.parse_args()


def load_spacy_model(model_name: str, gcs_uri: str = None):
    """
    Load a spaCy model, downloading from GCS if missing. We do not fall back
    to spaCy's public model registry; GCS is the single source of truth unless
    the model is already installed locally.
    """
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        if not gcs_uri:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed and no GCS URI was provided."
            )
        print(
            f"Model {model_name} missing; downloading from {gcs_uri} and installing...",
            flush=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = os.path.join(tmpdir, model_name)
            os.makedirs(target_dir, exist_ok=True)
            _download_gcs_prefix(gcs_uri, target_dir)
            # Locate the model directory (first subdir if present)
            candidates = [
                os.path.join(target_dir, d)
                for d in os.listdir(target_dir)
                if os.path.isdir(os.path.join(target_dir, d))
            ]
            model_path = candidates[0] if candidates else target_dir
            # Load directly from the downloaded path (no pip install needed)
            return spacy.load(model_path, disable=["parser", "ner"])


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("SEC-BERT Fine-tuning")
    print("=" * 60)

    # Download labeled CSVs from GCS
    print("\n[0/7] Downloading labeled datasets from GCS...")
    train_csv_path = download_csv_from_gcs(
        bucket_name=args.bucket,
        csv_filename=args.train_csv,
        gcp_project=args.gcp_project,
        gcs_prefix=args.gcs_prefix,
        local_path=args.local_train_path,
    )
    dev_csv_path = download_csv_from_gcs(
        bucket_name=args.bucket,
        csv_filename=args.dev_csv,
        gcp_project=args.gcp_project,
        gcs_prefix=args.gcs_prefix,
        local_path=args.local_dev_path,
    )
    
    # Load spaCy tokenizer (disable parser and NER for speed - only need tokenization)
    print("\n[1/7] Loading spaCy tokenizer...")
    spacy_tokenizer = load_spacy_model(
        model_name=args.spacy_model,
        gcs_uri=args.spacy_model_gcs_uri,
    )
    
    # Load model and tokenizer
    print(f"\n[2/7] Loading model: {BASE_MODEL_ID}")
    print(f"      Using standard transformers")
    
    # Use standard transformers loading
    config = AutoConfig.from_pretrained(BASE_MODEL_ID)
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_ID)
    model = BertModel.from_pretrained(BASE_MODEL_ID)
    
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load and prepare dataset
    print("\n[3/7] Loading dataset...")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
    if not os.path.exists(dev_csv_path):
        raise FileNotFoundError(f"Validation CSV not found: {dev_csv_path}")

    # Determine optimal number of processes for dataset tokenization
    num_proc = min(8, max(1, multiprocessing.cpu_count() - 1))
    
    tokenized_dataset = load_and_prepare_dataset(
        train_csv=train_csv_path,
        val_csv=dev_csv_path,
        tokenizer=tokenizer,
        spacy_tokenizer=spacy_tokenizer,
        max_length=MAX_SEQ_LENGTH,
        cache_dir=None,  # Can set to a path to enable caching
        num_proc=num_proc,
    )
    
    # Determine number of labels from dataset (labels are tensors after set_format)
    print("\n[4/7] Setting up model architecture...")
    raw_labels = tokenized_dataset["train"]["labels"]
    labels = [int(l) if hasattr(l, "item") else int(l) for l in raw_labels]
    num_labels = len(set(labels))
    print(f"      Detected {num_labels} labels in dataset")
    
    # Wrap model with classification head
    model = BertForClassification(model, num_labels)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.bert, 'gradient_checkpointing_enable'):
        model.bert.gradient_checkpointing_enable()
    
    # Setup training arguments
    print("\n[5/7] Setting up training arguments...")
    # Calculate warmup steps if not explicitly set (10% of total steps)
    total_steps = len(tokenized_dataset["train"]) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    effective_warmup_steps = WARMUP_STEPS if WARMUP_STEPS > 0 else max(1, int(total_steps * 0.1))
    
    # Determine optimal number of dataloader workers
    dataloader_workers = DATALOADER_NUM_WORKERS
    if dataloader_workers <= 0:
        dataloader_workers = min(8, max(1, multiprocessing.cpu_count() - 1))
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Can use larger batch for eval (no gradients)
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=effective_warmup_steps,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        lr_scheduler_type="cosine",  # Use cosine annealing with warmup
        logging_dir=LOGGING_DIR,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use FP16 if CUDA is available
        bf16=False,
        dataloader_pin_memory=True,  # Enable pin memory for faster GPU transfer
        dataloader_num_workers=dataloader_workers,  # Parallel data loading
        report_to=[],  # Disable all logging integrations (wandb/tensorboard)
    )
    
    # Create Trainer
    print("\n[6/7] Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    train_result = trainer.train()
    
    # Save model
    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)
    trainer.save_model()
    # Save both formats so downstream consumers (e.g., spaCy transformers) can load either
    # - safetensors: preferred secure format
    # - pytorch_model.bin: legacy/compatibility format
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)   # creates model.safetensors
    model.save_pretrained(OUTPUT_DIR, safe_serialization=False)  # creates pytorch_model.bin
    tokenizer.save_pretrained(OUTPUT_DIR)
    # Ensure config.json is saved (should be saved by trainer.save_model(), but explicitly save it)
    config.save_pretrained(OUTPUT_DIR)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Final evaluation...")
    print("=" * 60)
    eval_results = trainer.evaluate()
    print(f"\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

