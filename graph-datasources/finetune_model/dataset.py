"""
Dataset loading and preprocessing utilities.
"""

import re
import os
import hashlib
import spacy
from datasets import load_dataset
from typing import List


def sec_bert_preprocess_batch(texts: List[str], spacy_tokenizer, tokenizer) -> List[str]:
    """
    Preprocess a batch of texts using batch spaCy processing for efficiency.
    
    This function replaces numeric tokens with shape patterns (e.g., "123" -> "[XXX]")
    to match the SEC-BERT model's tokenization strategy.
    
    Args:
        texts: List of input texts to preprocess
        spacy_tokenizer: Loaded spaCy tokenizer (should have parser/ner disabled)
        tokenizer: HuggingFace tokenizer with special tokens
        
    Returns:
        List of preprocessed text strings
    """
    # Use spaCy's batch processing for efficiency
    processed_texts = []
    # Process in batches of 1000 for spaCy (optimal batch size)
    for i in range(0, len(texts), 1000):
        batch_texts = texts[i:i + 1000]
        # Use pipe() for batch processing - much faster than individual calls
        docs = list(spacy_tokenizer.pipe(batch_texts, batch_size=1000))
        
        for doc in docs:
            tokens = [t.text for t in doc]
            processed = []
            for tok in tokens:
                if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", tok):
                    shape = "[" + re.sub(r"\d", "X", tok) + "]"
                    if shape in tokenizer.additional_special_tokens:
                        processed.append(shape)
                    else:
                        processed.append("[NUM]")
                else:
                    processed.append(tok)
            processed_texts.append(" ".join(processed))
    
    return processed_texts


def sec_bert_shape_preprocess(text: str, spacy_tokenizer, tokenizer) -> str:
    """
    Preprocess text according to SEC-BERT tokenization rules.
    
    This function replaces numeric tokens with shape patterns (e.g., "123" -> "[XXX]")
    to match the SEC-BERT model's tokenization strategy.
    
    Args:
        text: Input text to preprocess
        spacy_tokenizer: Loaded spaCy tokenizer
        tokenizer: HuggingFace tokenizer with special tokens
        
    Returns:
        Preprocessed text string
    """
    tokens = [t.text for t in spacy_tokenizer(text)]
    processed = []
    for tok in tokens:
        if re.fullmatch(r"(\d+[\d,.]*)|([,.]\d+)", tok):
            shape = "[" + re.sub(r"\d", "X", tok) + "]"
            if shape in tokenizer.additional_special_tokens:
                processed.append(shape)
            else:
                processed.append("[NUM]")
        else:
            processed.append(tok)
    return " ".join(processed)


def tokenize_batch(batch, tokenizer, spacy_tokenizer, max_length=512):
    """
    Tokenize a batch of texts with SEC-BERT preprocessing using batch spaCy processing.
    
    Args:
        batch: Batch of examples with 'text' field (and optionally 'label')
        tokenizer: HuggingFace tokenizer
        spacy_tokenizer: spaCy tokenizer (should have parser/ner disabled)
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with 'input_ids', 'attention_mask', and optionally 'labels'
    """
    # Use batch preprocessing for efficiency
    texts = sec_bert_preprocess_batch(batch["text"], spacy_tokenizer, tokenizer)
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,  # Return lists, not tensors
    )
    
    # Add labels if present
    if "label" in batch:
        encodings["labels"] = batch["label"]
    
    return encodings


def load_and_prepare_dataset(
    train_csv: str,
    val_csv: str,
    tokenizer,
    spacy_tokenizer,
    max_length: int = 512,
    cache_dir: str = None,
    num_proc: int = 4,
):
    """
    Load and prepare dataset for training with caching support.
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        tokenizer: HuggingFace tokenizer
        spacy_tokenizer: spaCy tokenizer (should have parser/ner disabled)
        max_length: Maximum sequence length
        cache_dir: Directory to cache tokenized dataset (if None, no caching)
        num_proc: Number of processes for parallel tokenization
        
    Returns:
        Tokenized dataset ready for training
    """
    def _file_hash(path: str) -> str:
        """Return a stable hash of file contents to bust stale caches when data changes."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    
    train_sig = _file_hash(train_csv)
    val_sig = _file_hash(val_csv)

    # Determine cache path based on CSV paths
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(train_csv), ".dataset_cache")
    
    cache_path = os.path.join(
        cache_dir,
        f"tokenized_{os.path.basename(train_csv)}_{train_sig}_{os.path.basename(val_csv)}_{val_sig}"
    )
    
    # Try to load cached dataset
    if os.path.exists(cache_path):
        try:
            from datasets import load_from_disk
            print(f"Loading cached tokenized dataset from {cache_path}...")
            tokenized_dataset = load_from_disk(cache_path)
            tokenized_dataset.set_format(type="torch")
            print("Successfully loaded cached dataset!")
            return tokenized_dataset
        except Exception as e:
            print(f"Failed to load cached dataset: {e}. Will regenerate...")
    
    # Load raw dataset
    raw_dataset = load_dataset(
        "csv",
        data_files={
            "train": train_csv,
            "validation": val_csv
        }
    )
    
    # Tokenize dataset with parallel processing
    tokenized_dataset = raw_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, spacy_tokenizer, max_length),
        batched=True,
        num_proc=num_proc,  # Parallel processing
        remove_columns=[
            c for c in raw_dataset["train"].column_names
            if c not in ["input_ids", "attention_mask", "labels"]
        ]
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch")
    
    # Cache the tokenized dataset
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            tokenized_dataset.save_to_disk(cache_path)
            print(f"Cached tokenized dataset to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to cache dataset: {e}")
    
    return tokenized_dataset
