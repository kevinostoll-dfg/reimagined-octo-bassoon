"""
Inference script for fine-tuned SEC-BERT model.
"""

import spacy
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from dataset import sec_bert_shape_preprocess
from classification import BertForClassification
import os
import argparse
from typing import Tuple, Optional, Dict, Any


BASE_MODEL_ID = "nlpaueb/sec-bert-base"


def load_model_and_tokenizer(model_path: str, num_labels: int, device: str = "cpu"):
    """
    Load the fine-tuned model and tokenizer.
    
    This matches the training setup using standard transformers (not Unsloth).
    
    Args:
        model_path: Path to the fine-tuned model directory
        num_labels: Number of classification labels
        device: Device to load model on ("cpu" or "cuda")
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    # Load base model using standard transformers (matches training)
    base_model = BertModel.from_pretrained(BASE_MODEL_ID)
    
    # Wrap model with classification head (must match training)
    model = BertForClassification(base_model, num_labels)
    
    # Load the fine-tuned weights
    # Try loading in order: safetensors (preferred), then pytorch_model.bin
    weights_loaded = False
    if os.path.exists(os.path.join(model_path, "model.safetensors")):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
            model.load_state_dict(state_dict, strict=False)
            weights_loaded = True
            print("Loaded weights from model.safetensors")
        except Exception as e:
            print(f"Failed to load safetensors: {e}")
    
    if not weights_loaded and os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        try:
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device)
            model.load_state_dict(state_dict, strict=False)
            weights_loaded = True
            print("Loaded weights from pytorch_model.bin")
        except Exception as e:
            print(f"Failed to load pytorch_model.bin: {e}")
    
    if not weights_loaded:
        raise RuntimeError("Could not load fine-tuned weights. Check model path and files.")
    
    # Load tokenizer (prefer saved tokenizer, fallback to base model)
    if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        tokenizer = BertTokenizer.from_pretrained(model_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Compile model for faster inference (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile') and (device == "cuda" or str(device).startswith("cuda")):
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile() for faster inference")
    except Exception as e:
        print(f"Warning: Could not compile model with torch.compile(): {e}")
    
    return model, tokenizer


def predict(
    text: str, 
    model, 
    tokenizer, 
    spacy_tokenizer, 
    device: str = "cpu",
    return_probabilities: bool = True
) -> Tuple[int, Optional[Dict[int, float]]]:
    """
    Predict label for a single text.
    
    Args:
        text: Input text to classify
        model: Fine-tuned model
        tokenizer: Tokenizer
        spacy_tokenizer: spaCy tokenizer
        device: Device to run inference on
        return_probabilities: Whether to return probability distribution
        
    Returns:
        Tuple of (predicted_label, probabilities_dict) where probabilities_dict
        maps label_id -> probability, or None if return_probabilities=False
    """
    # Preprocess and tokenize
    processed_text = sec_bert_shape_preprocess(text, spacy_tokenizer, tokenizer)
    encodings = tokenizer(
        processed_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        probabilities = F.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probabilities, dim=-1).item()
    
    if return_probabilities:
        prob_dict = {
            int(i): float(prob) 
            for i, prob in enumerate(probabilities[0].cpu().numpy())
        }
        return predicted_label, prob_dict
    else:
        return predicted_label, None


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned SEC-BERT model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./fine-tuned-model",
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to classify (if not provided, will prompt)"
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        required=True,
        help="Number of classification labels"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    
    args = parser.parse_args()
    
    # Load spaCy tokenizer (disable parser and NER for speed - only need tokenization)
    print("Loading spaCy tokenizer...")
    spacy_tokenizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.num_labels, args.device
    )
    
    # Get text to classify
    if args.text:
        text = args.text
    else:
        text = input("Enter text to classify: ")
    
    # Predict
    print("\nProcessing text...")
    predicted_label, probabilities = predict(
        text, model, tokenizer, spacy_tokenizer, args.device
    )
    
    print(f"\nPredicted Label: {predicted_label}")
    if probabilities:
        print(f"Confidence: {probabilities[predicted_label]:.4f}")
        # Show top 5 predictions
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 predictions:")
        for label_id, prob in sorted_probs:
            print(f"  Label {label_id}: {prob:.4f}")


if __name__ == "__main__":
    main()

