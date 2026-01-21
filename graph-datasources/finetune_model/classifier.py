"""
Reusable SEC-BERT classifier for integration into the codebase.

Usage:
    from classifier import SECClassifier
    
    # Initialize classifier (loads model once)
    classifier = SECClassifier(model_path="./fine-tuned-model", num_labels=2788)
    
    # Classify single text
    label, confidence = classifier.classify("Your text here")
    
    # Classify batch of texts
    results = classifier.classify_batch(["text1", "text2", "text3"])
"""

import os
import spacy
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from transformers import BertModel, BertTokenizer

from dataset import sec_bert_shape_preprocess
from classification import BertForClassification


BASE_MODEL_ID = "nlpaueb/sec-bert-base"


class SECClassifier:
    """
    Fine-tuned SEC-BERT classifier for text classification.
    
    This class provides a simple interface for using the fine-tuned model
    in production code. It handles model loading, preprocessing, and inference.
    """
    
    def __init__(
        self,
        model_path: str,
        num_labels: int,
        device: Optional[str] = None,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the fine-tuned model directory
            num_labels: Number of classification labels
            device: Device to run inference on ("cpu" or "cuda"). 
                   If None, auto-detects CUDA availability.
            spacy_model: spaCy model name for preprocessing
        """
        self.model_path = model_path
        self.num_labels = num_labels
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load spaCy tokenizer (disable parser and NER for speed - only need tokenization)
        print(f"Loading spaCy model: {spacy_model}...")
        self.spacy_tokenizer = spacy.load(spacy_model, disable=["parser", "ner"])
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        print(f"Model loaded on {self.device}")
    
    def _load_model(self) -> Tuple[torch.nn.Module, BertTokenizer]:
        """Load the fine-tuned model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        # Load base model using standard transformers (matches training)
        base_model = BertModel.from_pretrained(BASE_MODEL_ID)
        
        # Wrap model with classification head (must match training)
        model = BertForClassification(base_model, self.num_labels)
        
        # Load the fine-tuned weights
        weights_loaded = False
        if os.path.exists(os.path.join(self.model_path, "model.safetensors")):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))
                model.load_state_dict(state_dict, strict=False)
                weights_loaded = True
            except Exception as e:
                print(f"Failed to load safetensors: {e}")
        
        if not weights_loaded and os.path.exists(os.path.join(self.model_path, "pytorch_model.bin")):
            try:
                state_dict = torch.load(
                    os.path.join(self.model_path, "pytorch_model.bin"), 
                    map_location=self.device
                )
                model.load_state_dict(state_dict, strict=False)
                weights_loaded = True
            except Exception as e:
                print(f"Failed to load pytorch_model.bin: {e}")
        
        if not weights_loaded:
            raise RuntimeError(
                f"Could not load fine-tuned weights from {self.model_path}. "
                "Check that model.safetensors or pytorch_model.bin exists."
            )
        
        # Load tokenizer (prefer saved tokenizer, fallback to base model)
        if os.path.exists(os.path.join(self.model_path, "tokenizer_config.json")):
            tokenizer = BertTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_ID)
        
        # Move model to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile') and (self.device == "cuda" or str(self.device).startswith("cuda")):
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compiled with torch.compile() for faster inference")
        except Exception as e:
            print(f"Warning: Could not compile model with torch.compile(): {e}")
        
        return model, tokenizer
    
    def classify(
        self, 
        text: str, 
        return_probabilities: bool = False,
        top_k: Optional[int] = None
    ) -> Union[int, Tuple[int, float], Tuple[int, Dict[int, float]]]:
        """
        Classify a single text.
        
        Args:
            text: Input text to classify
            return_probabilities: If True, return probability distribution
            top_k: If specified, return only top-k probabilities
            
        Returns:
            - If return_probabilities=False: predicted_label (int)
            - If return_probabilities=True: (predicted_label, probabilities_dict)
            - If top_k is specified: probabilities_dict contains only top-k labels
        """
        # Preprocess and tokenize
        processed_text = sec_bert_shape_preprocess(text, self.spacy_tokenizer, self.tokenizer)
        encodings = self.tokenizer(
            processed_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            probabilities = F.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=-1).item()
            confidence = float(probabilities[0][predicted_label].cpu().item())
        
        if not return_probabilities:
            return predicted_label
        
        # Build probability dictionary
        prob_dict = {
            int(i): float(prob) 
            for i, prob in enumerate(probabilities[0].cpu().numpy())
        }
        
        # Filter to top-k if specified
        if top_k is not None:
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            prob_dict = dict(sorted_probs)
        
        return predicted_label, prob_dict
    
    def classify_batch(
        self, 
        texts: List[str],
        return_probabilities: bool = False,
        batch_size: int = 64
    ) -> List[Union[int, Tuple[int, Dict[int, float]]]]:
        """
        Classify a batch of texts.
        
        Args:
            texts: List of texts to classify
            return_probabilities: If True, return probability distributions
            batch_size: Batch size for processing
            
        Returns:
            List of classification results (same format as classify())
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess batch using batch spaCy processing for efficiency
            from dataset import sec_bert_preprocess_batch
            processed_texts = sec_bert_preprocess_batch(
                batch_texts, self.spacy_tokenizer, self.tokenizer
            )
            
            # Tokenize batch
            encodings = self.tokenizer(
                processed_texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                probabilities = F.softmax(logits, dim=-1)
                predicted_labels = torch.argmax(probabilities, dim=-1).cpu().numpy()
            
            # Format results
            for j, label in enumerate(predicted_labels):
                if return_probabilities:
                    prob_dict = {
                        int(k): float(v) 
                        for k, v in enumerate(probabilities[j].cpu().numpy())
                    }
                    results.append((int(label), prob_dict))
                else:
                    results.append(int(label))
        
        return results
    
    def get_confidence(self, text: str) -> float:
        """
        Get confidence score for a text's predicted label.
        
        Args:
            text: Input text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        _, prob_dict = self.classify(text, return_probabilities=True, top_k=1)
        return list(prob_dict.values())[0]

