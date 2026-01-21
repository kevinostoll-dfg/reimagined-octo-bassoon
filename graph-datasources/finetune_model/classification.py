"""
Classification model wrapper for BERT encoder.
"""

import os
import torch
from torch import nn
from transformers import AutoConfig, BertModel


class BertForClassification(nn.Module):
    """
    BERT model wrapper for sequence classification tasks.
    
    Args:
        base_model: Base BERT model (from Unsloth FastModel)
        num_labels: Number of classification labels
    """
    
    def __init__(self, base_model, num_labels: int):
        super().__init__()
        self.bert = base_model
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        # Expose config for Trainer/transformers utilities and persist num_labels
        self.config = base_model.config
        self.config.num_labels = num_labels
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        num_items_in_batch=None,  # Trainer may pass this, ignore it
        **kwargs,
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            num_items_in_batch: Number of items in batch (passed by Trainer, ignored)
            **kwargs: Additional arguments passed to base model
            
        Returns:
            Dictionary with 'loss' (if labels provided) and 'logits'
        """
        # Filter out Trainer-specific arguments before passing to BERT
        bert_kwargs = {k: v for k, v in kwargs.items() if k != 'num_items_in_batch'}
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **bert_kwargs,
        )
        # Use [CLS] token embedding for classification
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.classifier.out_features),
                labels.view(-1)
            )
        
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        """
        Minimal save_pretrained to mirror transformers API so Trainer hooks work.
        Saves config plus both safetensors (if available) and pytorch_model.bin.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Persist num_labels on config for reloads
        self.config.num_labels = self.classifier.out_features
        self.config.save_pretrained(save_directory)

        state_dict = self.state_dict()
        weights_path = os.path.join(save_directory, "pytorch_model.bin")

        if safe_serialization:
            try:
                from safetensors.torch import save_file
            except ImportError:
                # Fallback to PyTorch serialization only
                torch.save(state_dict, weights_path)
            else:
                save_file(state_dict, os.path.join(save_directory, "model.safetensors"))
                torch.save(state_dict, weights_path)
        else:
            torch.save(state_dict, weights_path)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        """
        Minimal from_pretrained to reload models saved with save_pretrained.
        Prefers safetensors if present.
        """
        config = AutoConfig.from_pretrained(save_directory)
        num_labels = getattr(config, "num_labels", None)
        if num_labels is None:
            raise ValueError("num_labels missing from config; cannot initialize classifier head.")

        base_model = BertModel.from_pretrained(save_directory, config=config)
        model = cls(base_model, num_labels)

        # Load weights (prefer safetensors)
        safetensors_path = os.path.join(save_directory, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        else:
            weights_path = os.path.join(save_directory, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")

        model.load_state_dict(state_dict, strict=True)
        return model
