"""
Utility script to determine the number of labels from a trained model.
"""

import os
import sys

def get_num_labels(model_path: str) -> int:
    """
    Get the number of labels from a trained model.
    
    Args:
        model_path: Path to the fine-tuned model directory
        
    Returns:
        Number of classification labels
    """
    try:
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        if "classifier.weight" in state_dict:
            num_labels = state_dict["classifier.weight"].shape[0]
            return num_labels
    except Exception as e:
        print(f"Failed to load from safetensors: {e}")
    
    try:
        import torch
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"), 
            map_location="cpu"
        )
        if "classifier.weight" in state_dict:
            num_labels = state_dict["classifier.weight"].shape[0]
            return num_labels
    except Exception as e:
        print(f"Failed to load from pytorch_model.bin: {e}")
    
    raise RuntimeError("Could not determine number of labels from model files")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        model_path = "./fine-tuned-model"
    else:
        model_path = sys.argv[1]
    
    try:
        num_labels = get_num_labels(model_path)
        print(f"Number of labels: {num_labels}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

