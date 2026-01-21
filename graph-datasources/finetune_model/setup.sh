#!/bin/bash
# Setup script for fine-tuning environment

set -e

echo "Installing Python dependencies..."
pip install "unsloth[torch]" transformers accelerate datasets sentence-transformers spacy torch numpy scikit-learn peft

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Setup complete!"

