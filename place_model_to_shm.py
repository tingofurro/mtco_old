#!/usr/bin/env python3
"""
Script to load a Hugging Face causal language model to CPU and save it to /dev/shm for fast access.
Usage: python place_model_to_shm.py <model_name>
"""

import argparse
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Load a Hugging Face causal language model to CPU and save to /dev/shm"
    )
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Name of the Hugging Face model to load (e.g., 'gpt2', 'microsoft/DialoGPT-medium')"
    )
    
    args = parser.parse_args()
    
    # Create destination path in /dev/shm
    shm_path = Path(f"/dev/shm/{args.model_name}")
    
    # Check if /dev/shm exists
    if not Path("/dev/shm").exists():
        print("Error: /dev/shm does not exist on this system")
        sys.exit(1)
    
    # Create directory if it doesn't exist
    shm_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model '{args.model_name}' to CPU...")
    
    try:
        # Load model to CPU
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Save model to /dev/shm
        print(f"Saving model and tokenizer to {shm_path}...")
        model.save_pretrained(shm_path)
        tokenizer.save_pretrained(shm_path)
        print(f"✅ Model successfully saved to {shm_path}")
        print(f"📊 Directory size: {get_directory_size(shm_path):.2f} MB")
        
    except Exception as e:
        print(f"❌ Error loading/saving model: {str(e)}")
        # Clean up partial files if error occurred
        if shm_path.exists():
            import shutil
            shutil.rmtree(shm_path, ignore_errors=True)
        sys.exit(1)


def get_directory_size(path):
    """Calculate the total size of a directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


if __name__ == "__main__":
    main()
