#!/usr/bin/env python
"""
Main script to train CLIP with DreamSim triplet dataset.
This can be used directly with python or through torchrun for distributed training.
"""

from src.train import main

if __name__ == "__main__":
    # Entry point for training
    main() 