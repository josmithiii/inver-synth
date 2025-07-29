#!/usr/bin/env python3
"""
Device detection script for TensorFlow training.
Outputs the best available device type for use in Make variables.
"""
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from models.common.device_utils import detect_best_device
    
    device_name, device_type, use_mixed_precision = detect_best_device()
    print(device_type)
except Exception as e:
    # Fallback to CPU if there are any import or detection errors
    print("cpu")
    sys.exit(0)