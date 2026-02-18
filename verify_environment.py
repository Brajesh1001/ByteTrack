#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ByteTrack Environment Verification Script
Tests that all core dependencies are properly installed
"""

import sys

def verify_environment():
    print("=" * 50)
    print("ByteTrack Environment Verification")
    print("=" * 50)
    
    try:
        # Python version
        python_version = sys.version.split()[0]
        print(f"[OK] Python: {python_version}")
        
        # PyTorch
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     CUDA version: {torch.version.cuda}")
        
        # OpenCV
        import cv2
        print(f"[OK] OpenCV: {cv2.__version__}")
        
        # NumPy
        import numpy as np
        print(f"[OK] NumPy: {np.__version__}")
        
        # YOLOX (ByteTrack)
        import yolox
        print(f"[OK] YOLOX: {yolox.__version__}")
        
        # Additional key packages
        import torchvision
        print(f"[OK] TorchVision: {torchvision.__version__}")
        
        import onnx
        print(f"[OK] ONNX: {onnx.__version__}")
        
        import onnxruntime
        print(f"[OK] ONNX Runtime: {onnxruntime.__version__}")
        
        print("\n" + "=" * 50)
        print("SUCCESS: ALL CORE PACKAGES WORKING!")
        print("=" * 50)
        print(f"\nVirtual Environment: {sys.prefix}")
        print("\nByteTrack is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
