#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-flight check for MOT15 testing
Verifies all prerequisites before running the main test
"""

import os
import sys
from pathlib import Path

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"{Colors.YELLOW}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.END}\n")


def check_item(name, condition, success_msg, fail_msg, fix_msg=None):
    """Check a single prerequisite item"""
    if condition:
        print(f"{Colors.GREEN}[OK]{Colors.END} {name}: {success_msg}")
        return True
    else:
        print(f"{Colors.RED}[FAIL]{Colors.END} {name}: {fail_msg}")
        if fix_msg:
            print(f"  {Colors.BLUE}-->{Colors.END} {fix_msg}")
        return False


def main():
    print_header("MOT15 Testing - Pre-flight Check")
    
    all_checks_passed = True
    
    # Check 1: Python version
    print(f"{Colors.BOLD}1. Python Environment{Colors.END}")
    python_version = sys.version_info
    version_ok = python_version >= (3, 6)
    check_item(
        "Python Version",
        version_ok,
        f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
        f"Python {python_version.major}.{python_version.minor} (need >= 3.6)",
        "Please upgrade to Python 3.6 or higher"
    )
    all_checks_passed &= version_ok
    print()
    
    # Check 2: Required packages
    print(f"{Colors.BOLD}2. Python Packages{Colors.END}")
    required_packages = ['torch', 'cv2', 'numpy', 'loguru']
    
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            check_item(
                f"Package: {package}",
                True,
                "Installed",
                "",
                ""
            )
        except ImportError:
            check_item(
                f"Package: {package}",
                False,
                "",
                "Not installed",
                f"pip install {package if package != 'cv2' else 'opencv-python'}"
            )
            all_checks_passed = False
    print()
    
    # Check 3: MOT15 Dataset
    print(f"{Colors.BOLD}3. MOT15 Dataset{Colors.END}")
    mot15_path = Path("datasets/MOT15")
    mot15_exists = mot15_path.exists()
    check_item(
        "MOT15 Directory",
        mot15_exists,
        f"Found at {mot15_path}",
        "Not found",
        "Run: python setup_mot15.py"
    )
    all_checks_passed &= mot15_exists
    
    if mot15_exists:
        # Check train and test folders
        train_path = mot15_path / "train"
        test_path = mot15_path / "test"
        
        train_exists = train_path.exists()
        check_item(
            "  Train folder",
            train_exists,
            f"{len(list(train_path.iterdir())) if train_exists else 0} sequences",
            "Not found",
            ""
        )
        
        test_exists = test_path.exists()
        test_seq_count = len([d for d in test_path.iterdir() if d.is_dir()]) if test_exists else 0
        check_item(
            "  Test folder",
            test_exists and test_seq_count > 0,
            f"{test_seq_count} sequences (need 11)",
            "Not found or empty",
            "Re-run: python setup_mot15.py"
        )
        all_checks_passed &= (test_exists and test_seq_count == 11)
        
        # Check annotations (optional)
        ann_path = mot15_path / "annotations"
        ann_exists = ann_path.exists()
        check_item(
            "  Annotations",
            ann_exists,
            "COCO format annotations found",
            "Not found (optional for tracking)",
            "Optional: python tools/convert_mot15_to_coco.py"
        )
    print()
    
    # Check 4: Model Files
    print(f"{Colors.BOLD}4. Model Files{Colors.END}")
    
    # Check config
    config_path = Path("exps/example/mot/yolox_nano_mix_det.py")
    config_exists = config_path.exists()
    check_item(
        "Model Config",
        config_exists,
        str(config_path),
        "Not found",
        "Check ByteTrack installation"
    )
    all_checks_passed &= config_exists
    
    # Check checkpoint
    ckpt_path = Path("pretrained/bytetrack_nano_mot17.pth.tar")
    ckpt_exists = ckpt_path.exists()
    if ckpt_exists:
        ckpt_size = ckpt_path.stat().st_size / (1024 * 1024)
        check_item(
            "Model Checkpoint",
            True,
            f"{ckpt_path} ({ckpt_size:.1f} MB)",
            "",
            ""
        )
    else:
        check_item(
            "Model Checkpoint",
            False,
            "",
            "Not found",
            "Run: python download_model.py nano"
        )
        all_checks_passed = False
    print()
    
    # Check 5: Demo Script
    print(f"{Colors.BOLD}5. Required Scripts{Colors.END}")
    demo_script = Path("tools/demo_track.py")
    demo_exists = demo_script.exists()
    check_item(
        "Demo Script",
        demo_exists,
        str(demo_script),
        "Not found",
        "Check ByteTrack installation"
    )
    all_checks_passed &= demo_exists
    
    test_script = Path("test_mot15_nano.py")
    test_exists = test_script.exists()
    check_item(
        "Test Script",
        test_exists,
        str(test_script),
        "Not found",
        "Script should be in current directory"
    )
    all_checks_passed &= test_exists
    print()
    
    # Check 6: Disk Space
    print(f"{Colors.BOLD}6. System Resources{Colors.END}")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        space_ok = free_gb > 2.0
        check_item(
            "Disk Space",
            space_ok,
            f"{free_gb:.1f} GB free",
            f"{free_gb:.1f} GB free (need > 2 GB)",
            "Free up disk space for output files"
        )
        all_checks_passed &= space_ok
    except:
        print(f"{Colors.YELLOW}[WARN]{Colors.END} Disk Space: Could not check")
    
    # Check GPU (optional)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"{Colors.GREEN}[OK]{Colors.END} GPU Available: {gpu_name}")
            print(f"  {Colors.BLUE}-->{Colors.END} Use --device gpu for faster processing")
        else:
            print(f"{Colors.YELLOW}[INFO]{Colors.END} GPU: Not available (will use CPU)")
    except:
        print(f"{Colors.YELLOW}[INFO]{Colors.END} GPU: Could not check")
    print()
    
    # Final Summary
    print_header("Summary")
    
    if all_checks_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}[OK] All checks passed!{Colors.END}")
        print(f"\n{Colors.CYAN}You're ready to run:{Colors.END}")
        print(f"  {Colors.BOLD}python test_mot15_nano.py{Colors.END}")
        print(f"\nOr for Windows:")
        print(f"  {Colors.BOLD}run_mot15_test_nano.bat{Colors.END}")
        print()
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}[FAIL] Some checks failed{Colors.END}")
        print(f"\n{Colors.YELLOW}Please fix the issues above before running the test.{Colors.END}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
