#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ByteTrack Multi-Model Testing Script
Test different YOLOX model sizes and compare results
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Model configurations
MODEL_CONFIGS = {
    "nano": {
        "config": "exps/example/mot/yolox_nano_mix_det.py",
        "checkpoint": "pretrained/bytetrack_nano_mot17.pth.tar",
        "description": "Nano - Fastest, smallest model (0.90M params)",
        "params": "0.90M",
        "flops": "3.99G"
    },
    "tiny": {
        "config": "exps/example/mot/yolox_tiny_mix_det.py",
        "checkpoint": "pretrained/bytetrack_tiny_mot17.pth.tar",
        "description": "Tiny - Small model (5.03M params)",
        "params": "5.03M",
        "flops": "24.45G"
    },
    "s": {
        "config": "exps/example/mot/yolox_s_mix_det.py",
        "checkpoint": "pretrained/bytetrack_s_mot17.pth.tar",
        "description": "S - Small model, balanced (9M params)",
        "params": "9M",
        "flops": "26G"
    },
    "m": {
        "config": "exps/example/mot/yolox_m_mix_det.py",
        "checkpoint": "pretrained/bytetrack_m_mot17.pth.tar",
        "description": "M - Medium model (25M params)",
        "params": "25M",
        "flops": "73G"
    },
    "l": {
        "config": "exps/example/mot/yolox_l_mix_det.py",
        "checkpoint": "pretrained/bytetrack_l_mot17.pth.tar",
        "description": "L - Large model, high accuracy (54M params)",
        "params": "54M",
        "flops": "155G"
    },
    "x": {
        "config": "exps/example/mot/yolox_x_mix_det.py",
        "checkpoint": "pretrained/bytetrack_x_mot17.pth.tar",
        "description": "X - Extra large, best accuracy (99M params)",
        "params": "99M",
        "flops": "281G"
    }
}

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"{Colors.YELLOW}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}{text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}{text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}{text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}{text}{Colors.END}")

def check_file_exists(filepath):
    """Check if file exists"""
    return Path(filepath).exists()

def list_available_videos():
    """List available videos in the videos directory"""
    videos_dir = Path("videos")
    if not videos_dir.exists():
        return []
    return list(videos_dir.glob("*.mp4"))

def list_available_models():
    """List models that have checkpoints downloaded"""
    available = []
    for model_name, config in MODEL_CONFIGS.items():
        if check_file_exists(config["checkpoint"]):
            available.append(model_name)
    return available

def test_model(model_name, video_path, device="cpu", fp16=False, fuse=False, 
               track_thresh=0.5, track_buffer=30, match_thresh=0.8):
    """
    Test a specific model
    
    Args:
        model_name: Name of the model (nano, tiny, s, m, l, x)
        video_path: Path to video file
        device: Device to use (cpu or gpu)
        fp16: Use FP16 precision
        fuse: Fuse conv and bn layers
        track_thresh: Detection confidence threshold
        track_buffer: Number of frames to keep lost tracks
        match_thresh: Matching threshold for tracking
    
    Returns:
        dict: Test results including status, time, and error message
    """
    if model_name not in MODEL_CONFIGS:
        return {
            "model": model_name,
            "status": "ERROR",
            "error": f"Unknown model: {model_name}",
            "time": 0
        }
    
    model = MODEL_CONFIGS[model_name]
    
    print_header(f"Testing: {model['description']}")
    
    # Check if checkpoint exists
    if not check_file_exists(model["checkpoint"]):
        print_error(f"Model checkpoint not found: {model['checkpoint']}")
        print_info("Please download it from Google Drive (see TESTING_VARIOUS_MODELS.md)")
        return {
            "model": model_name,
            "status": "MISSING",
            "error": "Checkpoint not found",
            "time": 0
        }
    
    # Check if config exists
    if not check_file_exists(model["config"]):
        print_error(f"Model config not found: {model['config']}")
        return {
            "model": model_name,
            "status": "ERROR",
            "error": "Config not found",
            "time": 0
        }
    
    # Build command
    cmd = [
        sys.executable,
        "tools/demo_track.py",
        "video",
        "-f", model["config"],
        "-c", model["checkpoint"],
        "--path", video_path,
        "--device", device,
        "--track_thresh", str(track_thresh),
        "--track_buffer", str(track_buffer),
        "--match_thresh", str(match_thresh),
        "--save_result"
    ]
    
    if fp16:
        cmd.append("--fp16")
    if fuse:
        cmd.append("--fuse")
    
    print_info(f"Running: {' '.join(cmd)}")
    print()
    
    # Time the execution
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        
        print_success(f"\n[SUCCESS] Model '{model_name}' completed successfully!")
        print_success(f"  Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        return {
            "model": model_name,
            "status": "SUCCESS",
            "time": elapsed_time,
            "error": None
        }
    
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print_error(f"\n[FAILED] Model '{model_name}' failed with exit code: {e.returncode}")
        
        return {
            "model": model_name,
            "status": "FAILED",
            "time": elapsed_time,
            "error": f"Exit code: {e.returncode}"
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print_error(f"\n[ERROR] Model '{model_name}' encountered an error: {e}")
        
        return {
            "model": model_name,
            "status": "ERROR",
            "time": elapsed_time,
            "error": str(e)
        }

def print_results_summary(results):
    """Print summary of test results"""
    print_header("Test Results Summary")
    
    # Print table header
    print(f"{'Model':<10} {'Status':<12} {'Time':<20} {'Error':<30}")
    print("-" * 72)
    
    # Print results
    for result in results:
        model = result["model"]
        status = result["status"]
        
        if result["time"] > 0:
            time_str = f"{result['time']:.2f}s ({result['time']/60:.2f}m)"
        else:
            time_str = "N/A"
        
        error = result.get("error", "")
        if error is None:
            error = ""
        
        # Color code status
        if status == "SUCCESS":
            status_colored = f"{Colors.GREEN}{status}{Colors.END}"
        elif status == "MISSING":
            status_colored = f"{Colors.YELLOW}{status}{Colors.END}"
        else:
            status_colored = f"{Colors.RED}{status}{Colors.END}"
        
        print(f"{model:<10} {status_colored:<20} {time_str:<20} {error:<30}")
    
    print()
    
    # Count successes
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    total_count = len(results)
    
    print_success(f"Completed: {success_count}/{total_count} models tested successfully")
    print()

def find_output_videos():
    """Find generated output videos"""
    output_dir = Path("YOLOX_outputs")
    if not output_dir.exists():
        return []
    
    videos = list(output_dir.rglob("*.mp4"))
    videos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return videos[:10]  # Return 10 most recent

def main():
    parser = argparse.ArgumentParser(
        description="ByteTrack Multi-Model Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test both available models (nano and x)
  python test_models.py
  
  # Test specific models
  python test_models.py -m nano x
  
  # Test all downloaded models
  python test_models.py --all
  
  # Test with custom video
  python test_models.py -v videos/my_video.mp4
  
  # Test with GPU and optimizations
  python test_models.py -m x --device gpu --fp16 --fuse
  
  # Test with custom parameters
  python test_models.py -m nano --track-thresh 0.6 --match-thresh 0.7
        """
    )
    
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to test (default: tests available models)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all downloaded models"
    )
    
    parser.add_argument(
        "-v", "--video",
        default="videos/palace.mp4",
        help="Path to video file (default: videos/palace.mp4)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to use (default: cpu)"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision (requires GPU)"
    )
    
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Fuse conv and bn for faster inference"
    )
    
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="Number of frames to keep lost tracks (default: 30)"
    )
    
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=0.8,
        help="Matching threshold for tracking (default: 0.8)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--list-videos",
        action="store_true",
        help="List available videos and exit"
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print_header("Available Models")
        for model_name, config in MODEL_CONFIGS.items():
            # Use ASCII characters for Windows compatibility
            exists = "[Y]" if check_file_exists(config["checkpoint"]) else "[N]"
            print(f"{exists} {model_name:6s} - {config['description']}")
        print()
        print_info("Models with [Y] are downloaded and ready to use")
        print_info("Models with [N] need to be downloaded from Google Drive")
        return 0
    
    # List videos
    if args.list_videos:
        print_header("Available Videos")
        videos = list_available_videos()
        if videos:
            for video in videos:
                size_mb = video.stat().st_size / (1024 * 1024)
                print(f"  - {video} ({size_mb:.2f} MB)")
        else:
            print_warning("No videos found in videos/ directory")
        return 0
    
    # Main header
    print_header("ByteTrack Multi-Model Testing")
    
    # Verify video exists
    if not check_file_exists(args.video):
        print_error(f"Error: Video file not found: {args.video}")
        print_info("\nAvailable videos:")
        videos = list_available_videos()
        if videos:
            for video in videos:
                print(f"  - {video}")
        else:
            print_warning("  No videos found in videos/ directory")
        return 1
    
    # Determine which models to test
    if args.all:
        models_to_test = list_available_models()
        if not models_to_test:
            print_error("No models found. Please download models first.")
            print_info("See TESTING_VARIOUS_MODELS.md for download links")
            return 1
    elif args.models:
        models_to_test = args.models
    else:
        # Default: test available models
        models_to_test = list_available_models()
        if not models_to_test:
            print_error("No models found. Please download models first.")
            print_info("Available models: " + ", ".join(MODEL_CONFIGS.keys()))
            print_info("See TESTING_VARIOUS_MODELS.md for download links")
            return 1
    
    # Print configuration
    print_info(f"Video: {args.video}")
    print_info(f"Device: {args.device}")
    print_info(f"Models to test: {', '.join(models_to_test)}")
    print_info(f"FP16: {args.fp16}")
    print_info(f"Fuse: {args.fuse}")
    print_info(f"Track threshold: {args.track_thresh}")
    print_info(f"Track buffer: {args.track_buffer}")
    print_info(f"Match threshold: {args.match_thresh}")
    print()
    
    # Test each model
    results = []
    for model_name in models_to_test:
        result = test_model(
            model_name,
            args.video,
            device=args.device,
            fp16=args.fp16,
            fuse=args.fuse,
            track_thresh=args.track_thresh,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh
        )
        results.append(result)
        print()
    
    # Print summary
    print_results_summary(results)
    
    # Show output files
    print_info("Output videos saved to: YOLOX_outputs/")
    print_info("\nGenerated output files (10 most recent):")
    output_videos = find_output_videos()
    if output_videos:
        for video in output_videos:
            rel_path = video.relative_to(Path.cwd())
            size_mb = video.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(video.stat().st_mtime)
            print(f"  - {rel_path} ({size_mb:.2f} MB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        print_warning("  No output files found")
    
    print()
    print_info("To compare results, run: python compare_results.py")
    print()
    
    # Return exit code based on results
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
