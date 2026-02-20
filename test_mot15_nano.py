#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ByteTrack MOT15 Test Set Evaluation with Nano Model
Automated batch processing of all MOT15 test sequences
"""

import os
import sys
import time
import cv2
import torch
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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


# MOT15 Test sequences information
MOT15_TEST_SEQUENCES = {
    "ADL-Rundle-1": {
        "frames": 500,
        "fps": 30,
        "resolution": "1920x1080",
        "description": "Busy pedestrian street filmed at eye level"
    },
    "ADL-Rundle-3": {
        "frames": 625,
        "fps": 30,
        "resolution": "1920x1080",
        "description": "Crowded pedestrian street, stationary camera"
    },
    "AVG-TownCentre": {
        "frames": 450,
        "fps": 2.5,
        "resolution": "1920x1080",
        "description": "Pedestrian street filmed from elevated point"
    },
    "ETH-Crossing": {
        "frames": 219,
        "fps": 14,
        "resolution": "640x480",
        "description": "Street scene from moving platform"
    },
    "ETH-Jelmoli": {
        "frames": 440,
        "fps": 14,
        "resolution": "640x480",
        "description": "Street scene from moving platform"
    },
    "ETH-Linthescher": {
        "frames": 1194,
        "fps": 14,
        "resolution": "640x480",
        "description": "Street scene from moving platform"
    },
    "KITTI-16": {
        "frames": 209,
        "fps": 10,
        "resolution": "1224x370",
        "description": "Pedestrians crossing street filmed from car"
    },
    "KITTI-19": {
        "frames": 1059,
        "fps": 10,
        "resolution": "1238x374",
        "description": "Street scene from moving vehicle"
    },
    "PETS09-S2L2": {
        "frames": 436,
        "fps": 7,
        "resolution": "768x576",
        "description": "Crowded scene from elevated viewpoint"
    },
    "TUD-Crossing": {
        "frames": 201,
        "fps": 25,
        "resolution": "640x480",
        "description": "Road crossing from side view"
    },
    "Venice-1": {
        "frames": 450,
        "fps": 30,
        "resolution": "1920x1080",
        "description": "People walking around large square"
    }
}


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.CYAN}{'=' * 80}{Colors.END}")
    print(f"{Colors.YELLOW}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'=' * 80}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}[ERROR] {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}[INFO] {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.END}")


def check_prerequisites(args):
    """Check if all required files and directories exist"""
    print_header("Checking Prerequisites")
    
    all_ok = True
    
    # Check MOT15 dataset
    mot15_path = Path(args.data_root) / "MOT15"
    if not mot15_path.exists():
        print_error(f"MOT15 dataset not found at: {mot15_path}")
        print_info("Run: python setup_mot15.py")
        all_ok = False
    else:
        print_success(f"MOT15 dataset found: {mot15_path}")
    
    # Check test sequences
    test_path = mot15_path / "test"
    if not test_path.exists():
        print_error(f"Test set not found at: {test_path}")
        all_ok = False
    else:
        found_sequences = [d.name for d in test_path.iterdir() if d.is_dir()]
        expected_sequences = list(MOT15_TEST_SEQUENCES.keys())
        missing = set(expected_sequences) - set(found_sequences)
        
        if missing:
            print_warning(f"Missing sequences: {', '.join(missing)}")
        else:
            print_success(f"All {len(expected_sequences)} test sequences found")
    
    # Check model config
    if not Path(args.exp_file).exists():
        print_error(f"Model config not found: {args.exp_file}")
        all_ok = False
    else:
        print_success(f"Model config found: {args.exp_file}")
    
    # Check model checkpoint
    if not Path(args.ckpt).exists():
        print_error(f"Model checkpoint not found: {args.ckpt}")
        print_info("Run: python download_model.py nano")
        all_ok = False
    else:
        ckpt_size = Path(args.ckpt).stat().st_size / (1024 * 1024)
        print_success(f"Model checkpoint found: {args.ckpt} ({ckpt_size:.1f} MB)")
    
    # Check demo_track.py
    demo_track = Path("tools/demo_track.py")
    if not demo_track.exists():
        print_error(f"Demo script not found: {demo_track}")
        all_ok = False
    else:
        print_success(f"Demo script found: {demo_track}")
    
    print()
    return all_ok


def get_sequence_info(seq_name, data_root):
    """Get information about a sequence"""
    seq_path = Path(data_root) / "MOT15" / "test" / seq_name
    img_path = seq_path / "img1"
    
    if not img_path.exists():
        return None
    
    # Count images
    images = list(img_path.glob("*.jpg")) + list(img_path.glob("*.png"))
    num_frames = len(images)
    
    # Get image dimensions from first frame
    if images:
        first_img = cv2.imread(str(images[0]))
        if first_img is not None:
            height, width = first_img.shape[:2]
            resolution = f"{width}x{height}"
        else:
            resolution = "Unknown"
    else:
        resolution = "Unknown"
    
    # Get info from dict
    seq_info = MOT15_TEST_SEQUENCES.get(seq_name, {})
    
    return {
        "name": seq_name,
        "path": str(seq_path),
        "img_path": str(img_path),
        "frames": num_frames,
        "fps": seq_info.get("fps", 30),
        "resolution": resolution,
        "description": seq_info.get("description", "")
    }


def convert_sequence_to_video(seq_info, output_dir, fps=None):
    """Convert image sequence to video for demo_track.py"""
    if fps is None:
        fps = seq_info["fps"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = output_dir / f"{seq_info['name']}.mp4"
    
    if video_path.exists():
        print_info(f"Video already exists: {video_path}")
        return str(video_path)
    
    print_info(f"Converting {seq_info['name']} to video...")
    
    img_path = Path(seq_info["img_path"])
    images = sorted(list(img_path.glob("*.jpg")) + list(img_path.glob("*.png")))
    
    if not images:
        print_error(f"No images found in {img_path}")
        return None
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        print_error(f"Could not read first image: {images[0]}")
        return None
    
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    # Write frames
    for i, img_file in enumerate(images):
        img = cv2.imread(str(img_file))
        if img is not None:
            out.write(img)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(images)} frames")
    
    out.release()
    
    print_success(f"Video created: {video_path}")
    return str(video_path)


def track_sequence_from_images(seq_info, args):
    """Run tracking directly on image sequence"""
    print_info(f"Processing sequence: {seq_info['name']}")
    print_info(f"  Frames: {seq_info['frames']}")
    print_info(f"  Resolution: {seq_info['resolution']}")
    print_info(f"  Description: {seq_info['description']}")
    
    # Build command for demo_track.py with image path
    cmd = [
        sys.executable,
        "tools/demo_track.py",
        "image",
        "-f", args.exp_file,
        "-c", args.ckpt,
        "--path", seq_info["img_path"],
        "--device", args.device,
        "--track_thresh", str(args.track_thresh),
        "--track_buffer", str(args.track_buffer),
        "--match_thresh", str(args.match_thresh),
        "--fps", str(seq_info["fps"]),
    ]
    
    if args.save_result:
        cmd.append("--save_result")
    
    if args.fp16:
        cmd.append("--fp16")
    
    if args.fuse:
        cmd.append("--fuse")
    
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    # Run tracking
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed_time = time.time() - start_time
        
        # Calculate FPS
        processing_fps = seq_info["frames"] / elapsed_time if elapsed_time > 0 else 0
        
        print_success(f"Completed: {seq_info['name']}")
        print_success(f"  Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
        print_success(f"  Processing FPS: {processing_fps:.2f}")
        print()
        
        return {
            "sequence": seq_info["name"],
            "status": "SUCCESS",
            "frames": seq_info["frames"],
            "time": elapsed_time,
            "fps": processing_fps,
            "error": None
        }
    
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print_error(f"Failed: {seq_info['name']} (exit code: {e.returncode})")
        
        return {
            "sequence": seq_info["name"],
            "status": "FAILED",
            "frames": seq_info["frames"],
            "time": elapsed_time,
            "fps": 0,
            "error": f"Exit code: {e.returncode}"
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print_error(f"Error: {seq_info['name']} - {e}")
        
        return {
            "sequence": seq_info["name"],
            "status": "ERROR",
            "frames": seq_info["frames"],
            "time": elapsed_time,
            "fps": 0,
            "error": str(e)
        }


def find_tracking_results(output_dir):
    """Find generated tracking result files"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    
    # Look for .txt files in track_results
    result_files = []
    for txt_file in output_path.rglob("*.txt"):
        if txt_file.parent.name == "track_results" or "track" in txt_file.stem.lower():
            result_files.append(txt_file)
    
    return sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)


def print_results_summary(results):
    """Print detailed summary of all test results"""
    print_header("Test Results Summary")
    
    # Print table header
    print(f"{'Sequence':<20} {'Frames':<8} {'Status':<12} {'Time':<15} {'Proc.FPS':<12} {'Error':<30}")
    print("-" * 107)
    
    # Print results
    total_time = 0
    total_frames = 0
    success_count = 0
    
    for result in results:
        seq = result["sequence"]
        frames = result["frames"]
        status = result["status"]
        time_val = result["time"]
        fps = result["fps"]
        error = result.get("error", "")
        
        if time_val > 0:
            time_str = f"{time_val:.2f}s ({time_val/60:.2f}m)"
            total_time += time_val
        else:
            time_str = "N/A"
        
        if fps > 0:
            fps_str = f"{fps:.2f}"
        else:
            fps_str = "N/A"
        
        if error is None:
            error = ""
        
        # Color code status
        if status == "SUCCESS":
            status_colored = f"{Colors.GREEN}{status}{Colors.END}"
            success_count += 1
            total_frames += frames
        else:
            status_colored = f"{Colors.RED}{status}{Colors.END}"
        
        print(f"{seq:<20} {frames:<8} {status_colored:<20} {time_str:<15} {fps_str:<12} {error:<30}")
    
    print("-" * 107)
    print()
    
    # Print statistics
    print_info(f"Total sequences: {len(results)}")
    print_info(f"Successful: {success_count}/{len(results)}")
    print_info(f"Total frames processed: {total_frames:,}")
    print_info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    if total_time > 0 and total_frames > 0:
        avg_fps = total_frames / total_time
        print_info(f"Average processing FPS: {avg_fps:.2f}")
    
    print()


def save_results_report(results, output_dir):
    """Save detailed results report to file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"mot15_nano_test_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ByteTrack MOT15 Test Set Evaluation - Nano Model\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: bytetrack_nano_mot17.pth.tar\n")
        f.write(f"Config: exps/example/mot/yolox_nano_mix_det.py\n")
        f.write(f"Parameters: 0.90M\n")
        f.write(f"FLOPs: 3.99G\n\n")
        
        f.write("Tracking Parameters:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Track threshold: 0.6\n")
        f.write(f"Track buffer: 30\n")
        f.write(f"Match threshold: 0.8\n\n")
        
        f.write("Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Sequence':<20} {'Frames':<8} {'Status':<12} {'Time(s)':<12} {'Proc.FPS':<12}\n")
        f.write("-" * 80 + "\n")
        
        total_time = 0
        total_frames = 0
        success_count = 0
        
        for result in results:
            seq = result["sequence"]
            frames = result["frames"]
            status = result["status"]
            time_val = result["time"]
            fps = result["fps"]
            
            f.write(f"{seq:<20} {frames:<8} {status:<12} {time_val:<12.2f} {fps:<12.2f}\n")
            
            if status == "SUCCESS":
                success_count += 1
                total_frames += frames
            
            total_time += time_val
        
        f.write("-" * 80 + "\n\n")
        
        f.write("Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total sequences: {len(results)}\n")
        f.write(f"Successful: {success_count}/{len(results)}\n")
        f.write(f"Total frames processed: {total_frames:,}\n")
        f.write(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)\n")
        
        if total_time > 0 and total_frames > 0:
            avg_fps = total_frames / total_time
            f.write(f"Average processing FPS: {avg_fps:.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print_success(f"Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="ByteTrack MOT15 Test Set Evaluation with Nano Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all MOT15 sequences with nano model
  python test_mot15_nano.py
  
  # Test specific sequences only
  python test_mot15_nano.py --sequences Venice-1 KITTI-19
  
  # Test with GPU
  python test_mot15_nano.py --device gpu --fp16 --fuse
  
  # Test with custom tracking parameters
  python test_mot15_nano.py --track-thresh 0.5 --match-thresh 0.7
        """
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets",
        help="Root directory for datasets (default: datasets)"
    )
    
    parser.add_argument(
        "--exp_file",
        type=str,
        default="exps/example/mot/yolox_nano_mix_det.py",
        help="Model config file (default: yolox_nano_mix_det.py)"
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        default="pretrained/bytetrack_nano_mot17.pth.tar",
        help="Model checkpoint file (default: bytetrack_nano_mot17.pth.tar)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to use (default: cpu)"
    )
    
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=list(MOT15_TEST_SEQUENCES.keys()),
        help="Specific sequences to test (default: all)"
    )
    
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.6,
        help="Tracking confidence threshold (default: 0.6)"
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
        "--save-result",
        action="store_true",
        default=True,
        help="Save tracking results and videos (default: True)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="YOLOX_outputs/mot15_nano_test",
        help="Output directory for results (default: YOLOX_outputs/mot15_nano_test)"
    )
    
    parser.add_argument(
        "--convert-to-video",
        action="store_true",
        help="Convert image sequences to videos first (slower but cleaner output)"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_header("ByteTrack MOT15 Test Set Evaluation - Nano Model")
    
    # Check prerequisites
    if not check_prerequisites(args):
        print_error("Prerequisites check failed. Please fix the issues above.")
        return 1
    
    # Determine sequences to test
    if args.sequences:
        sequences_to_test = args.sequences
    else:
        sequences_to_test = list(MOT15_TEST_SEQUENCES.keys())
    
    print_info(f"Testing {len(sequences_to_test)} sequences")
    print_info(f"Model: Nano ({args.ckpt})")
    print_info(f"Device: {args.device}")
    print_info(f"Track threshold: {args.track_thresh}")
    print_info(f"Match threshold: {args.match_thresh}")
    print_info(f"Track buffer: {args.track_buffer}")
    print()
    
    # Process each sequence
    results = []
    start_time_all = time.time()
    
    for i, seq_name in enumerate(sequences_to_test, 1):
        print_header(f"Sequence {i}/{len(sequences_to_test)}: {seq_name}")
        
        # Get sequence info
        seq_info = get_sequence_info(seq_name, args.data_root)
        
        if seq_info is None:
            print_error(f"Sequence not found: {seq_name}")
            results.append({
                "sequence": seq_name,
                "status": "NOT_FOUND",
                "frames": 0,
                "time": 0,
                "fps": 0,
                "error": "Sequence not found"
            })
            continue
        
        # Track sequence
        result = track_sequence_from_images(seq_info, args)
        results.append(result)
    
    total_time_all = time.time() - start_time_all
    
    # Print results summary
    print_results_summary(results)
    
    print_info(f"Total execution time: {total_time_all:.2f}s ({total_time_all/60:.2f} min)")
    print()
    
    # Save report
    save_results_report(results, args.output_dir)
    
    # Show output location
    print_header("Output Files")
    print_info(f"Results directory: YOLOX_outputs/")
    print_info("Look for tracking results in subdirectories named after experiment")
    print()
    
    # Find tracking result files
    tracking_files = find_tracking_results("YOLOX_outputs")
    if tracking_files:
        print_info("Recent tracking result files:")
        for txt_file in tracking_files[:10]:
            rel_path = txt_file.relative_to(Path.cwd())
            size_kb = txt_file.stat().st_size / 1024
            mtime = datetime.fromtimestamp(txt_file.stat().st_mtime)
            print(f"  - {rel_path} ({size_kb:.1f} KB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    print()
    print_success("Testing complete!")
    
    # Return exit code
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
