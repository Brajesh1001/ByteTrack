#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ByteTrack Demo Runner with Environment Variables
Reads configuration from .env file
"""

import os
import sys
from pathlib import Path

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
except ImportError:
    print("Installing python-dotenv...")
    os.system(f"{sys.executable} -m pip install python-dotenv")
    from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_bool(key, default=False):
    """Convert environment variable to boolean"""
    value = os.getenv(key, str(default))
    return value.lower() in ('true', '1', 'yes', 'on')

def run_bytetrack():
    """Run ByteTrack with configuration from .env file"""
    
    # Get configuration from environment variables
    video_input = os.getenv('VIDEO_INPUT_PATH', './videos/palace.mp4')
    video_output_dir = os.getenv('VIDEO_OUTPUT_DIR', './YOLOX_outputs')
    model_config = os.getenv('MODEL_CONFIG', 'exps/example/mot/yolox_nano_mix_det.py')
    model_checkpoint = os.getenv('MODEL_CHECKPOINT', 'pretrained/bytetrack_nano_mot17.pth.tar')
    device = os.getenv('DEVICE', 'cpu')
    track_thresh = float(os.getenv('TRACK_THRESH', '0.5'))
    track_buffer = int(os.getenv('TRACK_BUFFER', '30'))
    match_thresh = float(os.getenv('MATCH_THRESH', '0.8'))
    fps = int(os.getenv('FPS', '30'))
    fp16 = get_env_bool('FP16', False)
    fuse = get_env_bool('FUSE', False)
    save_result = get_env_bool('SAVE_RESULT', True)
    
    # Verify paths exist
    if not Path(video_input).exists():
        print(f"Error: Video input path does not exist: {video_input}")
        print("Please update VIDEO_INPUT_PATH in .env file")
        return 1
    
    if not Path(model_config).exists():
        print(f"Error: Model config does not exist: {model_config}")
        return 1
    
    if not Path(model_checkpoint).exists():
        print(f"Error: Model checkpoint does not exist: {model_checkpoint}")
        print("Run: python download_model.py <model_name>")
        return 1
    
    # Build command
    cmd_parts = [
        sys.executable,
        "tools/demo_track.py",
        "video",
        "-f", model_config,
        "-c", model_checkpoint,
        "--path", video_input,
        "--device", device,
        "--track_thresh", str(track_thresh),
        "--track_buffer", str(track_buffer),
        "--match_thresh", str(match_thresh),
        "--fps", str(fps),
    ]
    
    if fp16:
        cmd_parts.append("--fp16")
    if fuse:
        cmd_parts.append("--fuse")
    if save_result:
        cmd_parts.append("--save_result")
    
    # Print configuration
    print("=" * 60)
    print("ByteTrack Configuration")
    print("=" * 60)
    print(f"Video Input:      {video_input}")
    print(f"Output Directory: {video_output_dir}")
    print(f"Model Config:     {model_config}")
    print(f"Checkpoint:       {model_checkpoint}")
    print(f"Device:           {device}")
    print(f"Track Threshold:  {track_thresh}")
    print(f"Track Buffer:     {track_buffer}")
    print(f"Match Threshold:  {match_thresh}")
    print(f"FPS:              {fps}")
    print(f"FP16:             {fp16}")
    print(f"Fuse:             {fuse}")
    print(f"Save Result:      {save_result}")
    print("=" * 60)
    print()
    
    # Run command
    cmd = " ".join(cmd_parts)
    print(f"Running: {cmd}")
    print()
    
    return os.system(cmd)

if __name__ == "__main__":
    sys.exit(run_bytetrack())
