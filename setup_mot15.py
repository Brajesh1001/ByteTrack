"""
Setup script for MOT15 dataset
Downloads and organizes MOT15 dataset for ByteTrack evaluation
"""
import os
import zipfile
import argparse
import urllib.request
from pathlib import Path


# MOT15 Download URLs
MOT15_URL = "https://motchallenge.net/data/2DMOT2015.zip"


def download_file(url, destination):
    """Download file with progress bar"""
    print(f"Downloading {url}...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100.0 / total_size, 100)
        print(f"\rProgress: {percent:.1f}%", end='')
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def organize_mot15_structure(base_path):
    """
    Organize MOT15 into the expected structure:
    datasets/mot15/
        train/
            SEQUENCE_NAME/
                img1/
                gt/
                det/
        test/
            SEQUENCE_NAME/
                img1/
                det/
    """
    mot15_path = Path(base_path) / "2DMOT2015"
    target_path = Path(base_path) / "mot15"
    
    if not mot15_path.exists():
        print(f"Error: {mot15_path} does not exist")
        return False
    
    # Create target structure
    target_path.mkdir(exist_ok=True)
    (target_path / "train").mkdir(exist_ok=True)
    (target_path / "test").mkdir(exist_ok=True)
    
    # Move train sequences
    train_path = mot15_path / "train"
    if train_path.exists():
        print("Organizing training sequences...")
        for seq in train_path.iterdir():
            if seq.is_dir() and not seq.name.startswith('.'):
                target_seq = target_path / "train" / seq.name
                if not target_seq.exists():
                    print(f"  Moving {seq.name}...")
                    seq.rename(target_seq)
    
    # Move test sequences
    test_path = mot15_path / "test"
    if test_path.exists():
        print("Organizing test sequences...")
        for seq in test_path.iterdir():
            if seq.is_dir() and not seq.name.startswith('.'):
                target_seq = target_path / "test" / seq.name
                if not target_seq.exists():
                    print(f"  Moving {seq.name}...")
                    seq.rename(target_seq)
    
    print("Dataset organization complete!")
    return True


def setup_mot15(data_root="datasets", download=True, organize=True):
    """
    Main setup function for MOT15
    
    Args:
        data_root: Root directory for datasets
        download: Whether to download the dataset
        organize: Whether to organize the directory structure
    """
    data_path = Path(data_root)
    data_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_path / "2DMOT2015.zip"
    mot15_path = data_path / "mot15"
    
    # Check if already set up
    if mot15_path.exists() and any(mot15_path.iterdir()):
        print("MOT15 dataset already exists!")
        response = input("Do you want to re-download and setup? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Download dataset
    if download:
        if not zip_path.exists():
            print("\n=== Downloading MOT15 Dataset ===")
            print("Note: This is a large file (~1.3 GB)")
            if not download_file(MOT15_URL, zip_path):
                print("Download failed. You can manually download from:")
                print(f"  {MOT15_URL}")
                print(f"And place it at: {zip_path}")
                return
        else:
            print(f"Using existing download: {zip_path}")
        
        # Extract dataset
        print("\n=== Extracting Dataset ===")
        if not extract_zip(zip_path, data_path):
            return
    
    # Organize structure
    if organize:
        print("\n=== Organizing Directory Structure ===")
        if not organize_mot15_structure(data_path):
            return
    
    print("\n" + "="*60)
    print("MOT15 Setup Complete!")
    print("="*60)
    print(f"\nDataset location: {mot15_path}")
    print("\nNext steps:")
    print("1. Convert annotations to COCO format:")
    print("   python tools/convert_mot15_to_coco.py")
    print("\n2. Run tracking on test set:")
    print("   python tools/track_mot15.py")
    print("\n3. Evaluate results:")
    print("   python tools/mota.py")
    

def main():
    parser = argparse.ArgumentParser(description="Setup MOT15 dataset for ByteTrack")
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading (use if you already have the zip file)"
    )
    parser.add_argument(
        "--skip_organize",
        action="store_true",
        help="Skip organizing directory structure"
    )
    
    args = parser.parse_args()
    
    setup_mot15(
        data_root=args.data_root,
        download=not args.skip_download,
        organize=not args.skip_organize
    )


if __name__ == "__main__":
    main()
