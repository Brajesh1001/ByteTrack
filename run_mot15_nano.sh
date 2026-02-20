#!/bin/bash
set -e

echo "=========================================="
echo " ByteTrack Nano Model - MOT15 Full Test"
echo "=========================================="

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKSPACE_DIR"

# ---------- Detect device ----------
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="gpu"
    FP16_FLAG="--fp16"
    echo "[INFO] GPU detected — using CUDA"
else
    DEVICE="cpu"
    FP16_FLAG=""
    echo "[INFO] No GPU detected — using CPU (slower but works)"
fi

# ---------- Step 1: Install dependencies ----------
echo ""
echo "[Step 1/6] Installing dependencies..."
pip install -q -r requirements.txt
python3 setup.py develop -q 2>/dev/null || python3 setup.py develop
pip install -q cython
pip install -q 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' 2>/dev/null || true
pip install -q cython_bbox 2>/dev/null || pip install -q cython-bbox 2>/dev/null || true
pip install -q gdown
echo "[Step 1/6] Done."

# ---------- Step 2: Download MOT15 dataset ----------
echo ""
echo "[Step 2/6] Downloading MOT15 dataset..."
if [ -d "datasets/MOT15/train" ] && [ -d "datasets/MOT15/test" ]; then
    echo "  MOT15 already exists, skipping download."
else
    mkdir -p datasets
    cd datasets
    if [ ! -f "MOT15.zip" ]; then
        wget -q --show-progress https://motchallenge.net/data/MOT15.zip -O MOT15.zip
    fi
    unzip -q -o MOT15.zip
    cd "$WORKSPACE_DIR"
fi

echo "  Train sequences:"
ls datasets/MOT15/train/ 2>/dev/null || echo "    (none found)"
echo "  Test sequences:"
ls datasets/MOT15/test/ 2>/dev/null || echo "    (none found)"
echo "[Step 2/6] Done."

# ---------- Step 3: Convert MOT15 to COCO format ----------
echo ""
echo "[Step 3/6] Converting MOT15 to COCO format..."
if [ -f "datasets/MOT15/annotations/train.json" ]; then
    echo "  Annotations already exist, skipping conversion."
else
    python3 tools/convert_mot15_to_coco.py
fi
echo "[Step 3/6] Done."

# ---------- Step 4: Download pretrained nano model ----------
echo ""
echo "[Step 4/6] Downloading pretrained nano model weights..."
mkdir -p pretrained
if [ -f "pretrained/bytetrack_nano_mot17.pth.tar" ]; then
    echo "  Weights already exist, skipping download."
else
    python3 -c "
import gdown
gdown.download(id='1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX', output='pretrained/bytetrack_nano_mot17.pth.tar', quiet=False)
"
fi
echo "[Step 4/6] Done."

# ---------- Step 5: Run tracking on MOT15 train (with evaluation) ----------
echo ""
echo "[Step 5/6] Running ByteTrack Nano on MOT15 TRAIN sequences..."
python3 tools/track_mot15.py \
    -f exps/example/mot/yolox_nano_mot15.py \
    -c pretrained/bytetrack_nano_mot17.pth.tar \
    -b 1 -d 1 --fuse \
    --track_thresh 0.4 \
    --match_thresh 0.8 \
    --device "$DEVICE" \
    $FP16_FLAG
echo "[Step 5/6] Done."

# ---------- Step 6: Run tracking on MOT15 test (for submission) ----------
echo ""
echo "[Step 6/6] Running ByteTrack Nano on MOT15 TEST sequences..."
python3 tools/track_mot15.py \
    -f exps/example/mot/yolox_nano_mot15.py \
    -c pretrained/bytetrack_nano_mot17.pth.tar \
    -b 1 -d 1 --fuse --test \
    --track_thresh 0.4 \
    --match_thresh 0.8 \
    --device "$DEVICE" \
    $FP16_FLAG
echo "[Step 6/6] Done."

echo ""
echo "=========================================="
echo " All MOT15 sequences processed!"
echo "=========================================="
echo ""
echo "Results locations:"
echo "  Train results: YOLOX_outputs/yolox_nano_mot15/track_results/"
echo "  Test results:  YOLOX_outputs/yolox_nano_mot15/track_results/"
echo ""
echo "To submit test results to MOTChallenge:"
echo "  1. Go to https://motchallenge.net/"
echo "  2. Upload the .txt files from the test track_results folder"
echo ""
