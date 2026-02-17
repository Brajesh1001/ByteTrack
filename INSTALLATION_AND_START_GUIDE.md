# ByteTrack Installation and Start Guide

This guide gives you a clean path from a fresh machine to your first ByteTrack run.

## Prerequisites

- OS: Linux (recommended)
- Python: 3.6+ (3.8+ recommended)
- GPU: NVIDIA GPU + CUDA for best performance (CPU works, but is much slower)
- System packages needed to compile extensions:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
```

## Option A: Local installation (recommended for development)

1) Clone and enter the repo:

```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
```

2) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

3) Install Python dependencies:

```bash
pip install -r requirements.txt
python setup.py develop
pip install cython
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install cython_bbox
```

4) Quick sanity check:

```bash
python - <<'PY'
import torch
import cv2
import yolox
print("torch:", torch.__version__)
print("opencv:", cv2.__version__)
print("yolox:", yolox.__version__)
PY
```

## Option B: Docker installation

Build and run:

```bash
docker build -t bytetrack:latest .
mkdir -p pretrained datasets YOLOX_outputs
xhost +local:
docker run --gpus all -it --rm \
  -v "$PWD/pretrained:/workspace/ByteTrack/pretrained" \
  -v "$PWD/datasets:/workspace/ByteTrack/datasets" \
  -v "$PWD/YOLOX_outputs:/workspace/ByteTrack/YOLOX_outputs" \
  -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
  --device /dev/video0:/dev/video0:mwr \
  --net=host \
  -e XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
  -e DISPLAY="$DISPLAY" \
  --privileged \
  bytetrack:latest
```

## Start in 5 minutes (first run)

1) Create the pretrained directory:

```bash
mkdir -p pretrained
```

2) Download a checkpoint from the model zoo in `README.md` and place it at:

```text
pretrained/bytetrack_x_mot17.pth.tar
```

3) Run tracking on your own video:

```bash
python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --path /absolute/path/to/your_video.mp4 \
  --fp16 \
  --fuse \
  --save_result
```

Output files are written under:

```text
YOLOX_outputs/yolox_x_mix_det/track_vis/<timestamp>/
```

## Optional: webcam run

```bash
python3 tools/demo_track.py webcam \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --camid 0 \
  --fp16 \
  --fuse \
  --save_result
```

## Troubleshooting

- `ModuleNotFoundError: pycocotools`  
  Re-run:
  ```bash
  pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
  ```

- C++ extension build fails during `python setup.py develop`  
  Make sure compiler tools are installed:
  ```bash
  sudo apt-get install -y build-essential python3-dev
  ```

- CUDA/GPU not detected  
  Verify driver and CUDA setup, or run with CPU by adding:
  ```bash
  --device cpu
  ```
