# Plan: Testing YOLOX-Nano on MOT15 Dataset

## Overview

This plan describes how to use the **ByteTrack YOLOX-Nano** model to run multi-object tracking
on **all videos** in the [MOT15 dataset](https://motchallenge.net/data/MOT15/).

The nano model is the lightest ByteTrack variant (0.90M params, 3.99 GFLOPs) and can run on
both GPU and CPU (CPU will be slower but functional).

---

## MOT15 Dataset

### Train Sequences (11 videos, with ground truth)
| Sequence | Frames | Resolution | FPS | Description |
|---|---|---|---|---|
| ADL-Rundle-6 | 525 | 1920x1080 | 30 | Indoor/outdoor pedestrians |
| ADL-Rundle-8 | 654 | 1920x1080 | 30 | Indoor/outdoor pedestrians |
| ETH-Bahnhof | 1000 | 640x480 | 14 | Zurich train station |
| ETH-Pedcross2 | 840 | 640x480 | 14 | Pedestrian crossing |
| ETH-Sunnyday | 354 | 640x480 | 14 | Outdoor sunny scene |
| KITTI-13 | 340 | 1242x375 | 10 | Autonomous driving |
| KITTI-17 | 145 | 1224x370 | 10 | Autonomous driving |
| PETS09-S2L1 | 795 | 768x576 | 7 | Outdoor surveillance |
| TUD-Campus | 71 | 640x480 | 25 | Campus pedestrians |
| TUD-Stadtmitte | 179 | 640x480 | 25 | City center pedestrians |
| Venice-2 | 600 | 1920x1080 | 30 | Venice square |

### Test Sequences (11 videos, no ground truth — submit to MOTChallenge)
| Sequence | Frames | Resolution | FPS |
|---|---|---|---|
| ADL-Rundle-1 | 500 | 1920x1080 | 30 |
| ADL-Rundle-3 | 625 | 1920x1080 | 30 |
| AVG-TownCentre | 450 | 1920x1080 | 2.5 |
| ETH-Crossing | 219 | 640x480 | 14 |
| ETH-Jelmoli | 440 | 640x480 | 14 |
| ETH-Linthescher | 1194 | 640x480 | 14 |
| KITTI-16 | 209 | 1224x370 | 10 |
| KITTI-19 | 1059 | 1238x374 | 10 |
| PETS09-S2L2 | 436 | 768x576 | 7 |
| TUD-Crossing | 201 | 640x480 | 25 |
| Venice-1 | 450 | 1920x1080 | 30 |

---

## Step-by-Step Plan

### Step 1: Environment Setup

Install all dependencies for ByteTrack:

```bash
pip install -r requirements.txt
python setup.py develop
pip install cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
```

### Step 2: Download the MOT15 Dataset

Download MOT15 from the official source and place it under `datasets/MOT15/`:

```bash
cd datasets
wget https://motchallenge.net/data/MOT15.zip
unzip MOT15.zip
```

Expected directory structure:
```
datasets/
  MOT15/
    train/
      ADL-Rundle-6/
        det/det.txt
        gt/gt.txt
        img1/000001.jpg ...
      ADL-Rundle-8/
      ETH-Bahnhof/
      ETH-Pedcross2/
      ETH-Sunnyday/
      KITTI-13/
      KITTI-17/
      PETS09-S2L1/
      TUD-Campus/
      TUD-Stadtmitte/
      Venice-2/
    test/
      ADL-Rundle-1/
      ADL-Rundle-3/
      AVG-TownCentre/
      ETH-Crossing/
      ETH-Jelmoli/
      ETH-Linthescher/
      KITTI-16/
      KITTI-19/
      PETS09-S2L2/
      TUD-Crossing/
      Venice-1/
```

### Step 3: Convert MOT15 to COCO Format

Run the MOT15-specific conversion script to create annotation JSON files:

```bash
python tools/convert_mot15_to_coco.py
```

This creates:
- `datasets/MOT15/annotations/train.json` — all training sequences
- `datasets/MOT15/annotations/train_half.json` — first half of training frames
- `datasets/MOT15/annotations/val_half.json` — second half of training frames
- `datasets/MOT15/annotations/test.json` — all test sequences

### Step 4: Download Pretrained Nano Model Weights

Download the ByteTrack nano model pretrained on MOT17 (trained on CrowdHuman + MOT17 + Cityperson + ETHZ):

```bash
mkdir -p pretrained
# Google Drive link from README:
# https://drive.google.com/file/d/1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX/view
# Or use gdown:
pip install gdown
gdown 1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX -O pretrained/bytetrack_nano_mot17.pth.tar
```

### Step 5: Run Tracking on MOT15 Train Set (with MOTA evaluation)

Run ByteTrack with the nano model on all MOT15 training sequences:

```bash
python tools/track_mot15.py \
  -f exps/example/mot/yolox_nano_mot15.py \
  -c pretrained/bytetrack_nano_mot17.pth.tar \
  -b 1 -d 1 --fuse \
  --track_thresh 0.4 \
  --match_thresh 0.8 \
  --device gpu
```

For CPU-only environments:
```bash
python tools/track_mot15.py \
  -f exps/example/mot/yolox_nano_mot15.py \
  -c pretrained/bytetrack_nano_mot17.pth.tar \
  -b 1 -d 1 --fuse \
  --track_thresh 0.4 \
  --match_thresh 0.8 \
  --device cpu
```

This will:
1. Load the nano model with pretrained weights
2. Run detection + ByteTrack on every frame of all 11 training videos
3. Save per-sequence tracking results as `.txt` files
4. Compute and print MOT metrics (MOTA, IDF1, etc.) using ground truth

### Step 6: Run Tracking on MOT15 Test Set (for MOTChallenge submission)

```bash
python tools/track_mot15.py \
  -f exps/example/mot/yolox_nano_mot15.py \
  -c pretrained/bytetrack_nano_mot17.pth.tar \
  -b 1 -d 1 --fuse --test \
  --track_thresh 0.4 \
  --match_thresh 0.8 \
  --device gpu
```

This produces `.txt` result files for each test sequence that can be submitted to
[MOTChallenge](https://motchallenge.net/) for official evaluation.

### Step 7: (Optional) Run Interpolation for Better Results

```bash
python tools/interpolation.py
```

### Automated Run

Use the provided master script to run everything automatically:

```bash
bash run_mot15_nano.sh
```

---

## Files Created for This Plan

| File | Purpose |
|---|---|
| `PLAN_MOT15_NANO.md` | This plan document |
| `tools/convert_mot15_to_coco.py` | Convert MOT15 dataset to COCO annotation format |
| `exps/example/mot/yolox_nano_mot15.py` | YOLOX-Nano experiment config tailored for MOT15 |
| `tools/track_mot15.py` | Tracking + evaluation script adapted for MOT15 sequences |
| `run_mot15_nano.sh` | Master script that automates the full pipeline |

---

## Expected Output

After running on the **train set**, you will see a table like:

```
          IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML   FP    FN  IDs   FM  MOTA  MOTP
OVERALL   xx.x  xx.x  xx.x  xx.x  xx.x  xx  xx  xx  xx  xxxx  xxxx  xxx  xxx  xx.x  xx.x
```

The nano model on MOT17 achieves ~69.0 MOTA. On MOT15 (which has lower resolution
sequences and different scenes) expect somewhat different results, but the model should
generalize reasonably since it was trained on diverse pedestrian data.

---

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLOX-Nano (depthwise separable convolutions) |
| Backbone depth | 0.33 |
| Backbone width | 0.25 |
| Parameters | 0.90M |
| FLOPs | 3.99G |
| Input size | 608 x 1088 |
| Test confidence | 0.001 |
| NMS threshold | 0.7 |
| Tracker | BYTETracker |
