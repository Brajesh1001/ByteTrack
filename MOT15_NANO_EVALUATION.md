# ByteTrack Nano Model -- MOT15 Evaluation Guide

This document provides a complete step-by-step guide for evaluating the **ByteTrack Nano** model on the [MOT15](https://motchallenge.net/data/MOT15/) benchmark dataset.

---

## Overview

| Property         | Value                          |
|------------------|--------------------------------|
| Model            | ByteTrack Nano (YOLOX-Nano)    |
| Parameters       | 0.90 M                         |
| FLOPs            | 3.99 G                         |
| Input Size       | 608 x 1088                     |
| Dataset          | MOT15 (train + test)           |
| Backbone         | YOLOPAFPN (depthwise=True)     |

---

## Step-by-Step Instructions

### Step 1: Install Dependencies

```bash
cd <ByteTrack_HOME>
pip3 install -r requirements.txt
python3 setup.py develop
pip3 install cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
```

### Step 2: Download MOT15 Dataset

Download the MOT15 dataset from [https://motchallenge.net/data/MOT15/](https://motchallenge.net/data/MOT15/) and place it under `datasets/MOT15/`:

```
datasets/
  └── MOT15/
       ├── train/
       │    ├── ADL-Rundle-6/
       │    │    ├── img1/
       │    │    ├── gt/
       │    │    │    └── gt.txt
       │    │    └── det/
       │    │         └── det.txt
       │    ├── ADL-Rundle-8/
       │    ├── ETH-Bahnhof/
       │    ├── ETH-Pedcross2/
       │    ├── ETH-Sunnyday/
       │    ├── KITTI-13/
       │    ├── KITTI-17/
       │    ├── PETS09-S2L1/
       │    ├── TUD-Campus/
       │    ├── TUD-Stadtmitte/
       │    └── Venice-2/
       └── test/
            ├── ADL-Rundle-1/
            ├── ADL-Rundle-3/
            ├── AVG-TownCentre/
            ├── ETH-Crossing/
            ├── ETH-Jelmoli/
            ├── ETH-Linthescher/
            ├── KITTI-16/
            ├── KITTI-19/
            ├── PETS09-S2L2/
            ├── TUD-Crossing/
            └── Venice-1/
```

You can download using wget:
```bash
mkdir -p datasets/MOT15
cd datasets/MOT15
wget https://motchallenge.net/data/MOT15.zip
unzip MOT15.zip
cd ../..
```

### Step 3: Convert MOT15 to COCO Format

```bash
python3 tools/convert_mot15_to_coco.py
```

This generates annotation JSON files under `datasets/MOT15/annotations/`:
- `train_half.json` -- first half of training sequences (for training)
- `val_half.json` -- second half of training sequences (for validation)
- `train.json` -- all training sequences
- `test.json` -- all test sequences

### Step 4: Download Pretrained Nano Model

Download the ByteTrack Nano pretrained weights (trained on CrowdHuman + MOT17 + Cityperson + ETHZ):

| Model | Link |
|-------|------|
| bytetrack_nano_mot17 | [Google Drive](https://drive.google.com/file/d/1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX/view?usp=sharing) |
| bytetrack_nano_mot17 | [Baidu (code:1ub8)](https://pan.baidu.com/s/1dMxqBPP7lFNRZ3kFgDmWdw) |

```bash
mkdir -p pretrained
# Place the downloaded file as:
# pretrained/bytetrack_nano_mot17.pth.tar
```

### Step 5: Run Evaluation on MOT15

#### Option A: Evaluate on MOT15 train set (with ground truth metrics)

```bash
python3 tools/evaluate_mot15_nano.py \
    -f exps/example/mot/yolox_nano_mot15.py \
    -c pretrained/bytetrack_nano_mot17.pth.tar \
    -b 1 -d 1 --fp16 --fuse
```

#### Option B: Evaluate using the standard track.py (generates result txt files)

```bash
python3 tools/track.py \
    -f exps/example/mot/yolox_nano_mot15.py \
    -c pretrained/bytetrack_nano_mot17.pth.tar \
    -b 1 -d 1 --fp16 --fuse
```

#### Option C: Generate test set submissions for MOTChallenge

```bash
python3 tools/track.py \
    -f exps/example/mot/yolox_nano_mot15.py \
    -c pretrained/bytetrack_nano_mot17.pth.tar \
    -b 1 -d 1 --fp16 --fuse --test
python3 tools/interpolation.py
```

Then submit the txt files from `YOLOX_outputs/yolox_nano_mot15/track_results/` to the [MOT15 leaderboard](https://motchallenge.net/results/MOT15/).

### Step 6: Review Results

The evaluation script prints a table with per-sequence and overall metrics.

---

## MOT15 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MOTA** | Multi-Object Tracking Accuracy -- combines FP, FN, and ID switches |
| **IDF1** | ID F1 Score -- ratio of correctly identified detections over average of GT and computed detections |
| **HOTA** | Higher Order Tracking Accuracy -- balances detection and association |
| **MT** | Mostly Tracked targets (tracked >= 80% of lifespan) |
| **ML** | Mostly Lost targets (tracked <= 20% of lifespan) |
| **FP** | False Positives |
| **FN** | False Negatives (Misses) |
| **IDs** | ID Switches |
| **Frag** | Fragmentations |
| **MOTP** | Multi-Object Tracking Precision (average overlap with GT) |
| **Rcll** | Recall |
| **Prcn** | Precision |

---

## Results Table

### ByteTrack Nano on MOT15 -- Per-Sequence Results (Train Set)

| Sequence         | MOTA | IDF1 | MT | ML | FP | FN | IDs | Frag | MOTP |
|------------------|------|------|----|----|----|----|-----|------|------|
| ADL-Rundle-6     |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| ADL-Rundle-8     |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| ETH-Bahnhof      |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| ETH-Pedcross2    |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| ETH-Sunnyday     |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| KITTI-13         |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| KITTI-17         |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| PETS09-S2L1      |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| TUD-Campus       |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| TUD-Stadtmitte   |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| Venice-2         |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |
| **OVERALL**      |  --  |  --  | -- | -- | -- | -- |  -- |  --  |  --  |

> Fill in the `--` placeholders with actual numbers after running the evaluation (Step 5).

### ByteTrack Nano -- Cross-Dataset Comparison

| Dataset | MOTA | IDF1 | IDs | Params(M) | FLOPs(G) |
|---------|------|------|-----|-----------|----------|
| MOT17   | 69.0 | 66.3 | 531 | 0.90      | 3.99     |
| MOT15   |  --  |  --  |  -- | 0.90      | 3.99     |

### ByteTrack Model Variants on MOT15 (for comparison if running multiple models)

| Model               | MOTA | IDF1 | HOTA | IDs | FP    | FN    | Params(M) | FLOPs(G) |
|---------------------|------|------|------|-----|-------|-------|-----------|----------|
| bytetrack_nano      |  --  |  --  |  --  |  -- |   --  |   --  | 0.90      | 3.99     |
| bytetrack_tiny      |  --  |  --  |  --  |  -- |   --  |   --  | 5.03      | 24.45    |
| bytetrack_s         |  --  |  --  |  --  |  -- |   --  |   --  | 9.0       | 26.8     |
| bytetrack_m         |  --  |  --  |  --  |  -- |   --  |   --  | 25.3      | 73.8     |
| bytetrack_l         |  --  |  --  |  --  |  -- |   --  |   --  | 54.2      | 155.6    |
| bytetrack_x         |  --  |  --  |  --  |  -- |   --  |   --  | 99.1      | 281.9    |

---

## Quick Reference: Commands Summary

```bash
# 1. Install
pip3 install -r requirements.txt && python3 setup.py develop
pip3 install cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox

# 2. Download & prepare MOT15
mkdir -p datasets/MOT15 && cd datasets/MOT15
wget https://motchallenge.net/data/MOT15.zip && unzip MOT15.zip
cd ../..

# 3. Convert to COCO format
python3 tools/convert_mot15_to_coco.py

# 4. Download nano model weights into pretrained/

# 5. Run evaluation
python3 tools/evaluate_mot15_nano.py \
    -f exps/example/mot/yolox_nano_mot15.py \
    -c pretrained/bytetrack_nano_mot17.pth.tar \
    -b 1 -d 1 --fp16 --fuse
```

---

## Notes

- The nano model uses **depthwise separable convolutions**, making it significantly lighter than standard ByteTrack models.
- MOT15 sequences have varying resolutions and camera angles. The default test size of 608x1088 works well across sequences.
- If you observe low MOTA on specific sequences, you can try adjusting `--track_thresh` (default 0.6) and `--conf` (default 0.01).
- For test set submission, run `tools/interpolation.py` after tracking to fill gaps in trajectories and improve MOTA.
- MOT15 ground truth does **not** include class labels or visibility flags, unlike MOT17. All labeled objects are treated as pedestrians.
