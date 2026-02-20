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

## MOT15 Evaluation Metrics -- What to Check

The evaluation produces **16 metrics** in two groups. Here is every metric, what
it measures, why it matters, and whether higher or lower is better.

### Primary Metrics (report these first)

These are the headline numbers your manager will look at.

| # | Metric   | Full Name                          | What It Measures | Good Direction | Formula / Note |
|---|----------|------------------------------------|------------------|----------------|----------------|
| 1 | **MOTA** | Multi-Object Tracking Accuracy     | Overall tracking quality combining missed targets, false alarms, and identity switches into one number | Higher is better (max 100%) | `MOTA = 1 - (FN + FP + IDs) / GT` |
| 2 | **IDF1** | ID F1 Score                        | How well the tracker maintains correct identities over time | Higher is better (max 100%) | Harmonic mean of ID Precision and ID Recall |
| 3 | **HOTA** | Higher Order Tracking Accuracy     | Balanced combination of detection accuracy and association accuracy | Higher is better (max 100%) | Geometric mean of DetA and AssA |

### Detection Quality Metrics

How well does the model *find* objects (regardless of identity)?

| # | Metric   | Full Name                          | What It Measures | Good Direction |
|---|----------|------------------------------------|------------------|----------------|
| 4 | **Rcll** (Recall) | Recall                    | Fraction of ground-truth objects that were detected | Higher is better |
| 5 | **Prcn** (Precision) | Precision              | Fraction of detections that match a real object | Higher is better |
| 6 | **FP**   | False Positives                    | Number of detector outputs that do not match any ground-truth object (ghost detections) | Lower is better |
| 7 | **FN**   | False Negatives (Misses)           | Number of ground-truth objects the detector failed to find | Lower is better |

### Identity / Association Metrics

How well does the tracker keep the *same ID* on each person across frames?

| # | Metric   | Full Name                          | What It Measures | Good Direction |
|---|----------|------------------------------------|------------------|----------------|
| 8 | **IDs**  | ID Switches                        | Number of times a tracked object's identity changes (e.g. person A becomes person B) | Lower is better |
| 9 | **Frag** | Fragmentations                     | Number of times a ground-truth trajectory is interrupted (tracker loses then re-finds the target) | Lower is better |
| 10 | **IDP** | ID Precision                       | Of the detections assigned an ID, how many have the correct ID | Higher is better |
| 11 | **IDR** | ID Recall                          | Of the ground-truth IDs, how many were correctly recovered | Higher is better |

### Target Lifecycle Metrics

How completely does the tracker follow each person through the sequence?

| # | Metric   | Full Name                          | What It Measures | Good Direction |
|---|----------|------------------------------------|------------------|----------------|
| 12 | **MT**  | Mostly Tracked                     | % of ground-truth targets tracked for >= 80% of their lifespan | Higher is better |
| 13 | **PT**  | Partially Tracked                  | % of ground-truth targets tracked between 20%-80% of their lifespan | Context-dependent |
| 14 | **ML**  | Mostly Lost                        | % of ground-truth targets tracked for <= 20% of their lifespan | Lower is better |

### Localization Metric

| # | Metric   | Full Name                          | What It Measures | Good Direction |
|---|----------|------------------------------------|------------------|----------------|
| 15 | **MOTP** | Multi-Object Tracking Precision   | Average overlap (IoU) between matched detections and ground truth -- measures *how accurately* bounding boxes are placed | Higher is better (max 100%) |

### Count Metric

| # | Metric    | Full Name                         | What It Measures |
|---|-----------|-----------------------------------|------------------|
| 16 | **GT**   | Num Ground-Truth Objects           | Total number of annotated ground-truth objects (for reference only, not a quality metric) |

### Quick Cheat Sheet

```
              Higher = Better          Lower = Better
              ───────────────          ──────────────
  Headline:   MOTA, IDF1, HOTA        -
  Detection:  Recall, Precision        FP, FN
  Identity:   IDP, IDR                 IDs, Frag
  Lifecycle:  MT                       ML
  Accuracy:   MOTP                     -
```

### What the Evaluation Script Prints

The `evaluate_mot15_nano.py` script prints **two tables** to the console:

1. **Normalized table** -- Rcll, Prcn, MT, ML, FP, FN, IDs, Frag as ratios + MOTA, MOTP (one row per sequence + OVERALL)
2. **Standard MOTChallenge table** -- IDF1, IDP, IDR, Rcll, Prcn, GT, MT, PT, ML, FP, FN, IDs, FM, MOTA, MOTP (one row per sequence + OVERALL)

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
