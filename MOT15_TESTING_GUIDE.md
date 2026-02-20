# MOT15 Testing with Nano Model

This guide explains how to test the ByteTrack nano model on the MOT15 dataset test sequences.

## Overview

The `test_mot15_nano.py` script provides automated batch processing of all MOT15 test sequences using the nano model. It processes 11 test sequences and generates tracking results in MOTChallenge format.

## Prerequisites

### 1. MOT15 Dataset

Download and setup the MOT15 dataset:

```bash
python setup_mot15.py
```

This will:
- Download MOT15 dataset (~1.3 GB)
- Extract and organize into proper structure
- Place dataset at `datasets/MOT15/`

### 2. Nano Model

The nano model should be at `pretrained/bytetrack_nano_mot17.pth.tar`

If not present, download it:
```bash
python download_model.py nano
```

Or manually download from [Google Drive](https://drive.google.com/file/d/1l5Xdh4fOcpFc18OgQf38VUbqq66-h5fO/view?usp=sharing)

### 3. Convert Annotations (Optional)

If you want to run evaluation with ground truth later:

```bash
python tools/convert_mot15_to_coco.py
```

This creates COCO format annotations at `datasets/MOT15/annotations/`

## MOT15 Test Sequences

The script processes these 11 sequences:

| Sequence | Frames | FPS | Resolution | Description |
|----------|--------|-----|------------|-------------|
| ADL-Rundle-1 | 500 | 30 | 1920x1080 | Busy pedestrian street |
| ADL-Rundle-3 | 625 | 30 | 1920x1080 | Crowded pedestrian street |
| AVG-TownCentre | 450 | 2.5 | 1920x1080 | Elevated pedestrian view |
| ETH-Crossing | 219 | 14 | 640x480 | Moving platform scene |
| ETH-Jelmoli | 440 | 14 | 640x480 | Moving platform scene |
| ETH-Linthescher | 1194 | 14 | 640x480 | Moving platform scene |
| KITTI-16 | 209 | 10 | 1224x370 | Pedestrians from car |
| KITTI-19 | 1059 | 10 | 1238x374 | Street from vehicle |
| PETS09-S2L2 | 436 | 7 | 768x576 | Crowded elevated view |
| TUD-Crossing | 201 | 25 | 640x480 | Road crossing |
| Venice-1 | 450 | 30 | 1920x1080 | Large square scene |

**Total: 5,783 frames**

## Usage

### Basic Usage

Test all sequences with default settings (CPU):

```bash
python test_mot15_nano.py
```

### Windows PowerShell Script

For Windows users, use the PowerShell script:

```powershell
.\run_mot15_test_nano.ps1
```

### Advanced Options

#### Test Specific Sequences

```bash
python test_mot15_nano.py --sequences Venice-1 KITTI-19 ADL-Rundle-1
```

#### Use GPU

```bash
python test_mot15_nano.py --device gpu --fp16 --fuse
```

#### Custom Tracking Parameters

```bash
python test_mot15_nano.py --track-thresh 0.5 --match-thresh 0.7 --track-buffer 50
```

### All Options

```bash
python test_mot15_nano.py --help
```

Key parameters:
- `--data_root`: Root directory for datasets (default: `datasets`)
- `--exp_file`: Model config file (default: `exps/example/mot/yolox_nano_mix_det.py`)
- `--ckpt`: Model checkpoint (default: `pretrained/bytetrack_nano_mot17.pth.tar`)
- `--device`: Device to use - `cpu` or `gpu` (default: `cpu`)
- `--sequences`: Specific sequences to test (default: all 11 sequences)
- `--track-thresh`: Detection confidence threshold (default: 0.6)
- `--track-buffer`: Frames to keep lost tracks (default: 30)
- `--match-thresh`: Matching threshold for tracking (default: 0.8)
- `--fp16`: Use FP16 precision (requires GPU)
- `--fuse`: Fuse conv and bn layers for faster inference
- `--save-result`: Save tracking results and videos (default: True)
- `--output-dir`: Output directory (default: `YOLOX_outputs/mot15_nano_test`)

## Output

### Directory Structure

After running, outputs are saved to:

```
YOLOX_outputs/
└── yolox_nano_mix_det/
    └── track_vis/
        └── YYYYMMDD_HHMMSS/
            ├── ADL-Rundle-1/
            │   └── track_results.txt
            ├── Venice-1/
            │   └── track_results.txt
            └── ...
```

### Tracking Results Format

Each sequence generates a `track_results.txt` file in MOTChallenge format:

```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

Example:
```
1, 1, 100.5, 200.3, 50.2, 120.8, 0.95, -1, -1, -1
1, 2, 300.1, 150.7, 48.9, 115.3, 0.87, -1, -1, -1
2, 1, 102.3, 201.5, 50.5, 121.2, 0.93, -1, -1, -1
```

Fields:
- `frame`: Frame number (1-indexed)
- `id`: Track ID
- `bb_left`: Bounding box left coordinate
- `bb_top`: Bounding box top coordinate
- `bb_width`: Bounding box width
- `bb_height`: Bounding box height
- `conf`: Detection confidence
- `x, y, z`: 3D coordinates (set to -1 for 2D tracking)

### Test Report

A detailed report is generated at:
```
YOLOX_outputs/mot15_nano_test/mot15_nano_test_report_YYYYMMDD_HHMMSS.txt
```

Report includes:
- Model configuration
- Tracking parameters
- Per-sequence results (frames, time, FPS)
- Summary statistics

## Performance Expectations

### Nano Model (CPU)

Typical processing times on CPU:

| Sequence Type | Frames | Expected Time | FPS |
|--------------|--------|---------------|-----|
| Short (200-500) | ~300 | 1-3 minutes | 2-5 |
| Medium (500-700) | ~600 | 3-5 minutes | 2-4 |
| Long (1000+) | ~1100 | 8-12 minutes | 1.5-3 |

**Total for all 11 sequences: ~45-60 minutes on CPU**

### Nano Model (GPU)

With GPU acceleration (--device gpu --fp16 --fuse):

| Sequence Type | Frames | Expected Time | FPS |
|--------------|--------|---------------|-----|
| Short | ~300 | 20-40 seconds | 10-15 |
| Medium | ~600 | 40-80 seconds | 8-12 |
| Long | ~1100 | 90-180 seconds | 6-12 |

**Total for all 11 sequences: ~10-20 minutes on GPU**

### Model Characteristics

**Nano Model:**
- Parameters: 0.90M
- FLOPs: 3.99G
- Speed: Fastest ⭐⭐⭐⭐⭐
- Accuracy: Good ⭐⭐⭐
- Use case: Real-time applications, resource-constrained devices

## Evaluation

### Submit to MOTChallenge

1. Organize tracking results into submission format:
```
<sequence_name>.txt
```

2. Create a ZIP file with all result files

3. Submit to [MOTChallenge MOT15](https://motchallenge.net/data/MOT15)

4. Get official metrics:
   - MOTA (Multiple Object Tracking Accuracy)
   - MOTP (Multiple Object Tracking Precision)
   - IDF1 (ID F1 Score)
   - MT (Mostly Tracked)
   - ML (Mostly Lost)
   - FP (False Positives)
   - FN (False Negatives)
   - IDs (ID Switches)

### Local Evaluation (with ground truth)

If you have ground truth annotations:

```bash
python tools/mota.py --gt datasets/MOT15/annotations/test.json \
                      --pred YOLOX_outputs/yolox_nano_mix_det/track_vis/
```

## Troubleshooting

### Issue: "MOT15 dataset not found"

**Solution:**
```bash
python setup_mot15.py
```

### Issue: "Model checkpoint not found"

**Solution:**
```bash
python download_model.py nano
```
Or manually place the checkpoint at `pretrained/bytetrack_nano_mot17.pth.tar`

### Issue: "No images found in sequence"

**Solution:** Check that sequences are properly extracted:
```bash
ls datasets/MOT15/test/Venice-1/img1/
```

### Issue: Out of memory on GPU

**Solution:** Use CPU instead:
```bash
python test_mot15_nano.py --device cpu
```

### Issue: Slow processing on CPU

**Solution:** 
1. Use GPU if available: `--device gpu --fp16 --fuse`
2. Test fewer sequences: `--sequences Venice-1 KITTI-16`
3. Use smaller sequences first

## Tips

### Fast Testing
Start with shorter sequences to verify setup:
```bash
python test_mot15_nano.py --sequences TUD-Crossing ETH-Crossing
```

### Quality vs Speed
For best tracking quality:
```bash
python test_mot15_nano.py --track-thresh 0.5 --match-thresh 0.9
```

For faster processing (lower quality):
```bash
python test_mot15_nano.py --track-thresh 0.7 --match-thresh 0.7
```

### GPU Optimization
For maximum GPU speed:
```bash
python test_mot15_nano.py --device gpu --fp16 --fuse
```

## Next Steps

After running the tests:

1. **Review Results**: Check the generated report and tracking files
2. **Visualize**: Generate visualization videos to inspect tracking quality
3. **Evaluate**: Submit results to MOTChallenge for official evaluation
4. **Optimize**: Adjust tracking parameters based on results
5. **Compare**: Test with other models (tiny, s, m, l, x) using `test_models.py`

## Related Scripts

- `setup_mot15.py` - Download and setup MOT15 dataset
- `tools/convert_mot15_to_coco.py` - Convert annotations to COCO format
- `tools/track_mot15.py` - Track with custom model
- `test_models.py` - Compare multiple models
- `tools/demo_track.py` - Interactive tracking demo

## References

- [MOT15 Challenge](https://motchallenge.net/data/MOT15)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review existing issues on GitHub
3. Create a new issue with:
   - Error message
   - Command used
   - System information (OS, Python version, GPU)
