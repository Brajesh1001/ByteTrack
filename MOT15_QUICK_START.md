# Quick Start: MOT15 Testing

## 🚀 Quick Run

### Windows
```batch
run_mot15_test_nano.bat
```

Or with PowerShell:
```powershell
.\run_mot15_test_nano.ps1
```

### Linux/Mac
```bash
python test_mot15_nano.py
```

## 📋 What This Does

Automatically tests the ByteTrack nano model on all 11 MOT15 test sequences:
- ADL-Rundle-1, ADL-Rundle-3
- AVG-TownCentre
- ETH-Crossing, ETH-Jelmoli, ETH-Linthescher
- KITTI-16, KITTI-19
- PETS09-S2L2
- TUD-Crossing
- Venice-1

**Total: 5,783 frames**

## ⏱️ Expected Time

- **CPU**: ~45-60 minutes for all sequences
- **GPU**: ~10-20 minutes for all sequences

## 📦 Prerequisites

1. **MOT15 Dataset** (download first if needed):
   ```bash
   python setup_mot15.py
   ```

2. **Nano Model** (should already be at `pretrained/bytetrack_nano_mot17.pth.tar`)

## 🎯 Quick Examples

### Test all sequences (default)
```bash
python test_mot15_nano.py
```

### Test specific sequences only
```bash
python test_mot15_nano.py --sequences Venice-1 KITTI-19
```

### Use GPU for faster processing
```bash
python test_mot15_nano.py --device gpu --fp16 --fuse
```

### Custom tracking parameters
```bash
python test_mot15_nano.py --track-thresh 0.5 --match-thresh 0.7
```

## 📊 Output

Results are saved to:
```
YOLOX_outputs/
└── yolox_nano_mix_det/
    └── track_vis/
        └── YYYYMMDD_HHMMSS/
            ├── ADL-Rundle-1/
            ├── Venice-1/
            └── ...
```

Each sequence folder contains:
- `track_results.txt` - Tracking results in MOTChallenge format
- Video visualizations (if enabled)

## 📖 Full Documentation

See [MOT15_TESTING_GUIDE.md](MOT15_TESTING_GUIDE.md) for complete documentation including:
- Detailed setup instructions
- All command-line options
- Performance benchmarks
- Troubleshooting guide
- Evaluation instructions

## 🔧 Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | cpu or gpu | cpu |
| `--sequences` | Specific sequences to test | All 11 |
| `--track-thresh` | Detection threshold | 0.6 |
| `--match-thresh` | Matching threshold | 0.8 |
| `--track-buffer` | Lost track buffer (frames) | 30 |
| `--fp16` | Use FP16 precision (GPU) | False |
| `--fuse` | Fuse conv+bn layers | False |

## 💡 Tips

**Start with a quick test** (2 short sequences):
```bash
python test_mot15_nano.py --sequences TUD-Crossing ETH-Crossing
```

**For maximum GPU speed**:
```bash
python test_mot15_nano.py --device gpu --fp16 --fuse
```

**For best tracking quality**:
```bash
python test_mot15_nano.py --track-thresh 0.5 --match-thresh 0.9
```

## ❓ Help

View all options:
```bash
python test_mot15_nano.py --help
```

## 📧 Issues?

Check [MOT15_TESTING_GUIDE.md](MOT15_TESTING_GUIDE.md) troubleshooting section.
