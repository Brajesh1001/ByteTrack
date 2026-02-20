"""
Evaluate ByteTrack models (nano, tiny, s, m, l, x) on MOT15 and build a
comparison table.  Works on Windows, Linux and macOS -- no bash required.

Usage (from the ByteTrack root directory):

    python run_all_models_mot15.py                          # full pipeline, all models
    python run_all_models_mot15.py --models nano,tiny,s     # only specific models
    python run_all_models_mot15.py --skip-install           # skip pip installs
    python run_all_models_mot15.py --skip-download          # skip dataset & weight downloads
"""

import argparse
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "nano": {
        "exp": "exps/example/mot/yolox_nano_mot15.py",
        "ckpt": "bytetrack_nano_mot17.pth.tar",
        "gdrive_id": "1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX",
        "label": "ByteTrack-Nano",
        "params": "0.90",
        "flops": "3.99",
        "test_size": "608x1088",
    },
    "tiny": {
        "exp": "exps/example/mot/yolox_tiny_mot15.py",
        "ckpt": "bytetrack_tiny_mot17.pth.tar",
        "gdrive_id": "1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju",
        "label": "ByteTrack-Tiny",
        "params": "5.03",
        "flops": "24.45",
        "test_size": "608x1088",
    },
    "s": {
        "exp": "exps/example/mot/yolox_s_mot15.py",
        "ckpt": "bytetrack_s_mot17.pth.tar",
        "gdrive_id": "1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj",
        "label": "ByteTrack-S",
        "params": "9.0",
        "flops": "26.8",
        "test_size": "608x1088",
    },
    "m": {
        "exp": "exps/example/mot/yolox_m_mot15.py",
        "ckpt": "bytetrack_m_mot17.pth.tar",
        "gdrive_id": "11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun",
        "label": "ByteTrack-M",
        "params": "25.3",
        "flops": "73.8",
        "test_size": "800x1440",
    },
    "l": {
        "exp": "exps/example/mot/yolox_l_mot15.py",
        "ckpt": "bytetrack_l_mot17.pth.tar",
        "gdrive_id": "1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz",
        "label": "ByteTrack-L",
        "params": "54.2",
        "flops": "155.6",
        "test_size": "800x1440",
    },
    "x": {
        "exp": "exps/example/mot/yolox_x_mot15.py",
        "ckpt": "bytetrack_x_mot17.pth.tar",
        "gdrive_id": "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
        "label": "ByteTrack-X",
        "params": "99.1",
        "flops": "281.9",
        "test_size": "800x1440",
    },
}

MODEL_ORDER = ["nano", "tiny", "s", "m", "l", "x"]

MOTCHALLENGE_COLS = [
    "IDF1", "IDP", "IDR", "Rcll", "Prcn",
    "GT", "MT", "PT", "ML",
    "FP", "FN", "IDs", "FM",
    "MOTA", "MOTP",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets" / "MOT15"
PRETRAINED_DIR = ROOT / "pretrained"
OUTPUT_DIR = ROOT / "YOLOX_outputs"


def run(cmd, **kwargs):
    """Run a shell command, streaming output in real time."""
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"[WARN] Command exited with code {result.returncode}")
    return result.returncode


def pip_install(*packages):
    run(f"{sys.executable} -m pip install --quiet " + " ".join(packages))


# ---------------------------------------------------------------------------
# Step 1 -- Install dependencies
# ---------------------------------------------------------------------------
def install_deps():
    print("\n" + "=" * 60)
    print("STEP 1 / 6 : Installing dependencies")
    print("=" * 60)

    pip_install("-r", str(ROOT / "requirements.txt"))

    run(f"{sys.executable} -m pip install --quiet -e .", cwd=str(ROOT))

    pip_install("cython")
    pip_install("git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI")

    if os.name == "nt":
        print("[INFO] On Windows, cython_bbox may need Visual C++ Build Tools.")
        print("       If it fails, install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    pip_install("cython_bbox")

    pip_install("gdown")

    print("Dependencies installed.\n")


# ---------------------------------------------------------------------------
# Step 2 -- Download MOT15
# ---------------------------------------------------------------------------
def download_mot15():
    print("\n" + "=" * 60)
    print("STEP 2 / 6 : Downloading MOT15 dataset")
    print("=" * 60)

    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"

    if train_dir.is_dir() and test_dir.is_dir():
        print("MOT15 already exists, skipping download.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR / "MOT15.zip"

    if not zip_path.is_file():
        import urllib.request
        url = "https://motchallenge.net/data/MOT15.zip"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, str(zip_path))

    print("Extracting MOT15.zip ...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(DATASET_DIR))

    nested = DATASET_DIR / "MOT15"
    if nested.is_dir() and not train_dir.is_dir():
        (nested / "train").rename(train_dir)
        (nested / "test").rename(test_dir)
        import shutil
        shutil.rmtree(str(nested), ignore_errors=True)

    print("MOT15 ready.\n")


# ---------------------------------------------------------------------------
# Step 3 -- Convert to COCO format
# ---------------------------------------------------------------------------
def convert_to_coco():
    print("\n" + "=" * 60)
    print("STEP 3 / 6 : Converting MOT15 to COCO format")
    print("=" * 60)

    ann_file = DATASET_DIR / "annotations" / "train.json"
    if ann_file.is_file():
        print("Annotations already exist, skipping.")
        return

    run(f"{sys.executable} tools/convert_mot15_to_coco.py", cwd=str(ROOT))
    print("Conversion done.\n")


# ---------------------------------------------------------------------------
# Step 4 -- Download pretrained weights
# ---------------------------------------------------------------------------
def download_weights(selected):
    print("\n" + "=" * 60)
    print("STEP 4 / 6 : Downloading pretrained model weights")
    print("=" * 60)

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    for name in selected:
        info = MODELS[name]
        ckpt_path = PRETRAINED_DIR / info["ckpt"]

        if ckpt_path.is_file():
            print(f"  [{name}] {info['ckpt']} already exists")
            continue

        print(f"  [{name}] downloading {info['ckpt']} ...")
        try:
            import gdown
            gdown.download(
                f"https://drive.google.com/uc?id={info['gdrive_id']}",
                str(ckpt_path),
                quiet=False,
            )
        except Exception as e:
            print(f"  [WARN] Download failed for {name}: {e}")
            print(f"         Download manually from Google Drive id={info['gdrive_id']}")
            print(f"         Save to: {ckpt_path}")

    print("Weights ready.\n")


# ---------------------------------------------------------------------------
# Step 5 -- Run evaluation for each model
# ---------------------------------------------------------------------------
def run_evaluations(selected):
    print("\n" + "=" * 60)
    print("STEP 5 / 6 : Running MOT15 evaluations")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name in selected:
        info = MODELS[name]
        ckpt_path = PRETRAINED_DIR / info["ckpt"]
        exp_path = ROOT / info["exp"]
        exp_name = Path(info["exp"]).stem
        log_path = OUTPUT_DIR / f"{exp_name}.log"

        if not ckpt_path.is_file():
            print(f"\n  [{name}] Checkpoint not found: {ckpt_path}  -- SKIPPING")
            results[name] = False
            continue

        print(f"\n{'─' * 60}")
        print(f"  Evaluating: {info['label']}  ({name})")
        print(f"  Exp file : {exp_path}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Log file : {log_path}")
        print(f"{'─' * 60}\n")

        cmd = (
            f"{sys.executable} tools/evaluate_mot15_nano.py"
            f" -f {exp_path}"
            f" -c {ckpt_path}"
            f" -b 1 -d 1 --fp16 --fuse"
        )

        with open(str(log_path), "w") as lf:
            proc = subprocess.Popen(
                cmd, shell=True, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            proc.wait()

        results[name] = proc.returncode == 0
        status = "PASS" if results[name] else "FAIL"
        print(f"\n  [{name}] {status}  (log: {log_path})")

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    print(f"\nEvaluation summary: {passed} passed, {failed} failed\n")
    return results


# ---------------------------------------------------------------------------
# Step 6 -- Parse logs and build comparison table
# ---------------------------------------------------------------------------
def parse_overall_row(log_text):
    for line in log_text.splitlines():
        if "OVERALL" not in line:
            continue
        cols = line.split()
        try:
            idx = cols.index("OVERALL")
        except ValueError:
            continue
        vals = cols[idx + 1:]
        if len(vals) < 15:
            continue
        return {col: vals[i] for i, col in enumerate(MOTCHALLENGE_COLS)}
    return {}


def parse_per_sequence(log_text):
    rows = []
    in_table = False
    for line in log_text.splitlines():
        stripped = line.strip()
        if "IDF1" in stripped and "MOTA" in stripped and "MOTP" in stripped:
            in_table = True
            continue
        if in_table:
            if stripped == "":
                break
            cols = stripped.split()
            if len(cols) >= 16:
                row = {"sequence": cols[0]}
                for i, col in enumerate(MOTCHALLENGE_COLS):
                    if i < len(cols) - 1:
                        row[col] = cols[i + 1]
                rows.append(row)
    return rows


def parse_timing(log_text):
    timing = {}
    for line in log_text.splitlines():
        m = re.search(r"Average forward time:\s*([\d.]+)\s*ms", line)
        if m:
            timing["forward_ms"] = m.group(1)
        m = re.search(r"Average track time:\s*([\d.]+)\s*ms", line)
        if m:
            timing["track_ms"] = m.group(1)
        m = re.search(r"Average inference time:\s*([\d.]+)\s*ms", line)
        if m:
            timing["inference_ms"] = m.group(1)
    return timing


def generate_table(selected):
    print("\n" + "=" * 60)
    print("STEP 6 / 6 : Generating comparison table")
    print("=" * 60)

    all_data = {}
    for name in selected:
        exp_name = Path(MODELS[name]["exp"]).stem
        for candidate in [
            OUTPUT_DIR / exp_name / "val_log.txt",
            OUTPUT_DIR / f"{exp_name}.log",
        ]:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="replace")
                all_data[name] = {
                    "overall": parse_overall_row(text),
                    "sequences": parse_per_sequence(text),
                    "timing": parse_timing(text),
                }
                break

    # ---- Build markdown ----
    lines = ["# ByteTrack All Models -- MOT15 Comparison Table\n"]

    # Main table
    lines.append("## Overall Comparison\n")
    hdr = (
        "| Model | Params(M) | FLOPs(G) | Test Size "
        "| MOTA | IDF1 | MOTP | Rcll | Prcn "
        "| MT | ML | FP | FN | IDs | FM |"
    )
    sep = (
        "|-------|-----------|----------|-----------|"
        "------|------|------|------|------|"
        "----|----|----|----|----|-----|"
    )
    lines.extend([hdr, sep])

    for name in MODEL_ORDER:
        m = MODELS[name]
        r = all_data.get(name, {}).get("overall", {})
        if not r:
            vals = " | ".join(["--"] * 11)
            lines.append(
                f"| {m['label']} | {m['params']} | {m['flops']} "
                f"| {m['test_size']} | {vals} |"
            )
        else:
            lines.append(
                f"| **{m['label']}** | {m['params']} | {m['flops']} "
                f"| {m['test_size']} "
                f"| **{r.get('MOTA','--')}** | **{r.get('IDF1','--')}** "
                f"| {r.get('MOTP','--')} "
                f"| {r.get('Rcll','--')} | {r.get('Prcn','--')} "
                f"| {r.get('MT','--')} | {r.get('ML','--')} "
                f"| {r.get('FP','--')} | {r.get('FN','--')} "
                f"| {r.get('IDs','--')} | {r.get('FM','--')} |"
            )
    lines.append("")

    # Timing table
    has_timing = any(
        all_data.get(n, {}).get("timing") for n in MODEL_ORDER
    )
    if has_timing:
        lines.append("## Inference Speed\n")
        lines.append("| Model | Forward (ms) | Track (ms) | Total (ms) | ~FPS |")
        lines.append("|-------|-------------|------------|------------|------|")
        for name in MODEL_ORDER:
            m = MODELS[name]
            t = all_data.get(name, {}).get("timing", {})
            if t:
                total = float(t.get("inference_ms", 0))
                fps = f"{1000.0 / total:.1f}" if total > 0 else "--"
                lines.append(
                    f"| {m['label']} | {t.get('forward_ms','--')} "
                    f"| {t.get('track_ms','--')} "
                    f"| {t.get('inference_ms','--')} | {fps} |"
                )
            else:
                lines.append(f"| {m['label']} | -- | -- | -- | -- |")
        lines.append("")

    # Per-sequence detail
    for name in MODEL_ORDER:
        m = MODELS[name]
        seqs = all_data.get(name, {}).get("sequences", [])
        if not seqs:
            continue
        lines.append(f"## {m['label']} -- Per-Sequence Results\n")
        lines.append(
            "| Sequence | MOTA | IDF1 | MOTP | Rcll | Prcn "
            "| MT | ML | FP | FN | IDs | FM |"
        )
        lines.append(
            "|----------|------|------|------|------|------|"
            "----|----|----|----|----|-----|"
        )
        for row in seqs:
            lines.append(
                f"| {row['sequence']} "
                f"| {row.get('MOTA','--')} | {row.get('IDF1','--')} "
                f"| {row.get('MOTP','--')} "
                f"| {row.get('Rcll','--')} | {row.get('Prcn','--')} "
                f"| {row.get('MT','--')} | {row.get('ML','--')} "
                f"| {row.get('FP','--')} | {row.get('FN','--')} "
                f"| {row.get('IDs','--')} | {row.get('FM','--')} |"
            )
        lines.append("")

    md = "\n".join(lines) + "\n"
    out_path = ROOT / "mot15_all_models_results.md"
    out_path.write_text(md, encoding="utf-8")

    print(f"\nResults written to: {out_path}\n")
    print(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all ByteTrack models on MOT15 (Windows/Linux/macOS)"
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated list of models to run (default: all). "
             "Choices: nano,tiny,s,m,l,x"
    )
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip dependency installation")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset and weight downloads")
    args = parser.parse_args()

    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        for m in selected:
            if m not in MODELS:
                print(f"Unknown model '{m}'. Choose from: {', '.join(MODEL_ORDER)}")
                sys.exit(1)
    else:
        selected = list(MODEL_ORDER)

    print("=" * 60)
    print(" ByteTrack ALL Models -- MOT15 Evaluation Pipeline")
    print("=" * 60)
    print(f"\n  Platform : {sys.platform}")
    print(f"  Python   : {sys.executable}")
    print(f"  Models   : {', '.join(selected)}")
    print(f"  Root     : {ROOT}\n")

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  [WARN] No CUDA GPU detected. Evaluation requires a GPU.")
            print("         Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/")
    except ImportError:
        print("  [WARN] PyTorch not installed yet (will be installed in Step 1)")

    if not args.skip_install:
        install_deps()
    else:
        print("\n[SKIP] Dependency installation (--skip-install)\n")

    if not args.skip_download:
        download_mot15()
        download_weights(selected)
    else:
        print("\n[SKIP] Downloads (--skip-download)\n")

    convert_to_coco()
    run_evaluations(selected)
    generate_table(selected)

    print("=" * 60)
    print(" ALL DONE!")
    print("=" * 60)
    print(f"\n  Logs  : YOLOX_outputs/yolox_*_mot15.log")
    print(f"  Table : mot15_all_models_results.md\n")


if __name__ == "__main__":
    main()
