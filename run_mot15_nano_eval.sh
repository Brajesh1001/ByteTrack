#!/usr/bin/env bash
#
# End-to-end script: evaluate ByteTrack Nano model on MOT15 dataset.
#
# Usage:
#   chmod +x run_mot15_nano_eval.sh
#   ./run_mot15_nano_eval.sh            # full pipeline (install + download + eval)
#   ./run_mot15_nano_eval.sh --skip-install   # skip dependency installation
#   ./run_mot15_nano_eval.sh --skip-download  # skip dataset & model download
#
set -euo pipefail

BYTETRACK_HOME="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="${BYTETRACK_HOME}/datasets/MOT15"
PRETRAINED_DIR="${BYTETRACK_HOME}/pretrained"
CKPT_PATH="${PRETRAINED_DIR}/bytetrack_nano_mot17.pth.tar"
EXP_FILE="${BYTETRACK_HOME}/exps/example/mot/yolox_nano_mot15.py"
EVAL_SCRIPT="${BYTETRACK_HOME}/tools/evaluate_mot15_nano.py"
RESULTS_MD="${BYTETRACK_HOME}/mot15_nano_results.md"
GDRIVE_FILE_ID="1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX"

SKIP_INSTALL=false
SKIP_DOWNLOAD=false
for arg in "$@"; do
    case "$arg" in
        --skip-install)  SKIP_INSTALL=true ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
    esac
done

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log()  { echo -e "\n\033[1;32m[STEP]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*"; exit 1; }

check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        fail "nvidia-smi not found. A CUDA GPU is required for evaluation."
    fi
    if ! nvidia-smi &>/dev/null; then
        fail "nvidia-smi failed. Make sure NVIDIA drivers are installed."
    fi
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

# --------------------------------------------------------------------------- #
# Step 1 -- Install dependencies
# --------------------------------------------------------------------------- #
install_deps() {
    log "Step 1/6: Installing dependencies"

    pip3 install --quiet -r "${BYTETRACK_HOME}/requirements.txt"

    cd "${BYTETRACK_HOME}"
    pip3 install --quiet -e .

    pip3 install --quiet cython
    pip3 install --quiet 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    pip3 install --quiet cython_bbox
    pip3 install --quiet gdown

    echo "Dependencies installed."
}

# --------------------------------------------------------------------------- #
# Step 2 -- Download MOT15 dataset
# --------------------------------------------------------------------------- #
download_mot15() {
    log "Step 2/6: Downloading MOT15 dataset"

    if [ -d "${DATASET_DIR}/train" ] && [ -d "${DATASET_DIR}/test" ]; then
        echo "MOT15 dataset already exists at ${DATASET_DIR}, skipping download."
        return
    fi

    mkdir -p "${DATASET_DIR}"
    cd "${DATASET_DIR}"

    if [ ! -f "MOT15.zip" ]; then
        wget --no-check-certificate -q --show-progress \
             "https://motchallenge.net/data/MOT15.zip" -O MOT15.zip \
        || fail "Failed to download MOT15. Download manually from https://motchallenge.net/data/MOT15/"
    fi

    unzip -q -o MOT15.zip
    # The zip may extract into a subdirectory; normalise into train/ and test/
    if [ -d "${DATASET_DIR}/MOT15/train" ] && [ ! -d "${DATASET_DIR}/train" ]; then
        mv "${DATASET_DIR}/MOT15/train" "${DATASET_DIR}/train"
        mv "${DATASET_DIR}/MOT15/test"  "${DATASET_DIR}/test"
        rm -rf "${DATASET_DIR}/MOT15"
    fi

    cd "${BYTETRACK_HOME}"
    echo "MOT15 dataset ready at ${DATASET_DIR}"
}

# --------------------------------------------------------------------------- #
# Step 3 -- Convert MOT15 to COCO format
# --------------------------------------------------------------------------- #
convert_to_coco() {
    log "Step 3/6: Converting MOT15 annotations to COCO format"

    if [ -f "${DATASET_DIR}/annotations/train.json" ]; then
        echo "COCO annotations already exist, skipping conversion."
        return
    fi

    cd "${BYTETRACK_HOME}"
    python3 tools/convert_mot15_to_coco.py
    echo "COCO annotations written to ${DATASET_DIR}/annotations/"
}

# --------------------------------------------------------------------------- #
# Step 4 -- Download pretrained nano model
# --------------------------------------------------------------------------- #
download_model() {
    log "Step 4/6: Downloading ByteTrack Nano pretrained weights"

    if [ -f "${CKPT_PATH}" ]; then
        echo "Checkpoint already exists at ${CKPT_PATH}, skipping download."
        return
    fi

    mkdir -p "${PRETRAINED_DIR}"

    if command -v gdown &>/dev/null; then
        gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" \
              -O "${CKPT_PATH}" \
        || fail "gdown failed. Download the model manually:\n  URL: https://drive.google.com/file/d/${GDRIVE_FILE_ID}/view\n  Save to: ${CKPT_PATH}"
    else
        fail "gdown is not installed. Install with 'pip install gdown' or download manually:\n  URL: https://drive.google.com/file/d/${GDRIVE_FILE_ID}/view\n  Save to: ${CKPT_PATH}"
    fi

    echo "Model checkpoint saved to ${CKPT_PATH}"
}

# --------------------------------------------------------------------------- #
# Step 5 -- Run tracking + MOT metrics evaluation
# --------------------------------------------------------------------------- #
run_evaluation() {
    log "Step 5/6: Running ByteTrack Nano evaluation on MOT15 train set"

    cd "${BYTETRACK_HOME}"
    python3 "${EVAL_SCRIPT}" \
        -f "${EXP_FILE}" \
        -c "${CKPT_PATH}" \
        -b 1 -d 1 --fp16 --fuse \
        2>&1 | tee "${BYTETRACK_HOME}/mot15_nano_eval.log"

    echo "Evaluation complete. Full log: mot15_nano_eval.log"
}

# --------------------------------------------------------------------------- #
# Step 6 -- Parse results and generate markdown table
# --------------------------------------------------------------------------- #
generate_results_table() {
    log "Step 6/6: Generating results table"

    python3 - "${BYTETRACK_HOME}/mot15_nano_eval.log" "${RESULTS_MD}" <<'PYEOF'
import sys, re, os

log_path = sys.argv[1]
out_path = sys.argv[2]

if not os.path.isfile(log_path):
    print(f"Log file not found: {log_path}")
    sys.exit(1)

with open(log_path) as f:
    log_text = f.read()

# Locate the final motmetrics summary table (the second one printed by
# evaluate_mot15_nano.py -- the standard motchallenge_metrics table).
# It looks like a whitespace-aligned text table with a header row containing
# IDF1, IDP, IDR, Rcll, Prcn, ... MOTA, MOTP  and per-sequence + OVERALL rows.

table_blocks = []
current_block = []
in_table = False
for line in log_text.splitlines():
    stripped = line.strip()
    if "IDF1" in stripped and "MOTA" in stripped:
        in_table = True
        current_block = [line]
        continue
    if in_table:
        if stripped == "" or stripped.startswith("2") and "INFO" in stripped:
            if current_block:
                table_blocks.append(current_block)
            current_block = []
            in_table = False
        else:
            current_block.append(line)
if current_block:
    table_blocks.append(current_block)

md_lines = [
    "# ByteTrack Nano -- MOT15 Evaluation Results\n",
    "## Raw Metrics Table (motchallenge format)\n",
    "```",
]

if table_blocks:
    for line in table_blocks[-1]:
        md_lines.append(line)
else:
    md_lines.append("(no motchallenge table found in log -- check mot15_nano_eval.log)")

md_lines.append("```\n")

# Also try to grab the normalized summary (the first table with Rcll Prcn ...)
norm_blocks = []
current_block = []
in_table = False
for line in log_text.splitlines():
    stripped = line.strip()
    if "Rcll" in stripped and "Prcn" in stripped and "MOTA" in stripped and "GT" not in stripped:
        in_table = True
        current_block = [line]
        continue
    if in_table:
        if stripped == "" or stripped.startswith("2") and "INFO" in stripped:
            if current_block:
                norm_blocks.append(current_block)
            current_block = []
            in_table = False
        else:
            current_block.append(line)
if current_block:
    norm_blocks.append(current_block)

if norm_blocks:
    md_lines.append("## Normalized Metrics Table\n")
    md_lines.append("```")
    for line in norm_blocks[-1]:
        md_lines.append(line)
    md_lines.append("```\n")

# Parse OVERALL row for the summary comparison table
overall_mota = "--"
overall_idf1 = "--"
overall_ids  = "--"
for block in table_blocks:
    for line in block:
        if "OVERALL" in line:
            cols = line.split()
            # Standard motchallenge header: IDF1 IDP IDR Rcll Prcn GT MT PT ML FP FN IDs FM MOTA MOTP num_objects
            # OVERALL is the row label, then the values follow
            idx = cols.index("OVERALL") if "OVERALL" in cols else -1
            if idx >= 0:
                vals = cols[idx+1:]
                if len(vals) >= 15:
                    overall_idf1 = vals[0]   # IDF1
                    overall_ids  = vals[11]  # IDs
                    overall_mota = vals[13]  # MOTA

md_lines.extend([
    "## Summary: ByteTrack Nano Cross-Dataset Comparison\n",
    "| Dataset | MOTA | IDF1 | IDs | Params(M) | FLOPs(G) |",
    "|---------|------|------|-----|-----------|----------|",
    f"| MOT17   | 69.0 | 66.3 | 531 | 0.90      | 3.99     |",
    f"| **MOT15**   | **{overall_mota}** | **{overall_idf1}** | **{overall_ids}** | **0.90**      | **3.99**     |",
    "",
])

with open(out_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")

print(f"Results table written to {out_path}")
PYEOF
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
echo "=============================================="
echo " ByteTrack Nano -- MOT15 Evaluation Pipeline"
echo "=============================================="

check_gpu

if [ "${SKIP_INSTALL}" = false ]; then
    install_deps
else
    warn "Skipping dependency installation (--skip-install)"
fi

if [ "${SKIP_DOWNLOAD}" = false ]; then
    download_mot15
    download_model
else
    warn "Skipping downloads (--skip-download)"
fi

convert_to_coco
run_evaluation
generate_results_table

log "All done!"
echo ""
echo "  Evaluation log : ${BYTETRACK_HOME}/mot15_nano_eval.log"
echo "  Results table  : ${RESULTS_MD}"
echo ""
cat "${RESULTS_MD}"
