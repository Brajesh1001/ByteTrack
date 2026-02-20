#!/usr/bin/env bash
#
# Evaluate ALL ByteTrack model variants (nano, tiny, s, m, l, x) on MOT15
# and produce a unified comparison table.
#
# Usage:
#   chmod +x run_all_models_mot15.sh
#   ./run_all_models_mot15.sh                    # full pipeline
#   ./run_all_models_mot15.sh --skip-install     # skip dep installation
#   ./run_all_models_mot15.sh --skip-download    # skip dataset & weight downloads
#   ./run_all_models_mot15.sh --models nano,tiny # only run specific models
#
set -euo pipefail

BYTETRACK_HOME="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="${BYTETRACK_HOME}/datasets/MOT15"
PRETRAINED_DIR="${BYTETRACK_HOME}/pretrained"
OUTPUT_DIR="${BYTETRACK_HOME}/YOLOX_outputs"
RESULTS_MD="${BYTETRACK_HOME}/mot15_all_models_results.md"

SKIP_INSTALL=false
SKIP_DOWNLOAD=false
SELECTED_MODELS=""

for arg in "$@"; do
    case "$arg" in
        --skip-install)  SKIP_INSTALL=true ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
        --models=*)      SELECTED_MODELS="${arg#*=}" ;;
    esac
done

# ---- Model registry ----
# Format: name|exp_file|ckpt_filename|gdrive_id
MODELS=(
    "nano|exps/example/mot/yolox_nano_mot15.py|bytetrack_nano_mot17.pth.tar|1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX"
    "tiny|exps/example/mot/yolox_tiny_mot15.py|bytetrack_tiny_mot17.pth.tar|1LFAl14sql2Q5Y9aNFsX_OqsnIzUD_1ju"
    "s|exps/example/mot/yolox_s_mot15.py|bytetrack_s_mot17.pth.tar|1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj"
    "m|exps/example/mot/yolox_m_mot15.py|bytetrack_m_mot17.pth.tar|11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun"
    "l|exps/example/mot/yolox_l_mot15.py|bytetrack_l_mot17.pth.tar|1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz"
    "x|exps/example/mot/yolox_x_mot15.py|bytetrack_x_mot17.pth.tar|1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5"
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log()  { echo -e "\n\033[1;32m>>>\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $*"; exit 1; }

should_run_model() {
    local name="$1"
    if [ -z "${SELECTED_MODELS}" ]; then
        return 0
    fi
    echo ",${SELECTED_MODELS}," | grep -q ",${name},"
}

check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        fail "nvidia-smi not found. A CUDA GPU is required."
    fi
    echo "GPU(s) detected:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
}

# --------------------------------------------------------------------------- #
# Step 1 -- Install dependencies
# --------------------------------------------------------------------------- #
install_deps() {
    log "Installing dependencies"
    pip3 install --quiet -r "${BYTETRACK_HOME}/requirements.txt"
    cd "${BYTETRACK_HOME}" && pip3 install --quiet -e .
    pip3 install --quiet cython
    pip3 install --quiet 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    pip3 install --quiet cython_bbox gdown
    echo "Done."
}

# --------------------------------------------------------------------------- #
# Step 2 -- Download MOT15 dataset
# --------------------------------------------------------------------------- #
download_mot15() {
    log "Downloading MOT15 dataset"
    if [ -d "${DATASET_DIR}/train" ] && [ -d "${DATASET_DIR}/test" ]; then
        echo "Already exists, skipping."
        return
    fi
    mkdir -p "${DATASET_DIR}" && cd "${DATASET_DIR}"
    [ -f "MOT15.zip" ] || wget --no-check-certificate -q --show-progress \
        "https://motchallenge.net/data/MOT15.zip" -O MOT15.zip
    unzip -q -o MOT15.zip
    if [ -d "${DATASET_DIR}/MOT15/train" ] && [ ! -d "${DATASET_DIR}/train" ]; then
        mv "${DATASET_DIR}/MOT15/train" "${DATASET_DIR}/train"
        mv "${DATASET_DIR}/MOT15/test"  "${DATASET_DIR}/test"
        rm -rf "${DATASET_DIR}/MOT15"
    fi
    cd "${BYTETRACK_HOME}"
    echo "Done."
}

# --------------------------------------------------------------------------- #
# Step 3 -- Convert MOT15 to COCO format
# --------------------------------------------------------------------------- #
convert_to_coco() {
    log "Converting MOT15 to COCO format"
    if [ -f "${DATASET_DIR}/annotations/train.json" ]; then
        echo "Annotations already exist, skipping."
        return
    fi
    cd "${BYTETRACK_HOME}"
    python3 tools/convert_mot15_to_coco.py
    echo "Done."
}

# --------------------------------------------------------------------------- #
# Step 4 -- Download pretrained weights
# --------------------------------------------------------------------------- #
download_weights() {
    log "Downloading pretrained model weights"
    mkdir -p "${PRETRAINED_DIR}"

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name exp ckpt gdrive_id <<< "${entry}"
        if ! should_run_model "${name}"; then
            continue
        fi
        local ckpt_path="${PRETRAINED_DIR}/${ckpt}"
        if [ -f "${ckpt_path}" ]; then
            echo "  [${name}] ${ckpt} already exists"
        else
            echo "  [${name}] downloading ${ckpt} ..."
            gdown "https://drive.google.com/uc?id=${gdrive_id}" -O "${ckpt_path}" \
                || warn "Failed to download ${ckpt}. Download manually from Google Drive id=${gdrive_id}"
        fi
    done
    echo "Done."
}

# --------------------------------------------------------------------------- #
# Step 5 -- Run evaluation for each model
# --------------------------------------------------------------------------- #
run_evaluations() {
    log "Running MOT15 evaluation for all selected models"
    cd "${BYTETRACK_HOME}"

    local passed=0
    local failed=0

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name exp ckpt gdrive_id <<< "${entry}"
        if ! should_run_model "${name}"; then
            continue
        fi

        local ckpt_path="${PRETRAINED_DIR}/${ckpt}"
        if [ ! -f "${ckpt_path}" ]; then
            warn "[${name}] Checkpoint not found at ${ckpt_path}, skipping."
            failed=$((failed + 1))
            continue
        fi

        local exp_name
        exp_name="$(basename "${exp}" .py)"

        echo ""
        log "Evaluating: ${name} (${exp_name})"
        echo "  Exp:  ${exp}"
        echo "  Ckpt: ${ckpt_path}"
        echo ""

        if python3 tools/evaluate_mot15_nano.py \
                -f "${exp}" \
                -c "${ckpt_path}" \
                -b 1 -d 1 --fp16 --fuse \
                2>&1 | tee "${OUTPUT_DIR}/${exp_name}.log"; then
            echo ""
            echo "  [${name}] DONE"
            passed=$((passed + 1))
        else
            warn "[${name}] evaluation failed -- check ${OUTPUT_DIR}/${exp_name}.log"
            failed=$((failed + 1))
        fi
    done

    echo ""
    echo "Evaluation summary: ${passed} passed, ${failed} failed"
}

# --------------------------------------------------------------------------- #
# Step 6 -- Generate comparison table
# --------------------------------------------------------------------------- #
generate_table() {
    log "Generating comparison table"
    cd "${BYTETRACK_HOME}"
    python3 tools/generate_mot15_comparison_table.py \
        --log-dir "${OUTPUT_DIR}" \
        --output "${RESULTS_MD}"
    echo ""
    echo "Results written to: ${RESULTS_MD}"
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
echo "======================================================="
echo " ByteTrack ALL Models -- MOT15 Evaluation Pipeline"
echo "======================================================="
echo ""

if [ -n "${SELECTED_MODELS}" ]; then
    echo "Selected models: ${SELECTED_MODELS}"
else
    echo "Running ALL models: nano, tiny, s, m, l, x"
fi
echo ""

check_gpu

if [ "${SKIP_INSTALL}" = false ]; then
    install_deps
else
    warn "Skipping dependency installation (--skip-install)"
fi

if [ "${SKIP_DOWNLOAD}" = false ]; then
    download_mot15
    download_weights
else
    warn "Skipping downloads (--skip-download)"
fi

convert_to_coco
run_evaluations
generate_table

log "All done!"
echo ""
echo "  Individual logs : ${OUTPUT_DIR}/yolox_*_mot15.log"
echo "  Comparison table: ${RESULTS_MD}"
echo ""
cat "${RESULTS_MD}"
