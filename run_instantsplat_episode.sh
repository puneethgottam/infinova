#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# User configuration (optional)
#
# Fill these in if you'd rather not pass them on the command line every time.
# Anything passed as a CLI argument or --cameras flag overrides these defaults.
# Leave a value empty ("") to require it on the command line.
# -----------------------------------------------------------------------------
DEFAULT_INPUT_FOLDER=""
DEFAULT_OUTPUT_FOLDER=""
DEFAULT_PYTHON_PATH=""
DEFAULT_START_FRAME=""
DEFAULT_END_FRAME=""
DEFAULT_CAMERAS=()   # e.g. (cam0 cam1 cam2 cam3 cam4), must be exactly 5 if set
# -----------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  run_instantsplat_episode.sh [INPUT_FOLDER OUTPUT_FOLDER PYTHON_PATH START_FRAME END_FRAME] [--cameras CAM1 CAM2 CAM3 CAM4 CAM5] [options]

Runs InstantSplat on a range of frames from a multi-camera episode folder.
Input images are expected to already be rectified; no rectification is performed.

All positional arguments and --cameras are optional if the corresponding
DEFAULT_* variable near the top of this script has been filled in. Any value
given on the command line overrides the default in the script.

Arguments:
  INPUT_FOLDER   Episode folder containing one subfolder per camera, each with
                 frame_######.png images (e.g. INPUT_FOLDER/<camera_name>/frame_000010.png)
  OUTPUT_FOLDER  Folder to write results to. For each frame, the Gaussian PLY and the
                 cameras.json produced during splatting are saved under
                 OUTPUT_FOLDER/instantsplat_gaussians
                 Pass "" (empty string) to use INPUT_FOLDER as the output folder.
  PYTHON_PATH    Path to the python interpreter with InstantSplatPP's dependencies installed
  START_FRAME    First frame number to process (inclusive)
  END_FRAME      Last frame number to process (inclusive)
  --cameras      Exactly 5 camera subfolder names under INPUT_FOLDER, in the order
                 the images should be fed to InstantSplat

Examples:
  run_instantsplat_episode.sh ./data/episode_000007 ./out/episode_000007 \
      /usr/bin/python3 10 20 --cameras cam0 cam1 cam2 cam3 cam4

  # With DEFAULT_* variables filled in at the top of the script:
  run_instantsplat_episode.sh

Environment overrides:
  WORK_ROOT=./instantsplat_reusable   Local scratch folder used to stage images and
                                       run training for each frame (reused/overwritten per frame)
  INSTANTSPLAT_DIR=./InstantSplatPP   Path to the InstantSplatPP repo (where init_geo.py / train.py live)
  GPU_ID=0
  N_VIEWS=5
  TRAIN_ITERATIONS=1000
  RESOLUTION=1
  MODEL_TYPE=                         Optional --model_type value forwarded to init_geo.py
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

INPUT_FOLDER="${DEFAULT_INPUT_FOLDER}"
OUTPUT_FOLDER="${DEFAULT_OUTPUT_FOLDER}"
PYTHON_PATH="${DEFAULT_PYTHON_PATH}"
START_FRAME="${DEFAULT_START_FRAME}"
END_FRAME="${DEFAULT_END_FRAME}"
CAMERAS=("${DEFAULT_CAMERAS[@]+"${DEFAULT_CAMERAS[@]}"}")

if [[ "$#" -gt 0 && "$1" != --* ]]; then
    if [[ "$#" -lt 5 ]]; then
        echo "Expected INPUT_FOLDER OUTPUT_FOLDER PYTHON_PATH START_FRAME END_FRAME." >&2
        usage >&2
        exit 2
    fi
    INPUT_FOLDER="$1"
    OUTPUT_FOLDER="$2"
    PYTHON_PATH="$3"
    START_FRAME="$4"
    END_FRAME="$5"
    shift 5
fi

if [[ "${1:-}" == "--cameras" ]]; then
    shift
    CAMERAS=()
    while [[ "$#" -gt 0 && "$1" != --* ]]; do
        CAMERAS+=("$1")
        shift
    done
fi

if [[ -z "${INPUT_FOLDER}" || -z "${PYTHON_PATH}" || -z "${START_FRAME}" || -z "${END_FRAME}" ]]; then
    echo "Missing INPUT_FOLDER/PYTHON_PATH/START_FRAME/END_FRAME: pass them as arguments or set the DEFAULT_* variables at the top of this script." >&2
    usage >&2
    exit 2
fi

OUTPUT_FOLDER="${OUTPUT_FOLDER:-${INPUT_FOLDER}}"

if [[ "${#CAMERAS[@]}" -ne 5 ]]; then
    echo "Expected exactly 5 camera names (via --cameras or DEFAULT_CAMERAS), got ${#CAMERAS[@]}." >&2
    exit 2
fi

WORK_ROOT="${WORK_ROOT:-./instantsplat_reusable}"
INSTANTSPLAT_DIR="${INSTANTSPLAT_DIR:-./InstantSplatPP}"
GPU_ID="${GPU_ID:-0}"
N_VIEWS="${N_VIEWS:-5}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-1000}"
RESOLUTION="${RESOLUTION:-1}"
MODEL_TYPE="${MODEL_TYPE:-}"

SOURCE_PATH="${WORK_ROOT}/source"
IMAGE_DIR="${SOURCE_PATH}/images"
MODEL_PATH="${WORK_ROOT}/model"
PLY_OUT_DIR="${OUTPUT_FOLDER}/instantsplat_gaussians"

validate_frame_number() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "${name} must be a non-negative integer, got '${value}'." >&2
        exit 2
    fi
}

frame_path() {
    local cam="$1"
    local frame="$2"
    printf "%s/%s/frame_%06d.png" "${INPUT_FOLDER}" "${cam}" "${frame}"
}

prepare_frame() {
    local frame="$1"

    rm -rf "${SOURCE_PATH}"
    mkdir -p "${IMAGE_DIR}"

    for cam in "${CAMERAS[@]}"; do
        local src
        src="$(frame_path "${cam}" "${frame}")"
        if [[ ! -f "${src}" ]]; then
            echo "Missing frame ${frame} for camera '${cam}': ${src}" >&2
            exit 1
        fi

        cp "${src}" "${IMAGE_DIR}/${cam}.png"
    done
}

run_instantsplat() {
    local frame="$1"
    local model_type_args=()
    if [[ -n "${MODEL_TYPE}" ]]; then
        model_type_args=(--model_type "${MODEL_TYPE}")
    fi

    rm -rf "${MODEL_PATH}"
    mkdir -p "${MODEL_PATH}"

    (
        cd "${INSTANTSPLAT_DIR}"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_PATH}" -W ignore ./init_geo.py \
            -s "${SOURCE_PATH}" \
            -m "${MODEL_PATH}" \
            --n_views "${N_VIEWS}" \
            --focal_avg \
            --co_vis_dsp \
            --conf_aware_ranking \
            --infer_video \
            "${model_type_args[@]}" \
            > "${MODEL_PATH}/01_init_geo_frame_${frame}.log" 2>&1

        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_PATH}" ./train.py \
            -s "${SOURCE_PATH}" \
            -m "${MODEL_PATH}" \
            -r "${RESOLUTION}" \
            --n_views "${N_VIEWS}" \
            --iterations "${TRAIN_ITERATIONS}" \
            --pp_optimizer \
            --optim_pose \
            > "${MODEL_PATH}/02_train_frame_${frame}.log" 2>&1
    )
}

copy_gaussian_ply() {
    local frame="$1"
    local ply="${MODEL_PATH}/point_cloud/iteration_${TRAIN_ITERATIONS}/point_cloud.ply"
    local dst="${PLY_OUT_DIR}/frame_$(printf '%06d' "${frame}")_gaussian.ply"

    if [[ ! -f "${ply}" ]]; then
        echo "Expected Gaussian PLY not found: ${ply}" >&2
        exit 1
    fi

    mkdir -p "${PLY_OUT_DIR}"
    cp -f "${ply}" "${dst}"
}

copy_cameras_json() {
    local frame="$1"
    local cameras_json="${MODEL_PATH}/cameras.json"
    local dst="${PLY_OUT_DIR}/frame_$(printf '%06d' "${frame}")_cameras.json"

    if [[ ! -f "${cameras_json}" ]]; then
        echo "Expected cameras.json not found: ${cameras_json}" >&2
        exit 1
    fi

    mkdir -p "${PLY_OUT_DIR}"
    cp -f "${cameras_json}" "${dst}"
}

validate_frame_number START_FRAME "${START_FRAME}"
validate_frame_number END_FRAME "${END_FRAME}"

if [[ "$((10#${START_FRAME}))" -gt "$((10#${END_FRAME}))" ]]; then
    echo "START_FRAME must be <= END_FRAME." >&2
    exit 2
fi

if [[ ! -d "${INPUT_FOLDER}" ]]; then
    echo "Input folder not found: ${INPUT_FOLDER}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_PATH}" && ! -f "${PYTHON_PATH}" ]]; then
    echo "Python interpreter not found: ${PYTHON_PATH}" >&2
    exit 1
fi

echo "Input folder: ${INPUT_FOLDER}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo "Cameras: ${CAMERAS[*]}"
echo "Frames: ${START_FRAME}..${END_FRAME}"
echo "Scratch work folder: ${WORK_ROOT}"
echo "PLY output folder: ${PLY_OUT_DIR}"

processed=0
for ((frame = 10#${START_FRAME}; frame <= 10#${END_FRAME}; frame++)); do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparing frame ${frame}"
    prepare_frame "${frame}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running InstantSplat for frame ${frame}"
    run_instantsplat "${frame}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Copying Gaussian PLY for frame ${frame}"
    copy_gaussian_ply "${frame}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Copying cameras.json for frame ${frame}"
    copy_cameras_json "${frame}"

    processed=$((processed + 1))
done

echo "Done. Processed ${processed} frame set(s). PLY files are in ${PLY_OUT_DIR}"
