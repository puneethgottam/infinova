#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# User configuration (optional)
#
# Fill these in if you'd rather not pass them on the command line every time.
# Anything passed as a CLI argument overrides these defaults.
# Leave a value empty ("") to require it on the command line.
# -----------------------------------------------------------------------------
DEFAULT_GAUSSIANS_DIR=""
DEFAULT_OUT_DIR=""
DEFAULT_PYTHON_PATH=""
DEFAULT_PAIRS=()        # e.g. (cam0,cam4 cam4,cam2 cam2,cam3 cam3,cam1)
DEFAULT_SAMPLES_PER_PAIR="20"
DEFAULT_Y_JITTER_RANGE=""   # e.g. "-0.05 0.40"
DEFAULT_Z_JITTER_RANGE=""   # e.g. "-0.10 0.20"
DEFAULT_SEED=""
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  render_novel_views.sh [GAUSSIANS_DIR OUT_DIR PYTHON_PATH] --pairs PAIR1 PAIR2 ... [options]

Samples novel camera positions between pairs of an episode's real cameras,
then renders every Gaussian PLY in GAUSSIANS_DIR from those novel positions.

GAUSSIANS_DIR must contain:
  - cameras.json              (the consensus-best camera file, from select_best_cameras_json.py)
  - one or more frame_*_gaussian.ply files (from run_instantsplat_episode.sh)

Arguments:
  GAUSSIANS_DIR  Folder produced by run_instantsplat_episode.sh for one episode
                 (contains cameras.json and frame_*_gaussian.ply files)
  OUT_DIR        Folder to write rendered novel-view PNGs to
  PYTHON_PATH    Path to the python interpreter with InstantSplatPP's dependencies installed
  --pairs        One or more camera pairs to sample between, e.g. --pairs cam0,cam4 cam4,cam2
                 Names must match the img_name fields in cameras.json.

Options:
  --samples-per-pair N     Number of positions to sample per pair (default: 20)
  --y-jitter-range MIN MAX Random Y offset range applied to each sampled position
  --z-jitter-range MIN MAX Random Z offset range applied to each sampled position
  --seed N                 Random seed for reproducible sampling

Examples:
  render_novel_views.sh ./out/episode_7/instantsplat_gaussians ./out/episode_7/novel_views \
      /usr/bin/python3 --pairs cam0,cam4 cam4,cam2 cam2,cam3 cam3,cam1 \
      --samples-per-pair 20 --y-jitter-range -0.05 0.40 --z-jitter-range -0.10 0.20

  # With DEFAULT_* variables filled in at the top of the script:
  render_novel_views.sh

Environment overrides:
  INSTANTSPLAT_DIR=./InstantSplatPP   Path to the InstantSplatPP repo (used only for reference; the
                                       render step imports it directly via render_ply_custom_pose.py)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

GAUSSIANS_DIR="${DEFAULT_GAUSSIANS_DIR}"
OUT_DIR="${DEFAULT_OUT_DIR}"
PYTHON_PATH="${DEFAULT_PYTHON_PATH}"
PAIRS=("${DEFAULT_PAIRS[@]+"${DEFAULT_PAIRS[@]}"}")
SAMPLES_PER_PAIR="${DEFAULT_SAMPLES_PER_PAIR}"
Y_JITTER_RANGE=(${DEFAULT_Y_JITTER_RANGE})
Z_JITTER_RANGE=(${DEFAULT_Z_JITTER_RANGE})
SEED="${DEFAULT_SEED}"

if [[ "$#" -gt 0 && "$1" != --* ]]; then
    if [[ "$#" -lt 3 ]]; then
        echo "Expected GAUSSIANS_DIR OUT_DIR PYTHON_PATH." >&2
        usage >&2
        exit 2
    fi
    GAUSSIANS_DIR="$1"
    OUT_DIR="$2"
    PYTHON_PATH="$3"
    shift 3
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --pairs)
            shift
            PAIRS=()
            while [[ "$#" -gt 0 && "$1" != --* ]]; do
                PAIRS+=("$1")
                shift
            done
            ;;
        --samples-per-pair)
            SAMPLES_PER_PAIR="$2"
            shift 2
            ;;
        --y-jitter-range)
            Y_JITTER_RANGE=("$2" "$3")
            shift 3
            ;;
        --z-jitter-range)
            Z_JITTER_RANGE=("$2" "$3")
            shift 3
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${GAUSSIANS_DIR}" || -z "${OUT_DIR}" || -z "${PYTHON_PATH}" ]]; then
    echo "Missing GAUSSIANS_DIR/OUT_DIR/PYTHON_PATH: pass them as arguments or set the DEFAULT_* variables at the top of this script." >&2
    usage >&2
    exit 2
fi

if [[ "${#PAIRS[@]}" -eq 0 ]]; then
    echo "Expected at least one camera pair (via --pairs or DEFAULT_PAIRS)." >&2
    exit 2
fi

CAMERAS_JSON="${GAUSSIANS_DIR}/cameras.json"
CAMERA_POSITIONS_FILE="${OUT_DIR}/sampled_camera_positions.txt"

if [[ ! -d "${GAUSSIANS_DIR}" ]]; then
    echo "Gaussians folder not found: ${GAUSSIANS_DIR}" >&2
    exit 1
fi

if [[ ! -f "${CAMERAS_JSON}" ]]; then
    echo "Missing ${CAMERAS_JSON}. Run select_best_cameras_json.py on this folder first." >&2
    exit 1
fi

if ! compgen -G "${GAUSSIANS_DIR}/frame_*_gaussian.ply" > /dev/null; then
    echo "No frame_*_gaussian.ply files found in ${GAUSSIANS_DIR}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "Gaussians folder: ${GAUSSIANS_DIR}"
echo "Output folder: ${OUT_DIR}"
echo "Camera pairs: ${PAIRS[*]}"
echo "Samples per pair: ${SAMPLES_PER_PAIR}"

sample_args=(
    --json "${CAMERAS_JSON}"
    --pairs "${PAIRS[@]}"
    --samples-per-pair "${SAMPLES_PER_PAIR}"
    --output "${CAMERA_POSITIONS_FILE}"
)
if [[ "${#Y_JITTER_RANGE[@]}" -eq 2 ]]; then
    sample_args+=(--y-jitter-range "${Y_JITTER_RANGE[@]}")
fi
if [[ "${#Z_JITTER_RANGE[@]}" -eq 2 ]]; then
    sample_args+=(--z-jitter-range "${Z_JITTER_RANGE[@]}")
fi
if [[ -n "${SEED}" ]]; then
    sample_args+=(--seed "${SEED}")
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sampling novel camera positions"
"${PYTHON_PATH}" "${SCRIPT_DIR}/sample_camera_positions_with_yz_jitter.py" "${sample_args[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering novel views"
"${PYTHON_PATH}" "${SCRIPT_DIR}/render_ply_custom_pose.py" \
    --lookat_cameras_json "${CAMERAS_JSON}" \
    --ply_root "${GAUSSIANS_DIR}" \
    --ply_pattern "frame_*_gaussian.ply" \
    --out_dir "${OUT_DIR}" \
    --camera_positions_file "${CAMERA_POSITIONS_FILE}"

echo "Done. Novel views are in ${OUT_DIR}"
