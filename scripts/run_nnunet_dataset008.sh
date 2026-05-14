#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_nnunet_dataset008.sh
#   bash scripts/run_nnunet_dataset008.sh --fold 1
#   bash scripts/run_nnunet_dataset008.sh --skip-training --skip-prediction

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_NAME="Dataset008_ME2-Stem"
DATASET_DIR="$REPO_DIR/datasets/$DATASET_NAME"
CONFIG_PATH="$REPO_DIR/configs/nnunet_dataset008.yaml"
ACTIVATE_ENV_SCRIPT="$SCRIPT_DIR/activate_mitoem2_env.sh"

NNUNET_RAW="${NNUNET_RAW:-$REPO_DIR/datasets}"
NNUNET_PREPROCESSED="${NNUNET_PREPROCESSED:-$REPO_DIR/nnunet_preprocessed}"
NNUNET_RESULTS="${NNUNET_RESULTS:-$REPO_DIR/nnunet_results}"

EXTRA_ARGS=()

usage() {
    cat >&2 <<EOF
Usage: bash scripts/run_nnunet_dataset008.sh [options]

Runs the 3D nnUNet pipeline for:
  $DATASET_DIR

Options forwarded to mitoem2.scripts.train:
  --fold N                 Set nnUNet fold. Default from config: 0
  --skip-boundary          Skip boundary label generation
  --skip-plan              Skip nnUNet planning/preprocessing
  --skip-training          Skip nnUNet training
  --skip-prediction        Skip nnUNet prediction
  --skip-postprocess       Skip postprocessing
  --skip-evaluation        Skip evaluation
  -h, --help               Show this help

Environment overrides:
  PYTHON=/path/to/python
  MITOEM2_SCRATCH_DIR=/path/to/scratch
  MITOEM2_ENV_PREFIX=/path/to/conda_env
  NNUNET_RAW=/path/to/nnUNet_raw
  NNUNET_PREPROCESSED=/path/to/nnUNet_preprocessed
  NNUNET_RESULTS=/path/to/nnUNet_results
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --fold)
            if [ "$#" -lt 2 ]; then
                echo "--fold requires a value" >&2
                exit 1
            fi
            EXTRA_ARGS+=(--nnunet-fold "$2")
            shift 2
            ;;
        --skip-boundary)
            EXTRA_ARGS+=(--nnunet-skip-boundary)
            shift
            ;;
        --skip-plan)
            EXTRA_ARGS+=(--nnunet-skip-plan)
            shift
            ;;
        --skip-training)
            EXTRA_ARGS+=(--nnunet-skip-training)
            shift
            ;;
        --skip-prediction)
            EXTRA_ARGS+=(--nnunet-skip-prediction)
            shift
            ;;
        --skip-postprocess)
            EXTRA_ARGS+=(--nnunet-skip-postprocess)
            shift
            ;;
        --skip-evaluation)
            EXTRA_ARGS+=(--nnunet-skip-evaluation)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset not found: $DATASET_DIR" >&2
    echo "Download it first with: bash scripts/download_dataset.sh small" >&2
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

if [ ! -f "$ACTIVATE_ENV_SCRIPT" ]; then
    echo "Environment activation script not found: $ACTIVATE_ENV_SCRIPT" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ACTIVATE_ENV_SCRIPT"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "$NNUNET_PREPROCESSED" "$NNUNET_RESULTS"

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"
export RESULTS_FOLDER="$NNUNET_RESULTS"

cd "$REPO_DIR"

echo "Running 3D nnUNet for $DATASET_NAME"
echo "nnUNet_raw:          $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results:      $nnUNet_results"
echo "Config:              $CONFIG_PATH"
echo

exec "$PYTHON_BIN" -m mitoem2.scripts.train \
    --method nnunet \
    --config "$CONFIG_PATH" \
    "${EXTRA_ARGS[@]}"
