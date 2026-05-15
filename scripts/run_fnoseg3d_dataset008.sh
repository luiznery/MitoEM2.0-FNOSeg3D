#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_fnoseg3d_dataset008.sh
#   bash scripts/run_fnoseg3d_dataset008.sh --epochs 50
#   bash scripts/run_fnoseg3d_dataset008.sh --batch-size 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_NAME="Dataset008_ME2-Stem"
DATASET_DIR="$REPO_DIR/datasets/$DATASET_NAME"
CONFIG_PATH="$REPO_DIR/configs/fnoseg3d_dataset008.yaml"
ACTIVATE_ENV_SCRIPT="$SCRIPT_DIR/activate_mitoem2_env.sh"

EXTRA_ARGS=()

usage() {
    cat >&2 <<EOF
Usage: bash scripts/run_fnoseg3d_dataset008.sh [options]

Runs the FNOSeg3D training pipeline for:
  $DATASET_DIR

Options:
  --epochs N               Set number of training epochs (default: 100)
  --batch-size N          Set batch size (default: 1)
  --learning-rate LR      Set learning rate (default: 0.001)
  --filters N             Set number of filters (default: 24)
  --modes N               Set number of frequency modes (default: 16)
  --blocks N              Set number of FNO blocks (default: 8)
  -h, --help              Show this help

Environment overrides:
  PYTHON=/path/to/python
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --epochs)
            if [ "$#" -lt 2 ]; then
                echo "--epochs requires a value" >&2
                exit 1
            fi
            EXTRA_ARGS+=("--epochs" "$2")
            shift 2
            ;;
        --batch-size)
            if [ "$#" -lt 2 ]; then
                echo "--batch-size requires a value" >&2
                exit 1
            fi
            EXTRA_ARGS+=("--batch-size" "$2")
            shift 2
            ;;
        --learning-rate)
            if [ "$#" -lt 2 ]; then
                echo "--learning-rate requires a value" >&2
                exit 1
            fi
            EXTRA_ARGS+=("--learning-rate" "$2")
            shift 2
            ;;
        --filters)
            if [ "$#" -lt 2 ]; then
                echo "--filters requires a value" >&2
                exit 1
            fi
            EXTRA_ARGS+=("--filters" "$2")
            shift 2
            ;;
        --modes)
            if [ "$#" -lt 2 ]; then
                echo "--modes requires a value" >&2
                exit 1
            fi
            EXTRA_ARGS+=("--modes" "$2")
            shift 2
            ;;
        --blocks)
            if [ "$#" -lt 2 ]; then
                echo "--blocks requires a value" >&2
                exit 1
            fi
            EXTRA_ARGS+=("--blocks" "$2")
            shift 2
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

cd "$REPO_DIR"

echo "Running FNOSeg3D training for $DATASET_NAME"
echo "Config:  $CONFIG_PATH"
echo

exec "$PYTHON_BIN" -m mitoem2.scripts.train \
    --method fnoseg3d \
    --config "$CONFIG_PATH" \
    "${EXTRA_ARGS[@]}"
