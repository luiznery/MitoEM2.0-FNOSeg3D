#!/usr/bin/env bash
set -euo pipefail

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

Runs a basic FNOSeg3D training loop for:
  $DATASET_DIR

Options:
  --epochs N          Number of epochs (default set in script)
  --batch-size N      Batch size
  --help              Show this help

Environment overrides:
  PYTHON=/path/to/python
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        *)
            # Forward any unknown options to the FNOSeg3D training script
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset not found: $DATASET_DIR" >&2
    echo "Download it first with: bash scripts/download_dataset.sh small" >&2
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
echo "Dataset: $DATASET_DIR"
echo

exec "$PYTHON_BIN" -m mitoem2.scripts.train_fnoseg3d \
    --data-root "$DATASET_DIR" \
    --output-dir "$REPO_DIR/nnunet_results/$DATASET_NAME/fnoseg3d"
    "${EXTRA_ARGS[@]}"
