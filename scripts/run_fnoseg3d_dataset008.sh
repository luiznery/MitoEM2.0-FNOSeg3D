#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_NAME="Dataset008_ME2-Stem"
DATASET_DIR="$REPO_DIR/datasets/$DATASET_NAME"
CONFIG_PATH="$REPO_DIR/configs/nnunet_dataset008.yaml"
ACTIVATE_ENV_SCRIPT="$SCRIPT_DIR/activate_mitoem2_env.sh"

EXTRA_ARGS=()

usage() {
    cat >&2 <<EOF
Usage: bash scripts/run_fnoseg3d_dataset008.sh [options]

Runs FNOSeg3D training for dataset:
  $DATASET_DIR

Options forwarded to mitoem2.scripts.train:
  --resume <path>           Resume from checkpoint
  -h, --help                Show this help

Environment overrides:
  PYTHON=/path/to/python
  MITOEM2_ENV_PREFIX=/path/to/conda_env
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --resume)
            if [ "$#" -lt 2 ]; then
                echo "--resume requires a value" >&2
                exit 1
            fi
            EXTRA_ARGS+=(--resume "$2")
            shift 2
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
    exit 1
fi

if [ ! -f "$ACTIVATE_ENV_SCRIPT" ]; then
    echo "Environment activation script not found: $ACTIVATE_ENV_SCRIPT" >&2
    exit 1
fi

# Activate the project's conda environment inside this script
# This ensures `python` and other tools come from the activated env.
# If the user set `PYTHON` env var, we will use that instead.
source "$ACTIVATE_ENV_SCRIPT"

# Verify python comes from the activated environment (or use PYTHON override)
if [ -n "${PYTHON-}" ]; then
    PYTHON_BIN="$PYTHON"
else
    if ! command -v python >/dev/null 2>&1; then
        echo "python not found in PATH after environment activation" >&2
        exit 1
    fi
    PYTHON_BIN="$(command -v python)"
fi

echo "Using python: $PYTHON_BIN"

cd "$REPO_DIR"

echo "Running FNOSeg3D training for $DATASET_NAME"
echo "Config: $CONFIG_PATH"

exec "$PYTHON_BIN" -m mitoem2.scripts.train \
    --method fnoseg3d \
    --config "$CONFIG_PATH" \
    "${EXTRA_ARGS[@]}"
