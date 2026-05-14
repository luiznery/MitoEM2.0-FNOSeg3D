#!/usr/bin/env bash

# Source this file to activate the mitoem2 conda environment:
#   source scripts/activate_mitoem2_env.sh
#
# Optional overrides:
#   MITOEM2_SCRATCH_DIR=/path/to/scratch source scripts/activate_mitoem2_env.sh
#   MITOEM2_ENV_PREFIX=/path/to/env source scripts/activate_mitoem2_env.sh

MITOEM2_SCRATCH_DIR="${MITOEM2_SCRATCH_DIR:-/tempory/luiznery}"
MITOEM2_MINIFORGE_DIR="${MITOEM2_MINIFORGE_DIR:-$MITOEM2_SCRATCH_DIR/miniforge3}"
MITOEM2_ENV_PREFIX="${MITOEM2_ENV_PREFIX:-$MITOEM2_SCRATCH_DIR/conda_envs/mitoem2}"
MITOEM2_CONDA_SH="$MITOEM2_MINIFORGE_DIR/etc/profile.d/conda.sh"

if [ ! -f "$MITOEM2_CONDA_SH" ]; then
    echo "Conda activation script not found: $MITOEM2_CONDA_SH" >&2
    echo "Run setup first: bash scripts/setup_mitoem2_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$MITOEM2_CONDA_SH"
conda activate "$MITOEM2_ENV_PREFIX"
