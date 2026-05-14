#!/usr/bin/env bash
set -euo pipefail

# Clone the FNOSeg3D upstream repository into external/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TARGET_DIR="$REPO_DIR/external/multimodal-3d-image-segmentation"

if [ -d "$TARGET_DIR" ]; then
    echo "Target directory already exists: $TARGET_DIR"
    exit 0
fi

echo "Cloning multimodal-3d-image-segmentation into $TARGET_DIR"
git clone https://github.com/ibm/multimodal-3d-image-segmentation.git "$TARGET_DIR"

echo "Done. You can now run FNOSeg3D training."
