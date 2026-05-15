#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_dataset.sh small
#   bash scripts/download_dataset.sh full

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS_DIR="$REPO_DIR/datasets"
DOWNLOADS_DIR="$DATASETS_DIR/.downloads"

usage() {
    cat >&2 <<EOF
Usage: bash scripts/download_dataset.sh <small|full>

Required argument:
  small   Download Dataset008_ME2-Stem.zip
  full    Download the full Zenodo files archive

Datasets are extracted under:
  $DATASETS_DIR

Downloaded zip files are removed after extraction succeeds.
EOF
}

if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

DATASET="$1"

case "$DATASET" in
    small)
        URL="https://zenodo.org/records/17635006/files/Dataset008_ME2-Stem.zip?download=1"
        ARCHIVE_NAME="Dataset008_ME2-Stem.zip"
        ;;
    full)
        URL="https://zenodo.org/api/records/17635006/files-archive"
        ARCHIVE_NAME="Dataset008_ME2-full-archive.zip"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown dataset: $DATASET" >&2
        usage
        exit 1
        ;;
esac

ARCHIVE_PATH="$DOWNLOADS_DIR/$ARCHIVE_NAME"
EXTRACT_DIR="$DATASETS_DIR"

mkdir -p "$DOWNLOADS_DIR" "$EXTRACT_DIR"

download() {
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --continue-at - "$URL" -o "$ARCHIVE_PATH"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "$ARCHIVE_PATH" "$URL"
    else
        echo "Need curl or wget to download the dataset." >&2
        exit 1
    fi
}

extract_zip() {
    if command -v unzip >/dev/null 2>&1; then
        unzip -o "$ARCHIVE_PATH" -d "$EXTRACT_DIR"
    elif command -v python3 >/dev/null 2>&1; then
        python3 - "$ARCHIVE_PATH" "$EXTRACT_DIR" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with ZipFile(archive) as zip_file:
    zip_file.extractall(destination)
PY
    elif command -v python >/dev/null 2>&1; then
        python - "$ARCHIVE_PATH" "$EXTRACT_DIR" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with ZipFile(archive) as zip_file:
    zip_file.extractall(destination)
PY
    else
        echo "Need unzip, python3, or python to extract the dataset." >&2
        exit 1
    fi
}

echo "Downloading $DATASET dataset..."
download

echo "Extracting to $EXTRACT_DIR..."
extract_zip

rm -f "$ARCHIVE_PATH"

cat <<EOF

Done.
Dataset directory:  $EXTRACT_DIR
EOF
