#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
MODEL_ROOT="$ROOT_DIR/ct-models"
ARCHIVE="$MODEL_ROOT/downloads/Dataset112_DentalSegmentator_v100.zip"
URL="https://zenodo.org/api/records/10829675/files/Dataset112_DentalSegmentator_v100.zip/content"

mkdir -p \
  "$MODEL_ROOT/downloads" \
  "$MODEL_ROOT/raw" \
  "$MODEL_ROOT/preprocessed" \
  "$MODEL_ROOT/results"

if [ ! -f "$ARCHIVE" ]; then
  echo "Downloading published DentalSegmentator Dataset112 weights (~230 MB)..."
  curl --fail --location "$URL" --output "$ARCHIVE"
fi

cd "$ROOT_DIR"
docker compose build ct-sidecar
docker compose run --rm ct-sidecar \
  nnUNetv2_install_pretrained_model_from_zip \
  /models/downloads/Dataset112_DentalSegmentator_v100.zip

echo "DentalSegmentator weights installed under ct-models/results."
