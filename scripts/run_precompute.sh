#!/usr/bin/env bash
set -euo pipefail

# Run feature precompute and embeddings precompute for the training dataset.
# Expects to be run from repo root. Uses Python virtualenv with dependencies installed.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/build_training_dataset"
TRAIN_CSV="$DATA_DIR/training_database.clean.csv"
OUT_FEATURES="$DATA_DIR/training_database.features.csv"
OUT_PREFIX="$DATA_DIR/training_database"

echo "Running feature pipeline..."
python3 "$ROOT_DIR/scripts/precompute_features.py" --input "$TRAIN_CSV" --out_csv "$OUT_FEATURES" --out_prefix "$OUT_PREFIX"

echo "Running embeddings precompute..."
python3 "$ROOT_DIR/scripts/precompute_embeddings.py" --path "$TRAIN_CSV" --out_csv "$OUT_PREFIX.emb.csv" --out_npy "$OUT_PREFIX.emb.npy"

echo "Precompute finished. Artifacts written to $DATA_DIR"
