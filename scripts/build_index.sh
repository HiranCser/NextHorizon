#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/build_training_dataset"
EMB_NPY="$DATA_DIR/training_database.emb.npy"
INDEX_OUT="$DATA_DIR/faiss.index"

if [ ! -f "$EMB_NPY" ]; then
  echo "Embedding file not found: $EMB_NPY"
  echo "Run scripts/run_precompute.sh first to generate embeddings."
  exit 2
fi

python3 - <<'PY'
import os
import sys
import numpy as np
try:
    import faiss
except Exception:
    print('faiss not installed. To build an ANN index install faiss (cpu): pip install faiss-cpu')
    sys.exit(3)

EMB_NPY = os.path.join(os.path.dirname(__file__), '..', 'build_training_dataset', 'training_database.emb.npy')
EMB_NPY = os.path.abspath(EMB_NPY)
arr = np.load(EMB_NPY)
print('Loaded embeddings', arr.shape)
d = arr.shape[1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(arr)
index.add(arr.astype('float32'))
out_path = os.path.join(os.path.dirname(__file__), '..', 'build_training_dataset', 'faiss.index')
faiss.write_index(index, out_path)
print('Wrote FAISS index to', out_path)
PY
