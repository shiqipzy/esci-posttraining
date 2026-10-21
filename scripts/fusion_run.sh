#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON_BIN="${ESCI_PYTHON:-/root/miniconda3/bin/python}"
# Prevent concurrent jobs writing to the same experiment directory.
exec 9>.fusion_run.lock
flock -n 9 || { echo 'Another fusion run is already active.'; exit 2; }
"$PYTHON_BIN" -B src/train_fusion_continue.py --gpu-check
"$PYTHON_BIN" -u -B src/train_fusion_continue.py --train --evaluate --resume
