#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${ESCI_PYTHON:-/root/miniconda3/bin/python}"
"$PYTHON_BIN" -B src/train_fusion_continue.py --preflight "$@"
