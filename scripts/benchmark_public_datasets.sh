#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="${ROOT_DIR}/src"

if [[ -d "${ROOT_DIR}/data/librispeech" ]]; then
  PYTHONPATH="${PYTHONPATH}" python3 -m siren.cli benchmark-corpus \
    --dataset-name librispeech \
    --dataset-root "${ROOT_DIR}/data/librispeech" \
    --output-dir "${ROOT_DIR}/results/detection_benchmarks/librispeech_public"
fi

if [[ -d "${ROOT_DIR}/data/asvspoof2019" ]]; then
  PYTHONPATH="${PYTHONPATH}" python3 -m siren.cli benchmark-corpus \
    --dataset-name asvspoof2019 \
    --dataset-root "${ROOT_DIR}/data/asvspoof2019" \
    --output-dir "${ROOT_DIR}/results/detection_benchmarks/asvspoof2019_public"
fi

echo "Public dataset benchmark commands completed."
