#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${ROOT_DIR}/data/downloads"
LIBRI_DIR="${ROOT_DIR}/data/librispeech"
ASVSPOOF_DIR="${ROOT_DIR}/data/asvspoof2019"

mkdir -p "${DOWNLOAD_DIR}" "${LIBRI_DIR}" "${ASVSPOOF_DIR}"

extract_if_present() {
  local archive="$1"
  local dest="$2"
  if [[ -f "${archive}" ]]; then
    if ! tar -tzf "${archive}" >/dev/null 2>&1; then
      echo "Skipping incomplete archive ${archive}"
      return
    fi
    echo "Extracting ${archive} -> ${dest}"
    mkdir -p "${dest}"
    tar -xzf "${archive}" -C "${dest}"
  fi
}

extract_zip_if_present() {
  local archive="$1"
  local dest="$2"
  if [[ -f "${archive}" ]]; then
    if ! unzip -tq "${archive}" >/dev/null 2>&1; then
      echo "Skipping incomplete archive ${archive}"
      return
    fi
    echo "Extracting ${archive} -> ${dest}"
    mkdir -p "${dest}"
    unzip -o "${archive}" -d "${dest}"
  fi
}

extract_if_present "${DOWNLOAD_DIR}/dev-clean.tar.gz" "${LIBRI_DIR}"
extract_if_present "${DOWNLOAD_DIR}/test-clean.tar.gz" "${LIBRI_DIR}"
extract_zip_if_present "${DOWNLOAD_DIR}/LA.zip" "${ASVSPOOF_DIR}"

echo "Dataset extraction complete."
