#!/usr/bin/env bash
set -euo pipefail

# Colab-friendly wrapper for downloading MOT17 into Google Drive.
# Usage examples:
#   bash scripts/download_mot17_colab.sh
#   bash scripts/download_mot17_colab.sh --variant labels
#   bash scripts/download_mot17_colab.sh --outdir /content/dataset --force

OUTDIR="/content/drive/MyDrive/datasets"
VARIANT="all"
FORCE=0
OVERWRITE=0
REMOVE_ARCHIVE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      OUTDIR="${2:?missing value for --outdir}"
      shift 2
      ;;
    --variant)
      VARIANT="${2:?missing value for --variant}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --keep-archive)
      REMOVE_ARCHIVE=0
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash scripts/download_mot17_colab.sh [options]

Options:
  --outdir <path>      Where MOT17 will be extracted (default: /content/drive/MyDrive/datasets)
  --variant <all|labels>
                       "all" downloads full MOT17; "labels" downloads annotations only
  --force              Redownload archive even if it already exists
  --overwrite          Remove extracted MOT17 directory before extraction
  --keep-archive       Keep downloaded zip after extraction
  -h, --help           Show this help
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$VARIANT" != "all" && "$VARIANT" != "labels" ]]; then
  echo "Invalid --variant: $VARIANT (expected all or labels)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found." >&2
  exit 1
fi

CMD=(python3 "${REPO_ROOT}/scripts/download_mot17.py" --variant "$VARIANT" --outdir "$OUTDIR")
[[ "$FORCE" -eq 1 ]] && CMD+=(--force)
[[ "$OVERWRITE" -eq 1 ]] && CMD+=(--overwrite)
[[ "$REMOVE_ARCHIVE" -eq 1 ]] && CMD+=(--remove-archive)

echo "[info] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[done] Dataset location: ${OUTDIR}/MOT17"
