#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
  cat <<'HELP'
Run the SportsMOT benchmark on a Colab GPU.
Default: provenance check for archived YOLO11n, best vs last on the same clip.
Upload these two files first:
  /content/yolo11n_sportsmot_best.pt
  /content/v_00HRwkvvjtQ_c001.txt

Run `bash run.sh benchmark` for the full YOLO11m comparison instead.

Optional environment variables:
  MOT_CHECKPOINT     yolo11m_best, yolo11s_best, yolo11n_best, yolo11n_last, yolo11n_10epoch
  MOT_WEIGHTS        Path to a local checkpoint matching the expected SHA-256
  MOT_ARCHIVE_TRACK  Path to the archived YOLO11n none track (provenance mode)
  MOT_VAL            Path to an already extracted SportsMOT val directory
  MOT_SEQUENCE       Run a single named clip
  MOT_METHOD         Run a single method (none, sparse4, sparse8, raft_large128, sift2)
  MOT_CACHE_DIR      Setup cache (default /content/motflow-cache)
  MOT_RESULTS_DIR    Output directory (default /content/motflow-results)
HELP
  exit 0
fi
task=${1:-provenance}
if [[ $# -gt 1 || "$task" != provenance && "$task" != benchmark ]]; then
  echo 'Usage: bash run.sh [provenance|benchmark|--help]' >&2
  exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
root=$PWD
checkpoint=${MOT_CHECKPOINT:-yolo11m_best}
cache=${MOT_CACHE_DIR:-/content/motflow-cache}
results=${MOT_RESULTS_DIR:-/content/motflow-results}
kit_commit=b06cfb4aacb0bb16c5778ac4ddb0d70ec6835444
val_commit=1b0b418ea611a8e934f1e773313e719e4fa1265e
val_sha256=35f1c585297426a2ec104f645c6bf8684e7e856c536775b167fe3f2ffd388ac3
provenance_clip=v_00HRwkvvjtQ_c001
provenance_sha256=d51866f1cfb601cc3ae3a86f4300e20ceab451bf77b346a959aa67fee09b5329

fail() {
  echo "$*" >&2
  exit 1
}

if [[ "$task" == provenance ]]; then
  best_weights=${MOT_WEIGHTS:-/content/yolo11n_sportsmot_best.pt}
  archive_track=${MOT_ARCHIVE_TRACK:-"/content/$provenance_clip.txt"}
  [[ -f "$best_weights" ]] || fail "Upload YOLO11n best from MOT_weights/ to $best_weights."
  [[ -f "$archive_track" ]] || fail "Upload the archived YOLO11n none track to $archive_track."
  [[ $(sha256sum < "$archive_track") == "$provenance_sha256 "* ]] || fail "Wrong archived track file: $archive_track. Use the experiments805nano YOLO11n none track."
  job_name=yolo11n-provenance
else
  job_name=$checkpoint
fi

command -v nvidia-smi >/dev/null || fail 'Enable a GPU runtime in Colab first.'
[[ -n $(nvidia-smi --query-gpu=name --format=csv,noheader) ]] || fail 'No NVIDIA GPU is available.'
mkdir -p -- "$cache" "$results"
cache=$(cd -- "$cache" && pwd)
results=$(cd -- "$results" && pwd)
kit_root="$cache/sportsmot-kit-$kit_commit"
kit="$kit_root/codes/evaluation/TrackEval"
python="$cache/venv/bin/python"
job=$(mktemp -d "$results/$job_name-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")

fetch() {
  local url=$1 target=$2
  if [[ ! -s "$target" ]]; then
    curl --fail --location --silent --show-error --retry 3 "$url" --output "$target.part"
    mv -- "$target.part" "$target"
  fi
}

setup_python() {
  local signature
  signature=$( { python3 --version; sha256sum "$root/run.sh" "$root/requirements.txt"; } | sha256sum )
  if [[ ! -x "$python" || ! -f "$cache/env.signature" || $(cat "$cache/env.signature") != "$signature" ]]; then
    echo "Setting up the Colab environment with $(python3 --version)."
    python3 -m venv "$cache/venv"
    "$python" -m pip install --disable-pip-version-check \
      torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    "$python" -m pip install --disable-pip-version-check -r "$root/requirements.txt" \
      scipy==1.16.3 pillow==12.1.1 pycocotools==2.0.11
    "$python" -m pip check
    printf '%s\n' "$signature" > "$cache/env.signature"
  fi
  "$python" -c 'import torch, torchvision; torch.zeros(1, device="cuda:0"); print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))'
  "$python" -m pip freeze > "$job/pip-freeze.txt"
}

prepare_kit() {
  if [[ ! -f "$kit_root/.ready" ]]; then
    local archive="$cache/sportsmot-kit-$kit_commit.tar.gz"
    fetch "https://codeload.github.com/MCG-NJU/SportsMOT/tar.gz/$kit_commit" "$archive"
    mkdir -p -- "$kit_root"
    tar -xzf "$archive" -C "$kit_root" --strip-components=1
    [[ -f "$kit/trackeval/__init__.py" ]] || fail 'SportsMOT TrackEval is missing from the archive.'
    touch "$kit_root/.ready"
  fi
}

prepare_val() {
  if [[ -z ${MOT_VAL:-} ]]; then
    local archive="$cache/sportsmot-val-$val_commit.tar" extracted="$cache/sportsmot-val-$val_commit" clip_dir
    if [[ ! -f "$extracted/.ready" ]]; then
      echo 'Downloading the official SportsMOT val archive (6.59 GB).'
      fetch "https://huggingface.co/datasets/MCG-NJU/SportsMOT/resolve/$val_commit/dataset/val.tar" "$archive"
      printf '%s  %s\n' "$val_sha256" "$archive" | sha256sum --check
      mkdir -p -- "$extracted"
      tar -xf "$archive" -C "$extracted"
      touch "$extracted/.ready"
    fi
    printf '%s\n' "$val_commit" > "$job/sportsmot-val-commit.txt"
    clip_dir=$(find "$extracted" -type d -name v_00HRwkvvjtQ_c001 -print -quit)
    [[ -n "$clip_dir" ]] || fail "Could not find the val clips under $extracted."
    MOT_VAL=$(dirname -- "$clip_dir")
  fi
  [[ -d "$MOT_VAL" ]] || fail "SportsMOT val directory not found: $MOT_VAL"
}

fetch_weights() {
  local weight_id
  case "$1" in
    yolo11m_best) weight_id=1mHuZBesRudoCe8dBUs9fcs2tKNbAhJBc ;;
    yolo11s_best) weight_id=1Kjo-ei1KV7nVa_jmnVzB8YWVu9tp0BqW ;;
    yolo11n_last) weight_id=1RbC3D9emgMee1VnjczDspzn8spKwXOp9 ;;
    *) fail "$1 requires MOT_WEIGHTS pointing to the verified .pt file." ;;
  esac
  fetch "https://drive.usercontent.google.com/download?id=$weight_id&export=download&confirm=t" "$2"
}

run_benchmark() {
  local weights=${MOT_WEIGHTS:-"$cache/$checkpoint.pt"}
  if [[ -z ${MOT_WEIGHTS:-} ]]; then
    fetch_weights "$checkpoint" "$weights"
  fi

  local args=(
    --checkpoint "$checkpoint" --weights "$weights" --trackeval-kit "$kit"
    --out "$job/benchmark"
  )
  if [[ -n ${MOT_SEQUENCE:-} ]]; then args+=(--sequence "$MOT_SEQUENCE"); fi
  if [[ -n ${MOT_METHOD:-} ]]; then args+=(--method "$MOT_METHOD"); fi
  # Fail on a wrong checkpoint before the large dataset download.
  "$python" scripts/rerun_verified_weights.py "${args[@]}" --data "${MOT_VAL:-$cache}" --dry-run > "$job/plan.yaml"
  prepare_val
  "$python" scripts/rerun_verified_weights.py "${args[@]}" --data "$MOT_VAL"
  if [[ -z ${MOT_METHOD:-} && -z ${MOT_SEQUENCE:-} ]]; then
    "$python" scripts/compare_gmc.py \
      --baseline "$job/benchmark/sparse8/trackeval" \
      --candidate "$job/benchmark/raft_large128/trackeval" \
      --out "$job/comparison"
  fi
}

run_provenance() {
  local last_weights="$cache/yolo11n_last.pt"
  local common=(--trackeval-kit "$kit" --sequence "$provenance_clip" --method none)
  local best=(--checkpoint yolo11n_best --weights "$best_weights" --out "$job/best")
  local last=(--checkpoint yolo11n_last --weights "$last_weights" --out "$job/last")
  fetch_weights yolo11n_last "$last_weights"
  "$python" scripts/rerun_verified_weights.py "${common[@]}" "${best[@]}" --data "${MOT_VAL:-$cache}" --dry-run > "$job/best-plan.yaml"
  "$python" scripts/rerun_verified_weights.py "${common[@]}" "${last[@]}" --data "${MOT_VAL:-$cache}" --dry-run > "$job/last-plan.yaml"
  prepare_val
  "$python" scripts/rerun_verified_weights.py "${common[@]}" "${best[@]}" --data "$MOT_VAL"
  "$python" scripts/rerun_verified_weights.py "${common[@]}" "${last[@]}" --data "$MOT_VAL"

  mkdir -p -- "$job/archive/tracks"
  cp -- "$archive_track" "$job/archive/tracks/$provenance_clip.txt"
  "$python" scripts/evaluate_tracks.py --gt "$MOT_VAL" --tracks "$job/archive/tracks" \
    --out "$job/archive/trackeval" --trackeval-kit "$kit" --sequence "$provenance_clip"
  "$python" scripts/compare_prediction_fingerprints.py --reference "$archive_track" \
    --candidate "$job/best/none/tracks/$provenance_clip.txt" \
    --candidate "$job/last/none/tracks/$provenance_clip.txt" \
    --verdict "$job/verdict.txt" > "$job/fingerprints.csv"
  cat "$job/verdict.txt"
}

run_workspace() {
  git rev-parse HEAD > "$job/git-commit.txt"
  printf '%s\n' "$kit_commit" > "$job/sportsmot-trackeval-commit.txt"
  nvidia-smi > "$job/nvidia-smi.txt"
  setup_python
  prepare_kit
  export PYTHONPATH="$kit"
  export TORCH_HOME="$cache/torch"
  export YOLO_CONFIG_DIR="$cache/ultralytics-config"
  mkdir -p -- "$TORCH_HOME" "$YOLO_CONFIG_DIR"
  if [[ "$task" == provenance ]]; then
    run_provenance
  else
    run_benchmark
  fi
}

set +e
( set -e; run_workspace ) 2>&1 | tee "$job/colab.log"
status=$?
set -e
printf '%s\n' "$status" > "$job/exit-status.txt"
tar -czf "$job.tar.gz" -C "$results" "$(basename -- "$job")"
echo "Results: $job"
echo "Downloadable archive: $job.tar.gz"
exit "$status"
