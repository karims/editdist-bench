#!/usr/bin/env bash
# Phase 5: the whole GPU run on a rented NVIDIA machine (Linux x86_64 with an
# NVIDIA driver for CUDA 12+). One command, ~30-60 minutes, ends with one
# archive to download:
#
#   git clone <repo> editdist-bench   (or unpack the tarball)
#   cd editdist-bench && bash scripts/gpu_run.sh
#
# Pick a machine image with the CUDA toolkit ("CUDA devel" / PyTorch images on
# RunPod, Lambda, Vast.ai all qualify). Plain Ubuntu also needs:
#   apt install -y python3-dev python3-venv build-essential
#
# Steps: record the machine -> venv + deps -> build -> all tests incl. GPU
# correctness -> main comparison -> batch-size sweep (H5.3) -> archive.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null || { echo "No nvidia-smi: not an NVIDIA machine." >&2; exit 1; }
nvidia-smi

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    PY="$(command -v "$c")"; break; fi
done
[ -n "$PY" ] || { echo "Need Python >= 3.10 (apt install python3.10-venv)" >&2; exit 1; }
"$PY" -m venv .venv-gpu
V=.venv-gpu/bin/python
$V -m pip install -q --upgrade pip
$V -m pip install -q -r requirements.txt
$V -m pip uninstall -q -y stringzillas-cpus          # same module name as the CUDA build
$V -m pip install -q -r requirements-gpu.txt
make PY="$V"

echo "== tests (must all pass, GPU ones included) =="
$V -m pytest -q
$V - <<'P'
from bench import cudamyers
import stringzillas as szs
assert cudamyers.available(), "CuPy cannot see the GPU"
print("GPU:", cudamyers.device_info())
print("StringZilla capabilities:", szs.__capabilities__)
P

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr -cd '[:alnum:]-')
OUT="results/gpu-$GPU_NAME"
mkdir -p "$OUT"
RUN="$V -m bench.run_prior_art --reps 11 --out $OUT"

echo "== 1/2 main comparison: GPU vs all-cores CPU (H5.1, H5.2) =="
$RUN --threads all,gpu --lengths 16,32,64,128,256 --errors 0.05,0.2 --alphabets bytes,dna \
     --pairs 1000000 --cross 2000 --exclude edlib,loop,expanded

echo "== 2/2 batch-size sweep: when does the GPU start to win (H5.3) =="
for n in 1000 10000 100000 1000000; do
  $RUN --threads all,gpu --shape pairs --lengths 32 --errors 0.05 --alphabets bytes \
       --pairs "$n" --exclude edlib,loop,expanded
done
for q in 30 100 300 1000 3000; do
  $RUN --threads all,gpu --shape cross --lengths 32 --errors 0.05 --alphabets bytes \
       --cross "$q" --exclude edlib,loop,expanded
done

tar czf "results-$GPU_NAME.tar.gz" "$OUT"
echo
echo "Done. Download results-$GPU_NAME.tar.gz and unpack it into the repo on your Mac."
