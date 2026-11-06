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
"$PY" -m venv --clear .venv-gpu                     # fresh every time: reruns stay clean
V=.venv-gpu/bin/python
$V -m pip install -q --upgrade pip
$V -m pip install -q -r requirements.txt
# stringzillas-cpus and stringzillas-cuda install the same module files, so
# removing the CPU build deletes them: force the CUDA build back in afterwards.
$V -m pip uninstall -q -y stringzillas-cpus
$V -m pip install -q -r requirements-gpu.txt
$V -m pip install -q --force-reinstall --no-deps stringzillas-cuda
$V -c "import stringzillas as s; assert 'cuda' in s.__capabilities__, s.__capabilities__" \
  || { echo "stringzillas CUDA build not active" >&2; exit 1; }
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

$V -c "from bench.sysload import effective_cpus as e; import os; print(f'CPUs: {os.cpu_count()} visible, {e()} allowed (multi-core libraries use {e()})')"

# GPU, multi-core CPU and single-core CPU run as separate processes, so no
# library's leftover worker threads can steal CPU time from another kind.
GRID="--lengths 16,32,64,128,256 --errors 0.05,0.2 --alphabets bytes,dna"
SKIP="--exclude edlib,loop,expanded"

echo "== 1/2 main comparison: GPU (H5.1, H5.2) =="
$RUN --threads gpu $GRID --pairs 1000000 --cross 2000 $SKIP
echo "== 1/2 main comparison: all allowed CPU cores (H5.2) =="
$RUN --threads all $GRID --pairs 1000000 --cross 2000 $SKIP
echo "== 1/2 main comparison: one CPU core (reference, smaller batches) =="
$RUN --threads 1 $GRID --pairs 100000 --cross 300 $SKIP

echo "== 2/2 batch-size sweep: when does the GPU start to win (H5.3) =="
for kind in gpu all; do
  for n in 1000 10000 100000 1000000; do
    $RUN --threads $kind --shape pairs --lengths 32 --errors 0.05 --alphabets bytes --pairs "$n" $SKIP
  done
  for q in 30 100 300 1000 3000; do
    $RUN --threads $kind --shape cross --lengths 32 --errors 0.05 --alphabets bytes --cross "$q" $SKIP
  done
done

tar czf "results-$GPU_NAME.tar.gz" "$OUT"
echo
echo "Done. Download results-$GPU_NAME.tar.gz and unpack it into the repo on your Mac."
