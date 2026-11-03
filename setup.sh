#!/usr/bin/env bash
# One-command setup: restore git history (if shipped as a bundle), create a
# Python >= 3.10 virtualenv, install dependencies, run the tests.
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VENV:-.venv}"   # e.g. VENV=.venv-linux when running inside a Linux VM

# 1. Git history arrives as editdist-bench.bundle when files are copied in by a
#    tool that cannot write .git directly.
if [ -f editdist-bench.bundle ]; then
  if [ ! -d .git ]; then
    git init -q -b main
  fi
  git fetch -q ./editdist-bench.bundle main
  if git rev-parse -q --verify HEAD >/dev/null && \
     ! git merge-base --is-ancestor HEAD FETCH_HEAD; then
    echo "Local commits are not in the bundle; not touching git. Resolve manually." >&2
    exit 1
  fi
  git reset -q FETCH_HEAD          # move branch; working tree already has the files
  rm editdist-bench.bundle
  echo "git: $(git log --oneline | wc -l | tr -d ' ') commits, HEAD $(git log -1 --format=%h)"
fi

# 2. Find Python >= 3.10.
PY=""
for c in python3.13 python3.12 python3.11 python3.10 \
         /opt/homebrew/bin/python3 "$HOME/miniconda3/bin/python3" python3; do
  if command -v "$c" >/dev/null 2>&1 && \
     "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Need Python >= 3.10. Install one with:  brew install python@3.12" >&2
  exit 1
fi
echo "python: $PY ($("$PY" --version 2>&1))"

# 3. Fresh virtualenv and dependencies.
rm -rf "$VENV"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r requirements.txt

# 4. Correctness before anything else.
"$VENV/bin/python" -m pytest -q

echo
echo "Ready. Next:"
echo "  source $VENV/bin/activate"
echo "  python -m bench.run_prior_art --quick"
