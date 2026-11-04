#!/usr/bin/env bash
# Official benchmark run on macOS (PROTOCOL §6.1). Extra arguments go to the
# runner; default is the quick grid:   scripts/official_run.sh [--quick | runner args]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Run ./setup.sh first." >&2; exit 1; }
[ "$(uname -s)" = Darwin ] || echo "warning: official runs are defined for macOS" >&2
[ -z "$(git status --porcelain --untracked-files=no)" ] || {
  echo "Uncommitted changes to tracked files; commit or stash them first." >&2; exit 1; }
ARGS=("$@"); [ ${#ARGS[@]} -gt 0 ] || ARGS=(--quick)

cat <<'T'
Before an official run:
  [ ] plugged in, lid open (or clamshell with external display)
  [ ] quit every other app (browsers, Slack, Docker, IDEs, the Claude app is OK if idle)
  [ ] no backup / cloud sync / Spotlight indexing in progress
      optional: sudo mdutil -a -i off   (afterwards: sudo mdutil -a -i on)
T
read -r -p "Press Enter when ready... " _

# Wait (up to 3 minutes) for the machine to settle below the --strict limits.
for i in $(seq 1 18); do
  if "$PY" - <<'P'
import sys
from bench import sysload
s = sysload.snapshot()
p = sysload.strict_problems(s)
print("  quiet" if not p else "  waiting: " + "; ".join(p))
sys.exit(1 if p else 0)
P
  then break; fi
  [ "$i" = 18 ] && { echo "Machine did not settle; not running." >&2; exit 2; }
  sleep 10
done

# caffeinate: no idle/display/system sleep while the run lasts.
exec caffeinate -dims "$PY" -m bench.run_prior_art --strict --reps 11 "${ARGS[@]}"
