"""How busy is this machine? Recorded before and after every run, and used by
--strict to refuse to start on a busy or unplugged machine (PROTOCOL §6)."""

from __future__ import annotations

import os
import re
import subprocess
import sys


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def top_processes(n: int = 5) -> list[dict]:
    """Other processes by recent CPU use (%, where 100 = one full core)."""
    me = os.getpid()
    rows = []
    for line in _run(["ps", "-A", "-o", "pid=,pcpu=,comm="]).splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, pcpu = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid != me:
            rows.append({"pid": pid, "cpu_pct": pcpu, "name": os.path.basename(parts[2].strip())})
    rows.sort(key=lambda r: -r["cpu_pct"])
    return rows[:n]


def on_ac_power() -> bool | None:
    if sys.platform != "darwin":
        return None
    out = _run(["pmset", "-g", "batt"])
    return "AC Power" in out if out else None


def thermal() -> str:
    """macOS: pmset's thermal report (a speed limit < 100 means throttling)."""
    if sys.platform != "darwin":
        return "n/a"
    return " | ".join(l.strip() for l in _run(["pmset", "-g", "therm"]).splitlines() if l.strip())


def snapshot() -> dict:
    top = top_processes(8)
    return {
        "loadavg_1_5_15": [round(x, 2) for x in os.getloadavg()],
        "others_cpu_pct": round(sum(r["cpu_pct"] for r in top_processes(10_000)), 1),
        "top_processes": top[:5],
        "ac_power": on_ac_power(),
        "thermal": thermal(),
    }


# --strict thresholds: other processes together may use at most a quarter of
# one core, and the machine must be plugged in.
MAX_OTHERS_CPU_PCT = 25.0


def strict_problems(s: dict) -> list[str]:
    probs = []
    if s["others_cpu_pct"] > MAX_OTHERS_CPU_PCT:
        busy = ", ".join(f'{r["name"]} {r["cpu_pct"]:.0f}%' for r in s["top_processes"])
        probs.append(f'other processes use {s["others_cpu_pct"]:.0f}% CPU '
                     f'(limit {MAX_OTHERS_CPU_PCT:.0f}%): {busy}')
    if s["ac_power"] is False:
        probs.append("running on battery")
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", s["thermal"])
    if m and int(m.group(1)) < 100:
        probs.append(f"CPU speed limited to {m.group(1)}% (thermal)")
    return probs
