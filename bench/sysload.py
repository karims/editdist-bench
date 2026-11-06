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


def effective_cpus() -> int:
    """CPUs this process may actually use: the smallest of the CPU affinity
    set and the container's CPU quota (cgroup v2 cpu.max or v1 cfs quota).
    On rented machines the quota is often far below os.cpu_count() (e.g. a
    pod seeing 128 threads but allowed 13.6 CPUs), and libraries told to use
    "all cores" then oversubscribe and get throttled."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = None
    try:
        q, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    if quota is not None:
        n = min(n, max(1, int(quota)))
    return max(1, n)


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


# --strict thresholds: other processes together may use at most half of one
# core (of 10 on the M4; measured wall/CPU stayed ~1.00 at this level), and
# the machine must be plugged in.
MAX_OTHERS_CPU_PCT = 50.0


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
