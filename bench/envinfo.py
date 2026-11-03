"""Capture the environment alongside every result (PROTOCOL §6)."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


def collect() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "logical_cpus": os.cpu_count(),
        "packages": {p: _version(p) for p in
                     ("rapidfuzz", "edlib", "stringzilla", "stringzillas-cpus", "numpy")},
    }
    if sys.platform == "darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["perf_cores"] = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        info["efficiency_cores"] = _run(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
        info["macos"] = platform.mac_ver()[0]
        info["power"] = _run(["pmset", "-g", "batt"]).splitlines()[:2]
    else:
        model = [l for l in _run(["cat", "/proc/cpuinfo"]).splitlines() if "model name" in l]
        info["cpu"] = model[0].split(":", 1)[1].strip() if model else platform.processor()
    try:
        import stringzillas as szs
        info["stringzilla_capabilities"] = list(getattr(szs, "__capabilities__", ()))
        from bench.baselines import stringzilla_semantics
        info["stringzilla_semantics"] = stringzilla_semantics()
    except Exception:
        pass
    from bench import cmyers
    info["c_kernels"] = cmyers.buildinfo() or "not built"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    info["git_commit"] = _run(["git", "-C", root, "rev-parse", "HEAD"])
    info["git_dirty"] = bool(_run(["git", "-C", root, "status", "--porcelain"]))
    return info
