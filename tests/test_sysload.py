from bench import sysload


def _snap(**kw):
    s = {"loadavg_1_5_15": [0, 0, 0], "others_cpu_pct": 3.0,
         "top_processes": [{"pid": 1, "cpu_pct": 3.0, "name": "x"}],
         "ac_power": True, "thermal": "No thermal warning level has been recorded"}
    s.update(kw)
    return s


def test_quiet_machine_passes():
    assert sysload.strict_problems(_snap()) == []


def test_busy_battery_and_throttled_are_reported():
    probs = sysload.strict_problems(_snap(
        others_cpu_pct=80.0, ac_power=False,
        thermal="CPU_Scheduler_Limit \t= 100 | CPU_Available_CPUs \t= 10 | CPU_Speed_Limit \t= 70"))
    assert len(probs) == 3
    assert "80%" in probs[0] and "battery" in probs[1] and "70%" in probs[2]


def test_full_speed_limit_is_fine():
    assert sysload.strict_problems(_snap(thermal="CPU_Speed_Limit \t= 100")) == []


def test_snapshot_shape():
    s = sysload.snapshot()
    assert len(s["loadavg_1_5_15"]) == 3 and s["others_cpu_pct"] >= 0
