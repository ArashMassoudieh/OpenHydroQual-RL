"""Shared paths, constants and helpers for the St. Francis RL study."""
import os
import numpy as np

from kernel import Kernel
from forecast import Indicator

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = {m: os.path.join(HERE, "kernels", m, "build", f"libSF_{m}.so") for m in ("rule", "rl", "passive")}
SERIES = os.path.join(HERE, "forecast_series.npz")
T_START, T_2021, T_SPLIT, T_END = 43831.0, 44197.0, 44927.0, 45291.0   # 2020-01-01, 2021-01-01, 2023-01-01, 2023-12-31
# Data splits: file suffix, tuning/training window, the time that separates training from
# test, and whether the test year comes after or before that time.
SPLIT_DEFS = {"full":   {"sfx": "", "window": (T_START, T_END), "mark": T_SPLIT, "test": None},
              "train":  {"sfx": "_train", "window": (T_START, T_SPLIT), "mark": T_SPLIT, "test": "after"},
              "ho2020": {"sfx": "_ho2020", "window": (T_2021, T_END), "mark": T_2021, "test": "before"}}
SPLITS = {d["sfx"]: d["window"] for d in SPLIT_DEFS.values()}   # suffix -> tuning/training window
OUT_NAMES = {"pond": "Pond (1):Storage", "eng": "Upper_ENg_soil:Storage", "native": "Lowersoil_native:Storage",
             "cb": "Catch basin (2):Storage", "agg": "Aggregate_storage_layer (1):Storage",
             "outflow": "Catch basin (2) - fixed_head (3):flow", "weir": "Pond (1) - Catch basin (2):flow",
             "valve": "Aggregate_storage_layer (1) - Catch basin (2):flow"}


def indicator():
    d = np.load(SERIES)
    return Indicator(d["t"], d["c"])


def rule_opening(ihat, S, I0):
    """The forecast valve as a controlled-orifice opening: sqrt(Ihat/S) when
    Ihat > I0, else closed (valve_K = S/(Ihat + 1e-7) in the valve type)."""
    return float(np.sqrt((ihat + 1e-7) / S)) if ihat > I0 else 0.0


def out_index(k):
    return {key: k.output_names.index(n) for key, n in OUT_NAMES.items()}


def run_full(kind, params=None, policy=None, dt=1 / 24, record=False, t0=None, t1=None, t_mark=None):
    """Run the whole record (or [t0, t1]) on a kernel. kind 'rule'/'passive': params
    are kernel parameters; kind 'rl': policy(t, outputs) -> opening is called
    every dt. Returns objective integrals (and the hourly record)."""
    k = Kernel(KERNELS[kind])
    k.initialize()
    if params:
        k.set_parameters(params)
    t0 = k.t_start if t0 is None else t0
    t1 = k.t_end if t1 is None else t1
    # spin-up from the start of the record under the same policy; integrals count from t0
    t, rec, ok, counting = k.t_start, [], True, t0 <= k.t_start
    stops = [x for x in (t0, t_mark) if x is not None and x > k.t_start]   # landed on exactly
    while ok and t < t1 - 1e-9:
        if not counting and t >= t0 - 1e-9:
            k.integral[:] = 0.0
            counting = True
        if policy is not None:
            k.set_parameters({"Opening": policy(t, k.outputs())})
        t_next = min(t + dt, t1)
        for x in stops:
            if t < x < t_next:
                t_next = x
        t = t_next
        ok = k.run_to(t)
        if record and counting:
            rec.append(np.concatenate([[t], k.outputs()]))
        if t_mark is not None and t == t_mark:
            mark = k.objective_integrals()
    res = k.objective_integrals()
    if t_mark is not None:
        res["mark"] = mark
    res["ok"] = ok
    return (res, np.array(rec)) if record else res
