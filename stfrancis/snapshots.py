"""Continuous run of the tuned rule (hourly, on the orifice kernel) saving the
kernel state every 6 h, as start states for RL episodes.
python snapshots.py --R 0"""
import argparse
import json
import os

import numpy as np

from common import HERE, KERNELS, SPLIT_DEFS, Kernel, indicator, rule_opening

ap = argparse.ArgumentParser()
ap.add_argument("--R", type=float, required=True)
ap.add_argument("--split", choices=sorted(SPLIT_DEFS), default="full")
a = ap.parse_args()
tag = f"R{a.R:g}" + SPLIT_DEFS[a.split]["sfx"]
res = os.path.join(HERE, "results")
p = json.load(open(os.path.join(res, f"rule_{tag}.json")))["params"]
ind = indicator()
k = Kernel(KERNELS["rl"])
k.initialize()
t, dt, every = k.t_start, 1 / 24, 6
out = {x: [] for x in ("t", "storage", "mass", "limited", "limit_factor")}
n = 0
while t < k.t_end - 1e-9:
    if n % every == 0:
        s = k.export_state()
        out["t"].append(t)
        for x in ("storage", "mass", "limited", "limit_factor"):
            out[x].append(s[x])
    k.set_parameters({"Opening": rule_opening(ind.ekr(t, 1 / p["tau_k"]), p["S"], p["I0"])})
    n += 1
    t = k.t_start + n * dt
    assert k.run_to(t)
np.savez(os.path.join(res, f"snapshots_{tag}.npz"), **{x: np.array(v) for x, v in out.items()})
print(tag, len(out["t"]), "snapshots;", k.objective_integrals())
