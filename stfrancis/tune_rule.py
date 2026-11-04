"""
Re-tune the forecast-informed valve rule (S, I0, tau_k) and the best fixed
orifice on the new additive objective, with the perfect forecast, over the
whole 2020-2023 record (the study's in-sample protocol):

  J = Excess / Excess_passive + R * Age_Load / Age_Load_passive

python tune_rule.py passive                 # passive reference -> results/passive.json
python tune_rule.py rule --R 0 [--R 1]      # CMA-ES on log(S), log(I0), log(tau_k)
python tune_rule.py orifice --R 0 [--R 1]   # sweep of a constant opening
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from common import HERE, SPLITS, SPLIT_DEFS, run_full

RES = os.path.join(HERE, "results")
# Bounds and starts in the convention of OHQ's corrected exponential kernel (trapezoidal
# rule with the 1/2): the study's bounds and optima, tuned with the old kernel (which
# doubled Ihat), with S and I0 halved -- the same controllers and the same search space.
BOUNDS = {"S": (5, 2000), "I0": (5e-7, 0.015), "tau_k": (0.001, 100)}
# study optima (E_0.001 + R Age) as starting points
STARTS = [{"S": 351.08, "I0": 5.18e-7, "tau_k": 1.583}, {"S": 281.34, "I0": 0.0052635, "tau_k": 2.237}]


SPLIT = ""          # "" whole record, "_train" 2020-2022 (set from --split)


def passive_ref():
    return json.load(open(os.path.join(RES, f"passive{SPLIT}.json")))


def score(res, R, ref):
    if not res["ok"]:
        return 1e9
    return res["Excess"] / ref["Excess"] + R * res["Age_Load"] / ref["Age_Load"]


def _eval_rule(p):
    return run_full("rule", p, t0=SPLITS[SPLIT][0], t1=SPLITS[SPLIT][1])


def _eval_orifice(u):
    return run_full("rl", policy=lambda t, o: u, dt=10.0, t0=SPLITS[SPLIT][0], t1=SPLITS[SPLIT][1])


def to_params(x):
    return {k: float(np.exp(np.clip(xi, np.log(lo), np.log(hi))))
            for xi, (k, (lo, hi)) in zip(x, BOUNDS.items())}


def cmaes(f_batch, x0, sigma0, gens, lam, log):
    """Plain (mu/mu_w, lambda) CMA-ES."""
    n = len(x0)
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1 / np.sum(w ** 2)
    cc, cs = (4 + mueff / n) / (n + 4 + 2 * mueff / n), (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    m, sig = np.array(x0, float), sigma0
    Cm, pc, ps = np.eye(n), np.zeros(n), np.zeros(n)
    rng = np.random.default_rng(0)
    best = (np.inf, None, None)
    for g in range(gens):
        D2, B = np.linalg.eigh(Cm)
        D = np.sqrt(np.maximum(D2, 1e-20))
        z = rng.standard_normal((lam, n))
        y = z @ np.diag(D) @ B.T
        X = m + sig * y
        F, extra = f_batch(X)
        idx = np.argsort(F)
        if F[idx[0]] < best[0]:
            best = (F[idx[0]], X[idx[0]].copy(), extra[idx[0]])
        log(f"gen {g:3d}  best {F[idx[0]]:.5f}  overall {best[0]:.5f}  sigma {sig:.3f}  "
            f"{to_params(best[1])}")
        yw = w @ y[idx[:mu]]
        m = m + sig * yw
        Cinvsqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinvsqrt @ yw)
        hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chin < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
        Y = y[idx[:mu]]
        Cm = ((1 - c1 - cmu) * Cm + c1 * (np.outer(pc, pc) + (not hs) * cc * (2 - cc) * Cm)
              + cmu * (Y.T * w) @ Y)
        sig *= np.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))
        if sig < 1e-3:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["passive", "rule", "orifice"])
    ap.add_argument("--R", type=float, action="append", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--split", choices=sorted(SPLIT_DEFS), default="full",
                    help="tuning window: whole record, 2020-2022 (train) or 2021-2023 (ho2020)")
    a = ap.parse_args()
    global SPLIT
    SPLIT = SPLIT_DEFS[a.split]["sfx"]
    os.makedirs(RES, exist_ok=True)

    if a.what == "passive":
        r = run_full("passive", t0=SPLITS[SPLIT][0], t1=SPLITS[SPLIT][1])
        json.dump({"Excess": r["Excess"], "Age_Load": r["Age_Load"]},
                  open(os.path.join(RES, f"passive{SPLIT}.json"), "w"), indent=1)
        print(r)
        return

    ref = passive_ref()
    pool = Pool(a.workers)
    for R in a.R or [0.0]:
        tag = f"R{R:g}{SPLIT}"
        logf = open(os.path.join(RES, f"{a.what}_{tag}_log.txt"), "w")

        def log(s):
            print(s, flush=True)
            logf.write(s + "\n")
            logf.flush()

        tic = time.time()
        if a.what == "orifice":
            us = np.geomspace(2e-4, 0.05, 48)
            rs = pool.map(_eval_orifice, us)
            J = [score(r, R, ref) for r in rs]
            for u, r, j in zip(us, rs, J):
                log(f"opening {u:.5f}  J {j:.5f}  Excess {r['Excess']:.1f}  Age {r['Age_Load']:.1f}")
            i = int(np.argmin(J))
            out = {"opening": float(us[i]), "J": J[i], "Excess": rs[i]["Excess"], "Age_Load": rs[i]["Age_Load"],
                   "sweep": [[float(u), j] for u, j in zip(us, J)]}
        else:
            # the better of the two study optima on this objective is the start
            r0 = pool.map(_eval_rule, STARTS)
            j0 = [score(r, R, ref) for r in r0]
            for s, j in zip(STARTS, j0):
                log(f"start {s}  J {j:.5f}")
            x0 = np.log([STARTS[int(np.argmin(j0))][k] for k in BOUNDS])

            def f_batch(X):
                ps = [to_params(x) for x in X]
                rs = pool.map(_eval_rule, ps)
                return np.array([score(r, R, ref) for r in rs]), rs

            Jb, xb, rb = cmaes(f_batch, x0, 1.0, a.gens, 12, log)
            out = {"params": to_params(xb), "J": float(Jb), "Excess": rb["Excess"], "Age_Load": rb["Age_Load"]}
        out.update({"R": R, "Excess_rel": out["Excess"] / ref["Excess"],
                    "Age_rel": out["Age_Load"] / ref["Age_Load"], "minutes": (time.time() - tic) / 60})
        log(json.dumps(out))
        json.dump(out, open(os.path.join(RES, f"{a.what}_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
