"""
Pick the training checkpoint with the best evaluation objective J (passive-
normalized Q_p and young fraction) on 3-year validation windows, instead of
simply keeping the last one.

    python select_checkpoint.py "results/ppo_pond_u*.pt" --out results/ppo_pond.pt

The validation windows are the two training-period tuning windows (starting
1996.4 and 2009.8); the test windows (2020 onward) are not touched.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import glob
import json
import shutil
from multiprocessing import Pool

import numpy as np

from evaluate_pond import run_episode, TUNE_YEARS
from ohq_pond_env import evaluation_objective


def main(args):
    ckpts = sorted(glob.glob(args.pattern))
    jobs = [(("fixed", 1.0), "train", y, None, False) for y in TUNE_YEARS]
    jobs += [(("ppo",), "train", y, c, False) for c in ckpts for y in TUNE_YEARS]
    with Pool(args.workers) as pool:
        res = pool.map(run_episode, jobs, chunksize=1)
    n = len(TUNE_YEARS)
    passive = res[:n]
    table = []
    for i, c in enumerate(ckpts):
        runs = res[n + i * n: n + (i + 1) * n]
        J = [evaluation_objective(r["Q_p"], r["young_fraction"], p["Q_p"], p["young_fraction"])
             for r, p in zip(runs, passive)]
        row = dict(checkpoint=c, J=float(np.mean(J)), J_windows=J,
                   Q_p=[r["Q_p"] for r in runs], young=[r["young_fraction"] for r in runs])
        table.append(row)
        print(f"{os.path.basename(c):24s} J {row['J']:.3f}  windows {np.round(J, 3)}  "
              f"Q_p {np.round(row['Q_p'])}  young {np.round(row['young'], 3)}", flush=True)
    best = min(table, key=lambda r: r["J"])
    json.dump(dict(passive=[{k: p[k] for k in ("Q_p", "young_fraction")} for p in passive], checkpoints=table,
                   best=best["checkpoint"]), open(args.log, "w"), indent=1)
    shutil.copy(best["checkpoint"], args.out)
    print(f"best: {best['checkpoint']} (J {best['J']:.3f}) -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pattern")
    p.add_argument("--out", default="results/ppo_pond.pt")
    p.add_argument("--log", default="results/pond_checkpoint_selection.json")
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args())
