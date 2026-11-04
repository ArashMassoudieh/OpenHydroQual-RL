"""
Forward stepwise selection of PPO observation features.

Round k trains, for every feature not yet selected, PPO with the selected features plus
that one (n_seeds seeds), scores the final policy of each run on the training window of
the split, and adds the feature with the best mean score. The held-out year is scored and
reported for every run but never used in a choice. A case stops when its best score has
not improved by at least `--min-gain` for `--patience` rounds in a row, or when the pool is
exhausted. Finished runs are kept (results/stepwise/...), so a restarted search resumes.

The reference before any feature is the tuned rule applied hourly (= zero correction).

python stepwise.py --cases 0 1 --split ho2020
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
OUT = os.path.join(HERE, "results", "stepwise")
# candidate pool: the 16 observation values less three near-duplicates (results/feature_corr.npy):
# u_rule is a monotone function of ihat, eng tracks agg (rank corr 0.98), cb tracks weir (0.95)
POOL = ["pond", "agg", "outflow", "weir", "rain_0_3h", "rain_3_12h", "rain_12_24h", "rain_1_3d",
        "rain_3_7d", "rain_past_24h", "ihat", "season"]
LOCK = threading.Lock()


def log(msg):
    with LOCK:
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with open(os.path.join(OUT, "stepwise_log.txt"), "a") as f:
            f.write(line + "\n")


def run_dir(R, feats, seed):
    return os.path.join(OUT, f"R{R}", "+".join(feats), f"seed{seed}")


def train_and_score(R, feats, seed, a):
    d = run_dir(R, feats, seed)
    res = os.path.join(d, "result.json")
    if os.path.exists(res):
        return json.load(open(res))
    os.makedirs(d, exist_ok=True)
    kw = json.dumps({"R": R, "split": a.split, "features": feats})
    cmd = [PY, "train_ppo.py", "--env", "stfrancis", "--env-kwargs", kw, "--total-steps", str(a.steps),
           "--num-envs", str(a.workers), "--rollout-steps", "720", "--epochs", "10", "--minibatch", "1024",
           "--lr", "3e-4", "--gamma", "0.995", "--lam", "0.95", "--normalize-rewards",
           "--bc-rule", "sf_env:zero_policy", "--bc-steps", "2000", "--bc-log-std", "-1.2",
           "--seed", str(seed), "--out", os.path.join(d, "policy.pt")]
    with open(os.path.join(d, "train_log.txt"), "w") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        log(f"R{R} {'+'.join(feats)} seed {seed}: training failed (see {d}/train_log.txt)")
        return None
    with open(os.path.join(d, "eval_log.txt"), "w") as f:
        p = subprocess.run([PY, os.path.abspath(__file__), "--score", d, "--R", str(R), "--split", a.split,
                            "--features", ",".join(feats)], cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    return json.load(open(res)) if os.path.exists(res) else None


def score_run(d, R, split, feats):
    """Whole-record rollout of the final policy; J on the training window and on the held-out year."""
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    import torch
    from common import SPLIT_DEFS
    from evaluate_sf import rollout, _job
    torch.set_num_threads(1)
    r = rollout(R, os.path.join(d, "policy.pt"), split=split, features=feats)
    pas = _job((R, "passive", None, split))
    out = {"features": feats, "failed": r["failed"]}
    before = SPLIT_DEFS[split]["test"] == "before"
    for w in ("train", "test"):
        use_mark = (w == "test") == before
        v = r["mark"] if use_mark else {k: r[k] - r["mark"][k] for k in ("Excess", "Age_Load")}
        p = pas["mark"] if use_mark else {k: pas[k] - pas["mark"][k] for k in ("Excess", "Age_Load")}
        ex, ag = v["Excess"] / p["Excess"], v["Age_Load"] / p["Age_Load"]
        out[w] = {"J": ex + R * ag, "Excess_rel": ex, "Age_rel": ag}
    json.dump(out, open(os.path.join(d, "result.json"), "w"), indent=1)
    print(out)


def reference(R, split):
    """The tuned rule applied hourly (zero correction), on both windows."""
    f = os.path.join(OUT, f"R{R}", "reference.json")
    if os.path.exists(f):
        return json.load(open(f))
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    from evaluate_sf import _job
    from common import SPLIT_DEFS
    r, p = _job((R, "ppo", None, split)), _job((R, "passive", None, split))
    before = SPLIT_DEFS[split]["test"] == "before"
    out = {}
    for w in ("train", "test"):
        use_mark = (w == "test") == before
        v = r["mark"] if use_mark else {k: r[k] - r["mark"][k] for k in ("Excess", "Age_Load")}
        q = p["mark"] if use_mark else {k: p[k] - p["mark"][k] for k in ("Excess", "Age_Load")}
        out[w] = {"J": v["Excess"] / q["Excess"] + R * v["Age_Load"] / q["Age_Load"]}
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(out, open(f, "w"), indent=1)
    return out


def search(R, a, pool):
    ref = reference(R, a.split)
    selected, best, stale, rounds = [], ref["train"]["J"], 0, []
    log(f"R{R}: reference (tuned rule, hourly) train J {ref['train']['J']:.5f}  test J {ref['test']['J']:.5f}")
    while len(selected) < len(POOL):
        cands = [f for f in POOL if f not in selected]
        jobs = {(f, s): pool.submit(train_and_score, R, selected + [f], s, a) for f in cands for s in a.seeds}
        table = []
        for f in cands:
            rs = [jobs[(f, s)].result() for s in a.seeds]
            rs = [r for r in rs if r and not r["failed"]]
            if not rs:
                continue
            table.append({"feature": f, "train_J": float(np.mean([r["train"]["J"] for r in rs])),
                          "test_J": float(np.mean([r["test"]["J"] for r in rs])),
                          "train_J_seeds": [r["train"]["J"] for r in rs], "test_J_seeds": [r["test"]["J"] for r in rs]})
        table.sort(key=lambda t: t["train_J"])
        b = table[0]
        gain = 1 - b["train_J"] / best
        rounds.append({"round": len(selected) + 1, "selected_before": list(selected), "candidates": table,
                       "added": b["feature"], "gain_vs_previous_best": gain})
        selected.append(b["feature"])
        log(f"R{R} round {len(selected)}: add {b['feature']:14s} train J {b['train_J']:.5f} "
            f"(seeds {', '.join(f'{x:.4f}' for x in b['train_J_seeds'])})  test J {b['test_J']:.5f}  "
            f"gain {gain * 100:+.1f}%")
        json.dump({"R": R, "split": a.split, "reference": ref, "selected": selected, "rounds": rounds},
                  open(os.path.join(OUT, f"R{R}", "summary.json"), "w"), indent=1)
        if gain >= a.min_gain:
            best, stale = b["train_J"], 0
        else:
            stale += 1
            if stale >= a.patience:
                log(f"R{R}: stopping, no gain >= {a.min_gain * 100:.0f}% for {a.patience} rounds")
                break
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--split", default="ho2020")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=4, help="rollout workers per training run")
    ap.add_argument("--parallel", type=int, default=4, help="training runs at a time")
    ap.add_argument("--min-gain", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--score", default="")
    ap.add_argument("--R", type=float, default=0)
    ap.add_argument("--features", default="")
    a = ap.parse_args()
    if a.score:
        score_run(a.score, int(a.R), a.split, a.features.split(","))
        return
    os.makedirs(OUT, exist_ok=True)
    log(f"stepwise selection: cases {a.cases}, split {a.split}, {a.steps} steps x seeds {a.seeds}, pool {POOL}")
    with ThreadPoolExecutor(a.parallel) as pool, ThreadPoolExecutor(len(a.cases)) as drivers:
        futs = {R: drivers.submit(search, R, a, pool) for R in a.cases}
        for R, f in futs.items():
            log(f"R{R}: selected {f.result()}")


if __name__ == "__main__":
    main()
