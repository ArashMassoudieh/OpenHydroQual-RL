"""
Whole-record (2020-2023, perfect forecast) comparison on
J = Excess/Excess_passive + R * Age_Load/Age_passive:

  rule (native)     the tuned forecast valve in its own kernel (switches inside the hour)
  rule (hourly)     the same rule applied hourly on the orifice kernel (= PPO with zero correction)
  orifice           best constant opening
  ppo:<ckpt>        deterministic PPO policy (correction on the hourly rule)

python evaluate_sf.py --R 0 --ckpts 'results/ppo_sf_R0_u*.pt' [--out results/eval_R0.json]
python evaluate_sf.py --R 0 --split train --ckpts 'results/ppo_sf_R0_train_u*.pt'
    held-out test: controllers tuned/trained on 2020-2022; every run covers the whole
    record under its own control, J is reported for 2020-2022 (selection) and for 2023
    (test; normalized by the passive cell in 2023)
python evaluate_sf.py --R 0 --split ho2020 --ckpts 'results/ppo_sf_R0_ho2020_u*.pt'
    the same with 2020 held out (tuned/trained on 2021-2023, tested on 2020)
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import run_full, SPLIT_DEFS          # noqa: E402
from sf_env import StFrancisEnv                  # noqa: E402

RES = os.path.join(HERE, "results")
T0, T1 = 43831.0, 45291.0


def rollout(R, ckpt=None, record=False, split="full", features=None):
    torch.set_num_threads(1)
    env = StFrancisEnv(R=R, episode_days=T1 - T0, split=split, features=features)
    obs, _ = env.reset(options={"start": T0})
    model = None
    if ckpt:
        from train_ppo import ActorCritic
        model = ActorCritic(env.observation_space.shape[0], 1, 64)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    rec, done, mark = [], False, None
    while not done:
        a = np.zeros(1, np.float32) if model is None else model.act(obs[None], deterministic=True)[0]
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        if abs(info["t"] - SPLIT_DEFS[split]["mark"]) < 1e-6:
            mark = env.k.integral[env.obj].copy()
        if record:
            o = env.k.outputs()
            rec.append([info["t"], info["opening"], info["u_rule"], o[env.ix["outflow"]], o[env.ix["weir"]],
                        o[env.ix["valve"]], o[env.ix["pond"]], info["dExcess"], info["dAge"]])
    I = env.k.integral[env.obj]
    out = {"Excess": float(I[0]), "Age_Load": float(I[1]), "failed": bool(term)}
    if mark is not None:
        out["mark"] = {"Excess": float(mark[0]), "Age_Load": float(mark[1])}
    return (out, np.array(rec)) if record else out


def _job(job):
    R, kind, arg, split = job
    mark = SPLIT_DEFS[split]["mark"]
    if kind == "rule_native":
        r = run_full("rule", arg, t_mark=mark)
    elif kind == "orifice":
        r = run_full("rl", policy=lambda t, o: arg, dt=1 / 24, t_mark=mark)
    elif kind == "passive":
        r = run_full("passive", t_mark=mark)
    else:
        return rollout(R, arg, split=split)
    return {"Excess": r["Excess"], "Age_Load": r["Age_Load"], "failed": not r["ok"],
            "mark": {k: r["mark"][k] for k in ("Excess", "Age_Load")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, required=True)
    ap.add_argument("--ckpts", default="")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--split", choices=sorted(SPLIT_DEFS), default="full")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sfx = SPLIT_DEFS[a.split]["sfx"]
    tag = f"R{a.R:g}{sfx}"
    rule = json.load(open(os.path.join(RES, f"rule_{tag}.json")))["params"]
    orif = json.load(open(os.path.join(RES, f"orifice_{tag}.json")))["opening"]
    jobs = [("passive", (a.R, "passive", None, a.split)),
            ("rule (native)", (a.R, "rule_native", rule, a.split)), ("rule (hourly)", (a.R, "ppo", None, a.split)),
            (f"orifice {orif:.5f}", (a.R, "orifice", orif, a.split))]
    for c in sorted(glob.glob(a.ckpts)) if a.ckpts else []:
        jobs.append((f"ppo:{os.path.basename(c)}", (a.R, "ppo", c, a.split)))
    with Pool(a.workers) as pool:
        rs = pool.map(_job, [j for _, j in jobs])
    pas = rs[0]
    after = {k: (lambda r, k=k: r[k] - r["mark"][k]) for k in ("Excess", "Age_Load")}
    if SPLIT_DEFS[a.split]["test"] == "after":      # test year after the mark (2023)
        windows = {"train": lambda r: r["mark"], "test": lambda r: {k: f(r) for k, f in after.items()}}
    elif SPLIT_DEFS[a.split]["test"] == "before":   # test year before the mark (2020)
        windows = {"train": lambda r: {k: f(r) for k, f in after.items()}, "test": lambda r: r["mark"]}
    else:
        windows = {"full": lambda r: r}
    table = []
    for (name, _), r in zip(jobs, rs):
        row = {"policy": name, "failed": r["failed"]}
        line = f"{name:30s}"
        for w, f in windows.items():
            v, p = f(r), f(pas)
            ex, ag = v["Excess"] / p["Excess"], v["Age_Load"] / p["Age_Load"]
            J = ex + a.R * ag
            row[w] = {"J": J, "Excess_rel": ex, "Age_rel": ag, "Excess": v["Excess"], "Age_Load": v["Age_Load"]}
            line += f"  {w}: J {J:.5f} (Ex {ex:.4f}, Age {ag:.4f})"
        table.append(row)
        print(line + ("  FAILED" if r["failed"] else ""), flush=True)
    sel = "train" if sfx else "full"
    ppo = [t for t in table if t["policy"].startswith("ppo:") and not t["failed"]]
    rule_row = table[1]
    if ppo:
        b = min(ppo, key=lambda t: t[sel]["J"])
        msg = f"selected PPO (best {sel} J): {b['policy']}"
        for w in windows:
            msg += f"  | {w}: J {b[w]['J']:.5f} vs rule {rule_row[w]['J']:.5f} ({(b[w]['J'] / rule_row[w]['J'] - 1) * 100:+.1f}%)"
        print(msg)
    if a.out:
        json.dump(table, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
