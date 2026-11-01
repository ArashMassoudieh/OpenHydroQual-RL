"""
Evaluate PPO on the stormwater pond against tuned baselines on held-out rain
(2020 onward), using the exact metrics: Q_p (outflow exceeded a fraction p of
the time) and the young-water fraction from the Laplace age tracer.

    python evaluate_pond.py --policy results/ppo_pond.pt --episodes 30

Stage 1 (tuning, on training-period windows) can run while training is still
going: python evaluate_pond.py --tune-only
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import itertools
import json
import re
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "report/figures"
TUNE_FILE = "results/pond_tuning.json"
FIXED_U = [0.03, 0.05, 0.1, 0.2, 1.0]
RULE_GRID = list(itertools.product([0.0, 0.02], [0.05, 0.1, 0.2, 0.4], [0.01, 0.015, 0.03]))
TUNE_YEARS = (1996.4, 2009.8)       # starts of the two 3-year tuning windows (training period)


def make_policy(spec, policy_path):
    from ohq_pond_env import passive_policy, forecast_rule_policy
    kind = spec[0]
    if kind == "fixed":
        return lambda env, obs: passive_policy(env, spec[1])
    if kind == "rule":
        u_hold, u_pre, thr = spec[1:]
        return lambda env, obs: forecast_rule_policy(env, u_hold, u_pre, thr)
    import torch
    torch.set_num_threads(1)
    from train_ppo import ActorCritic
    model = ActorCritic(9, 1)
    model.load_state_dict(torch.load(policy_path))
    model.eval()
    return lambda env, obs: model.act(obs, deterministic=True)


def run_episode(job):
    """job = (policy spec, split, window, policy path, keep trajectory); window is a
    test-window index (test split) or a start year (train split)."""
    from ohq_pond_env import PondEnv
    spec, split, window, policy_path, keep = job
    env = PondEnv(split=split)
    policy = make_policy(spec, policy_path)
    if split == "test":
        opts = {"test_window": window}
    else:
        opts = {"start": int(env.valid_starts[np.searchsorted(env.year[env.valid_starts], window)])}
    obs, _ = env.reset(seed=0, options=opts)
    rows, ret = [], 0.0
    while True:
        obs, r, term, trunc, info = env.step(policy(env, obs))
        ret += r
        if keep:
            rows.append({k: info[k] for k in ("u", "depth", "C", "inflow", "Q_mean", "Q_max", "rain", "overflow")})
        if term or trunc:
            break
    out = {k: info[k] for k in ("Q_p", "young_fraction", "outflow_volume", "Q_peak", "per_year")}
    out["return"] = ret
    out["overflow"] = float(sum(r["overflow"] for r in rows)) if keep else None
    out["start"] = env.start
    if keep:
        out["traj"] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    env.close()
    return out


def tune(pool, n):
    seeds = list(TUNE_YEARS)[:n]
    specs = [("fixed", u) for u in FIXED_U] + [("rule", *g) for g in RULE_GRID]
    res = pool.map(run_episode, [(sp, "train", s, None, False) for sp in specs for s in seeds], chunksize=1)
    table = []
    for i, sp in enumerate(specs):
        chunk = res[i * n:(i + 1) * n]
        table.append(dict(spec=list(sp),
                          Q_p=float(np.mean([c["Q_p"] for c in chunk])),
                          young=float(np.mean([c["young_fraction"] for c in chunk]))))
    json.dump(table, open(TUNE_FILE, "w"), indent=1)
    return table


def normalize_table(table):
    """J of each tuned setting, normalized by the passive pond (fixed opening 1)
    on the same tuning windows (ratio of the two-window means)."""
    from ohq_pond_env import evaluation_objective
    p = [r for r in table if r["spec"] == ["fixed", 1.0]][0]
    for r in table:
        r["J"] = evaluation_objective(r["Q_p"], r["young"], p["Q_p"], p["young"])
    return table


def best(table, kind):
    rows = [r for r in table if r["spec"][0] == kind]
    return min(rows, key=lambda r: r["J"])


def plot_tradeoff(table, results):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    p = [r for r in table if r["spec"] == ["fixed", 1.0]][0]
    for kind, mk, lab in (("fixed", "s", "fixed opening (tuning)"), ("rule", "o", "forecast rule (tuning)")):
        rows = [r for r in table if r["spec"][0] == kind]
        ax.scatter([r["Q_p"] / p["Q_p"] for r in rows], [r["young"] / p["young"] for r in rows],
                   marker=mk, s=22, alpha=0.5, label=lab)
    passive = results["Open valve"]
    for i, (k, runs) in enumerate(results.items()):
        x = np.mean([r["Q_p"] / passive[w]["Q_p"] for w, r in enumerate(runs)])
        y = np.mean([r["young_fraction"] / passive[w]["young_fraction"] for w, r in enumerate(runs)])
        ax.scatter(x, y, marker="*", s=220, color=f"C{i + 2}", edgecolor="k", label=f"{k} (test)", zorder=5)
    xs = np.linspace(0, max(1.6, ax.get_xlim()[1]), 50)
    for J in (0.4, 0.8, 1.2, 1.6, 2.0):
        ax.plot(xs, J - xs, color="0.7", lw=0.6, ls=":")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(r"$Q_p$ / passive $Q_p$  ($Q_p$: outflow exceeded 0.1% of the time)")
    ax.set_ylabel("young fraction / passive young fraction")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title("dotted: lines of equal objective $J$", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/pond_tradeoff.pdf")
    plt.close(fig)


def plot_trajectory(trajs, idx, name):
    first = next(iter(trajs.values()))[idx]["traj"]
    n = len(first["u"]) // 24 * 24
    day = lambda x, f=np.mean: f(np.asarray(x)[:n].reshape(-1, 24), axis=1)
    days = np.arange(n // 24)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].bar(days, day(first["rain"], np.sum) * 1000, width=1, color="C0")
    axes[0].set_ylabel("rain [mm/d]")
    for i, (label, tr) in enumerate(trajs.items()):
        t = tr[idx]["traj"]
        axes[1].plot(days, day(t["depth"]), color=f"C{i}", lw=0.8, label=label)
        axes[2].plot(days, day(t["Q_max"], np.max), color=f"C{i}", lw=0.8)
        axes[3].plot(days, day(t["C"]), color=f"C{i}", lw=0.8)
    axes[1].axhline(2.0, color="r", ls="--", lw=0.8)
    axes[1].set_ylabel("daily mean pond\ndepth [m]")
    axes[1].legend(fontsize=7, ncol=4, loc="upper right")
    axes[2].set_ylabel("daily max outflow\nto outlet [m$^3$/d]")
    axes[3].set_ylabel("daily mean pond\ntracer C")
    axes[3].set_xlabel("day of the 3-year window")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)
    # 20-day zoom, hourly, around the wettest day
    k = int(np.argmax(day(first["rain"], np.sum)))
    h0, h1 = max(0, (k - 8) * 24), min(n, (k + 12) * 24)
    hours = np.arange(h0, h1) / 24
    fig, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
    axes[0].bar(hours, first["rain"][h0:h1] * 1000, width=1 / 24, color="C0")
    axes[0].set_ylabel("rain [mm/h]")
    for i, (label, tr) in enumerate(trajs.items()):
        t = tr[idx]["traj"]
        c = f"C{i}"
        axes[1].step(hours, t["u"][h0:h1], where="post", color=c, label=label)
        axes[2].plot(hours, t["depth"][h0:h1], color=c)
        axes[3].plot(hours, t["Q_max"][h0:h1], color=c)
        axes[4].plot(hours, t["C"][h0:h1], color=c)
    axes[1].set_ylabel("valve opening")
    axes[1].legend(fontsize=7, ncol=4, loc="upper right")
    axes[2].axhline(2.0, color="r", ls="--", lw=0.8)
    axes[2].set_ylabel("pond depth [m]")
    axes[3].set_ylabel("hourly max outflow\nto outlet [m$^3$/d]")
    axes[4].set_ylabel("pond tracer C")
    axes[4].set_xlabel("day of the 3-year window")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}_zoom.pdf")
    plt.close(fig)


def write_tables(results):
    """Per-policy means over the two 3-year test windows, and paired per-year
    differences (6 test years) against PPO."""
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Policy & window & $J$ & $Q_p$ [m$^3$/d] & young fraction & peak [m$^3$/d] & overflow [m$^3$] \\", r"\midrule"]
    for k, runs in results.items():
        for w, r in enumerate(runs):
            lines.append(f"{k if w == 0 else ''} & {['2020--22', '2023--25'][w]} & {r['J']:.2f} & {r['Q_p']:.0f} & "
                         f"{r['young_fraction']:.3f} & {r['Q_peak']:.0f} & {r['overflow']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("report/pond_results_table.tex", "w").write("\n".join(lines) + "\n")
    year = lambda k, m: np.array([y[m] for r in results[k] for y in r["per_year"]])
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"baseline & mean $\Delta J$ (per year) & std.\ error & PPO better in & mean $\Delta Q_p$ / $\Delta$young \\", r"\midrule"]
    for base in [k for k in results if k != "PPO"]:
        dJ = year("PPO", "J") - year(base, "J")
        dQ = np.mean(year("PPO", "Q_p") - year(base, "Q_p"))
        dY = np.mean(year("PPO", "young_fraction") - year(base, "young_fraction"))
        se = dJ.std(ddof=1) / np.sqrt(len(dJ))
        wins = int((dJ < 0).sum())
        print(f"PPO - {base:24s}: per-year dJ {dJ.mean():+.3f} +- {se:.3f}, better in {wins}/{len(dJ)} years; dQ_p {dQ:+.0f}, dyoung {dY:+.3f}")
        lines.append(f"{base} & ${dJ.mean():+.3f}$ & {se:.3f} & {wins}/{len(dJ)} years & ${dQ:+.0f}$ / ${dY:+.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("report/pond_paired_summary.tex", "w").write("\n".join(lines) + "\n")


def plot_learning_curve(log_path):
    # 3-year episodes finish rarely, so plot the mean reward per step of each rollout.
    steps, rews, stds = [], [], []
    for line in open(log_path):
        m = re.search(r"steps\s+(\d+).*std\s+([\d.]+)\s+reward/step\s+(-?[\d.]+)", line)
        if m:
            steps.append(int(m.group(1))); stds.append(float(m.group(2))); rews.append(float(m.group(3)))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    smooth = np.convolve(rews, np.ones(10) / 10, mode="valid") if len(rews) >= 10 else rews
    ax[0].plot(np.array(steps) / 1e3, rews, color="C0", alpha=0.3, lw=0.8)
    ax[0].plot(np.array(steps[len(steps) - len(smooth):]) / 1e3, smooth, color="C0")
    ax[0].set_ylabel("mean reward per hour\n(rollout; bold: 10-update mean)")
    ax[1].plot(np.array(steps) / 1e3, stds, color="C3")
    ax[1].set_ylabel(r"policy std $\sigma$")
    for a in ax:
        a.set_xlabel("environment steps (thousands)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/pond_learning_curve.pdf")
    plt.close(fig)


def main(args):
    os.makedirs(FIG, exist_ok=True)
    with Pool(args.workers) as pool:
        if args.tune_only or not os.path.exists(TUNE_FILE):
            table = tune(pool, args.tune_episodes)
        else:
            table = json.load(open(TUNE_FILE))
        table = normalize_table(table)
        bf, br = best(table, "fixed"), best(table, "rule")
        print("best fixed opening:", bf, "\nbest forecast rule:", br)
        if args.tune_only:
            return
        policies = {"Open valve": ("fixed", 1.0),
                    f"Fixed u={bf['spec'][1]:g}": tuple(bf["spec"]),
                    "Tuned forecast rule": tuple(br["spec"]),
                    "PPO": ("ppo",)}
        results = {k: pool.map(run_episode, [(sp, "test", w, args.policy, True) for w in (0, 1)])
                   for k, sp in policies.items()}

    # Normalized objective: each window (and each test year) against the passive
    # pond (valve fully open) under the same rain.
    from ohq_pond_env import evaluation_objective
    passive = results["Open valve"]
    for runs in results.values():
        for w, r in enumerate(runs):
            r["J"] = evaluation_objective(r["Q_p"], r["young_fraction"], passive[w]["Q_p"], passive[w]["young_fraction"])
            for y, py in zip(r["per_year"], passive[w]["per_year"]):
                y["J"] = evaluation_objective(y["Q_p"], y["young_fraction"], py["Q_p"], py["young_fraction"])

    summary = {k: {m: [r[m] for r in v] for m in ("J", "Q_p", "young_fraction", "Q_peak", "overflow", "return")}
               for k, v in results.items()}
    for k, v in results.items():
        for w, r in enumerate(v):
            ov = r["overflow"]
            print(f"{k:24s} window {w}: J {r['J']:6.3f}  Q_p {r['Q_p']:7.0f}  young {r['young_fraction']:.3f}  "
                  f"peak {r['Q_peak']:6.0f}  overflow {ov:7.0f}  end depth {r['traj']['depth'][-1]:.2f}")
    write_tables(results)
    open("report/pond_tuning_summary.tex", "w").write(
        f"fixed opening $u={bf['spec'][1]:g}$; forecast rule $u_{{hold}}={br['spec'][1]:g}$, "
        f"$u_{{pre}}={br['spec'][2]:g}$, threshold {1000 * br['spec'][3]:g}~mm")
    for w in (0, 1):
        plot_trajectory(results, w, f"pond_trajectory_window{w}")
    plot_tradeoff(table, results)
    plot_learning_curve(args.log)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.boxplot([[y["J"] for r in results[k] for y in r["per_year"]] for k in results],
               tick_labels=list(results), showmeans=True)
    ax.set_ylabel("objective $J$ per test year\n(lower is better)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIG}/pond_J_boxplot.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="results/ppo_pond.pt")
    p.add_argument("--log", default="results/train_log_pond.txt")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--tune-episodes", type=int, default=2)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--tune-only", action="store_true")
    main(p.parse_args())
