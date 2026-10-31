"""
Evaluate the trained PPO policy against two baselines on held-out storm
sequences (same seeds for every policy) and write figures/tables for the report.

    python evaluate.py --policy results/ppo_reservoir.pt --episodes 30
"""
import argparse
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ohq_reservoir_env import OHQReservoirEnv, rule_curve_policy
from train_ppo import ActorCritic

FIG = "report/figures"


def demand_only_policy(env):
    q = env.demand[env.day]
    return np.array([2.0 * q / env.Q_release_max - 1.0], dtype=np.float32)


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    rows, total = [], 0.0
    while True:
        obs, r, term, trunc, info = env.step(policy(env, obs))
        info["reward"] = r
        rows.append(info)
        total += r
        if term or trunc:
            break
    traj = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    traj["return"] = total
    traj["inflow_t"], traj["inflow_q"] = env.inflow_t.copy(), env.inflow_q.copy()
    return traj


def summarize(trajs, env):
    ret = np.array([t["return"] for t in trajs])
    supply = np.array([t["r_supply"].sum() for t in trajs])
    flood = np.array([t["r_flood"].sum() for t in trajs])
    spill = np.array([t["r_spill"].sum() for t in trajs])
    terminal = ret - supply - flood - spill
    deficit_frac = np.array([t["deficit"].sum() / t["demand"].sum() for t in trajs])
    spill_vol = np.array([t["spill"].sum() for t in trajs])
    flood_days = np.array([((t["release"] + t["spill"]) > env.Q_safe + 1e-6).sum() for t in trajs])
    return dict(ret=ret, supply=supply, flood=flood, spill=spill, terminal=terminal,
                deficit_frac=deficit_frac, spill_vol=spill_vol, flood_days=flood_days)


def plot_learning_curve(log_path):
    steps, rets, stds = [], [], []
    for line in open(log_path):
        m = re.search(r"steps\s+(\d+).*mean return \(last 20\)\s+(-?[\d.]+)\s+std\s+([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            rets.append(float(m.group(2)))
            stds.append(float(m.group(3)))
    if not steps:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(np.array(steps) / 1e3, rets, color="C0")
    ax[0].set_xlabel("environment steps (thousands)")
    ax[0].set_ylabel("mean episode return (last 20)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(np.array(steps) / 1e3, stds, color="C3")
    ax[1].set_xlabel("environment steps (thousands)")
    ax[1].set_ylabel(r"policy std $\sigma$")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/learning_curve.pdf")
    plt.close(fig)


def plot_trajectories(trajs_by_policy, env, seed_idx, name):
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    first = next(iter(trajs_by_policy.values()))[seed_idx]
    days = np.arange(len(first["storage"])) + 1
    tin = first["inflow_t"]
    mask = tin <= env.episode_days
    axes[0].plot(tin[mask], first["inflow_q"][mask], color="0.3", lw=1)
    axes[0].axhline(env.Q_safe, color="r", ls="--", lw=0.8, label=r"$Q_{safe}$")
    axes[0].set_ylabel("inflow [m$^3$/d]")
    axes[0].legend(loc="upper right")
    for i, (label, trajs) in enumerate(trajs_by_policy.items()):
        tr = trajs[seed_idx]
        axes[1].plot(np.r_[0, days], np.r_[np.nan, tr["storage"]] / env.S_cap, color=f"C{i}", label=label)
        axes[2].step(days, tr["release"], where="pre", color=f"C{i}", label=label)
        axes[3].step(days, tr["spill"], where="pre", color=f"C{i}", label=label)
    axes[1].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[1].set_ylabel(r"storage $S/S_{cap}$")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[2].step(days, first["demand"], where="pre", color="k", ls=":", lw=1, label="demand")
    axes[2].axhline(env.Q_safe, color="r", ls="--", lw=0.8)
    axes[2].set_ylabel("release [m$^3$/d]")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[3].set_ylabel("spill [m$^3$/d]")
    axes[3].set_xlabel("day")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)


def plot_policy_map(model, env):
    """Release chosen by PPO as a function of storage and 1-day forecast."""
    S = np.linspace(0, 1.05, 80)
    F = np.linspace(0, 4, 80)
    Z = np.zeros((len(F), len(S)))
    D = env.demand_mean / env.Q_safe
    for i, f in enumerate(F):
        obs = np.array([[s, f, f, f, 0.6 * f, D, 0.5] for s in S], dtype=np.float32)
        Z[i] = 0.5 * (model.act(obs, deterministic=True)[:, 0] + 1) * env.Q_release_max
    fig, ax = plt.subplots(figsize=(5.5, 4))
    cs = ax.contourf(S, F * env.Q_safe, Z, levels=20, cmap="viridis")
    ax.contour(S, F * env.Q_safe, Z, levels=[env.demand_mean, env.Q_safe], colors=["w", "r"],
               linewidths=1)
    fig.colorbar(cs, label="PPO release [m$^3$/d]")
    ax.set_xlabel(r"storage $S/S_{cap}$")
    ax.set_ylabel("forecast inflow, next day [m$^3$/d]")
    ax.set_title(r"white: release = mean demand, red: release = $Q_{safe}$", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/policy_map.pdf")
    plt.close(fig)


def plot_threshold_tuning(tune, best):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(list(tune), list(tune.values()), "o-")
    ax.axvline(best, color="C3", ls="--", lw=0.8)
    ax.set_xlabel(r"flood-rule threshold [fraction of $S_{cap}$]")
    ax.set_ylabel("mean return (tuning seeds)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/threshold_tuning.pdf")
    plt.close(fig)


def plot_return_distribution(summaries):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = list(summaries)
    ax.boxplot([summaries[k]["ret"] for k in labels], tick_labels=labels, showmeans=True)
    ax.set_ylabel("episode return")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIG}/return_boxplot.pdf")
    plt.close(fig)


def write_table(summaries, path):
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Policy & Return & Supply & Flood & Spill & Terminal & Deficit [\%] & Flood days \\",
             r"\midrule"]
    for k, s in summaries.items():
        lines.append(
            f"{k} & ${s['ret'].mean():.2f}\\pm{s['ret'].std():.2f}$ & {s['supply'].mean():.2f} & "
            f"{s['flood'].mean():.2f} & {s['spill'].mean():.2f} & {s['terminal'].mean():.2f} & "
            f"{100 * s['deficit_frac'].mean():.1f} & {s['flood_days'].mean():.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main(args):
    os.makedirs(FIG, exist_ok=True)
    env = OHQReservoirEnv()
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    model.load_state_dict(torch.load(args.policy))
    model.eval()

    # Tune the rule-curve flood threshold on separate seeds (not the test set).
    tune_seeds = [20_000 + i for i in range(args.episodes)]
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    tune = {th: np.mean([run_episode(env, lambda e, o, th=th: rule_curve_policy(e, th), s)["return"]
                         for s in tune_seeds]) for th in thresholds}
    best_th = max(tune, key=tune.get)
    print("rule-curve threshold tuning:", {k: round(v, 2) for k, v in tune.items()}, "best", best_th)
    plot_threshold_tuning(tune, best_th)
    with open("report/tuning_summary.tex", "w") as f:
        f.write(f"{best_th:.1f}")

    policies = {
        "Demand only": lambda e, o: demand_only_policy(e),
        "Rule curve (0.8)": lambda e, o: rule_curve_policy(e),
        f"Tuned rule ({best_th:.1f})": lambda e, o: rule_curve_policy(e, best_th),
        "PPO": lambda e, o: model.act(o, deterministic=True),
    }
    seeds = [10_000 + i for i in range(args.episodes)]
    trajs = {k: [run_episode(env, p, s) for s in seeds] for k, p in policies.items()}
    summaries = {k: summarize(v, env) for k, v in trajs.items()}

    for k, s in summaries.items():
        print(f"{k:12s} return {s['ret'].mean():7.2f} +- {s['ret'].std():5.2f}  "
              f"supply {s['supply'].mean():6.2f}  flood {s['flood'].mean():6.2f}  "
              f"spill {s['spill'].mean():6.2f}  terminal {s['terminal'].mean():6.2f}  "
              f"deficit {100 * s['deficit_frac'].mean():5.1f}%  flood days {s['flood_days'].mean():4.1f}")
    write_table(summaries, "report/results_table.tex")

    # Paired comparison on identical storm sequences.
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"baseline & mean(PPO $-$ baseline) & std.\ error & PPO better in \\", r"\midrule"]
    for base in [k for k in summaries if k != "PPO"]:
        diff = summaries["PPO"]["ret"] - summaries[base]["ret"]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        wins = int((diff > 0).sum())
        print(f"PPO - {base:16s}: {diff.mean():6.2f} +- {se:4.2f} (s.e.), PPO better in {wins}/{len(diff)}")
        lines.append(f"{base} & ${diff.mean():+.2f}$ & {se:.2f} & {wins}/{len(diff)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open("report/paired_summary.tex", "w").write("\n".join(lines) + "\n")

    # Episode where PPO gains most over the rule curve, and a typical one.
    gain = summaries["PPO"]["ret"] - summaries["Rule curve (0.8)"]["ret"]
    plot_trajectories(trajs, env, int(np.argmax(gain)), "trajectory_best_gain")
    plot_trajectories(trajs, env, int(np.argsort(gain)[len(gain) // 2]), "trajectory_median")
    plot_return_distribution(summaries)
    plot_policy_map(model, env)
    plot_learning_curve(args.log)
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="results/ppo_reservoir.pt")
    p.add_argument("--log", default="results/train_log.txt")
    p.add_argument("--episodes", type=int, default=30)
    main(p.parse_args())
