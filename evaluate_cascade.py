"""
Evaluate PPO on the two-reservoir cascade against rule-based baselines on
held-out storm seasons (same seeds for every policy), tune the rule's two
thresholds on separate seeds, and write figures/tables for the report.

    python evaluate_cascade.py --policy results/ppo_cascade.pt --episodes 30
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import re
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "report/figures"
ETAS = np.round(np.arange(0.1, 0.91, 0.1), 2)


def _policy(name, params, policy_path):
    """Build a policy callable inside a worker process."""
    from ohq_cascade_env import cascade_rule_policy
    if name == "PPO":
        import torch
        torch.set_num_threads(1)
        from train_ppo import ActorCritic
        model = ActorCritic(16, 2)
        model.load_state_dict(torch.load(policy_path))
        model.eval()
        return lambda env, obs: model.act(obs, deterministic=True)
    eta_A, eta_B = params
    return lambda env, obs: cascade_rule_policy(env, eta_A, eta_B)


def run_episode(job):
    """job = (policy name, params, seed, policy path, keep trajectory)."""
    from ohq_cascade_env import CascadeEnv
    name, params, seed, policy_path, keep = job
    env = CascadeEnv()
    policy = _policy(name, params, policy_path)
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
    if keep:
        traj["inflow_t"] = env.inflow_t
        traj["inflow"] = {k: v.copy() for k, v in env.inflow.items()}
    else:
        traj = {k: traj[k] for k in ("return", "r_supply", "r_flood", "r_spill", "deficit",
                                     "demand", "river", "spill_A", "spill_B")}
    env.close()
    return traj


def summarize(trajs, Q_safe):
    g = lambda f: np.array([f(t) for t in trajs])
    s = dict(ret=g(lambda t: t["return"]), supply=g(lambda t: t["r_supply"].sum()),
             flood=g(lambda t: t["r_flood"].sum()), spill=g(lambda t: t["r_spill"].sum()),
             deficit_frac=g(lambda t: t["deficit"].sum() / t["demand"].sum()),
             spill_vol=g(lambda t: (t["spill_A"] + t["spill_B"]).sum()),
             flood_days=g(lambda t: (t["river"] > Q_safe + 1e-6).sum()))
    s["terminal"] = s["ret"] - s["supply"] - s["flood"] - s["spill"]
    return s


def write_tables(summaries, prefix):
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Policy & Return & Supply & Flood & Spill & Terminal & Deficit [\%] & Flood days \\",
             r"\midrule"]
    for k, s in summaries.items():
        lines.append(f"{k} & ${s['ret'].mean():.2f}\\pm{s['ret'].std():.2f}$ & {s['supply'].mean():.2f} & "
                     f"{s['flood'].mean():.2f} & {s['spill'].mean():.2f} & {s['terminal'].mean():.2f} & "
                     f"{100 * s['deficit_frac'].mean():.1f} & {s['flood_days'].mean():.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"report/{prefix}_results_table.tex", "w").write("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"baseline & mean(PPO $-$ baseline) & std.\ error & PPO better in \\", r"\midrule"]
    for base in [k for k in summaries if k != "PPO"]:
        diff = summaries["PPO"]["ret"] - summaries[base]["ret"]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        wins = int((diff > 0).sum())
        print(f"PPO - {base:22s}: {diff.mean():6.2f} +- {se:4.2f} (s.e.), PPO better in {wins}/{len(diff)}")
        lines.append(f"{base} & ${diff.mean():+.2f}$ & {se:.2f} & {wins}/{len(diff)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"report/{prefix}_paired_summary.tex", "w").write("\n".join(lines) + "\n")


def plot_tuning(grid, best):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(grid.T, origin="lower", cmap="viridis",
                   extent=[ETAS[0] - 0.05, ETAS[-1] + 0.05, ETAS[0] - 0.05, ETAS[-1] + 0.05])
    ax.plot(*best, "r*", ms=14)
    fig.colorbar(im, label="mean return (tuning seeds)")
    ax.set_xlabel(r"threshold $\eta_A$ (fraction of $S^A_{cap}$)")
    ax.set_ylabel(r"threshold $\eta_B$ (fraction of $S^B_{cap}$)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/cascade_threshold_tuning.pdf")
    plt.close(fig)


def plot_trajectories(trajs_by_policy, seed_idx, name, env_consts):
    cap_A, cap_B, Q_safe, horizon = env_consts
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    first = next(iter(trajs_by_policy.values()))[seed_idx]
    days = np.arange(len(first["S_A"])) + 1
    t = first["inflow_t"]
    m = t <= horizon
    for k, c in (("A", "C0"), ("B", "C1"), ("trib", "C2")):
        axes[0].plot(t[m], first["inflow"][k][m], color=c, lw=1, label={"trib": "tributary"}.get(k, f"inflow {k}"))
    axes[0].axhline(Q_safe, color="r", ls="--", lw=0.8, label=r"$Q_{safe}$")
    axes[0].set_ylabel("inflow [m$^3$/d]")
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    for i, (label, trajs) in enumerate(trajs_by_policy.items()):
        tr = trajs[seed_idx]
        c = f"C{i}"
        axes[1].plot(days, tr["S_A"] / cap_A, color=c, label=label)
        axes[1].plot(days, tr["S_B"] / cap_B, color=c, ls=":")
        axes[2].step(days, tr["Q_A"], where="pre", color=c, ls=":")
        axes[2].step(days, tr["Q_B"], where="pre", color=c, label=label)
        axes[3].step(days, tr["river"], where="pre", color=c, label=label)
        axes[4].step(days, tr["spill_A"] + tr["spill_B"], where="pre", color=c, label=label)
    axes[1].set_ylabel("storage / capacity\n(solid A, dotted B)")
    axes[1].axhline(1, color="k", ls=":", lw=0.6)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[2].set_ylabel("release [m$^3$/d]\n(solid B, dotted A)")
    axes[3].step(days, first["demand"], where="pre", color="k", ls=":", lw=1, label="demand")
    axes[3].axhline(Q_safe, color="r", ls="--", lw=0.8)
    axes[3].set_ylabel("river flow at\nflood point [m$^3$/d]")
    axes[3].legend(fontsize=8, loc="upper left")
    axes[4].set_ylabel("spill A + B\n[m$^3$/d]")
    axes[4].set_xlabel("day")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)


def plot_box(summaries):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels = list(summaries)
    ax.boxplot([summaries[k]["ret"] for k in labels], tick_labels=labels, showmeans=True)
    ax.set_ylabel("episode return")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIG}/cascade_return_boxplot.pdf")
    plt.close(fig)


def plot_learning_curve(log_path):
    steps, rets, stds = [], [], []
    for line in open(log_path):
        mm = re.search(r"steps\s+(\d+).*mean return \(last 20\)\s+(-?[\d.]+)\s+std\s+([\d.]+)", line)
        if mm:
            steps.append(int(mm.group(1))); rets.append(float(mm.group(2))); stds.append(float(mm.group(3)))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(np.array(steps) / 1e3, rets)
    ax[0].set_xlabel("environment steps (thousands)")
    ax[0].set_ylabel("mean episode return (last 20)")
    ax[1].plot(np.array(steps) / 1e3, stds, color="C3")
    ax[1].set_xlabel("environment steps (thousands)")
    ax[1].set_ylabel(r"policy std $\sigma$")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/cascade_learning_curve.pdf")
    plt.close(fig)


def main(args):
    os.makedirs(FIG, exist_ok=True)
    from ohq_cascade_env import CascadeEnv
    probe = CascadeEnv()
    consts = (probe.cap_A, probe.cap_B, probe.Q_safe, probe.episode_days)
    probe.close()

    with Pool(args.workers) as pool:
        # 1) tune the rule's thresholds on separate seeds
        tune_seeds = [20_000 + i for i in range(args.episodes)]
        jobs = [("rule", (a, b), s, None, False) for a in ETAS for b in ETAS for s in tune_seeds]
        rets = np.array([t["return"] for t in pool.map(run_episode, jobs, chunksize=4)])
        grid = rets.reshape(len(ETAS), len(ETAS), len(tune_seeds)).mean(axis=2)
        ia, ib = np.unravel_index(np.argmax(grid), grid.shape)
        best = (float(ETAS[ia]), float(ETAS[ib]))
        print(f"tuned thresholds eta_A={best[0]} eta_B={best[1]}  (mean tuning return {grid[ia, ib]:.2f})")
        plot_tuning(grid, best)
        open("report/cascade_tuning_summary.tex", "w").write(
            f"$\\eta_A={best[0]:.1f}$, $\\eta_B={best[1]:.1f}$")

        # 2) test seeds, all policies
        policies = {"Supply only": ("rule", (1.0, 1.0)),
                    "Rule (0.6, 0.5)": ("rule", (0.6, 0.5)),
                    f"Tuned rule ({best[0]:.1f}, {best[1]:.1f})": ("rule", best),
                    "PPO": ("PPO", None)}
        seeds = [10_000 + i for i in range(args.episodes)]
        trajs = {}
        for label, (name, params) in policies.items():
            trajs[label] = pool.map(run_episode, [(name, params, s, args.policy, True) for s in seeds])

    summaries = {k: summarize(v, consts[2]) for k, v in trajs.items()}
    for k, s in summaries.items():
        print(f"{k:22s} J {s['ret'].mean():7.2f} +- {s['ret'].std():5.2f}  supply {s['supply'].mean():6.2f}  "
              f"flood {s['flood'].mean():6.2f}  spill {s['spill'].mean():6.2f}  terminal {s['terminal'].mean():5.2f}  "
              f"deficit {100 * s['deficit_frac'].mean():4.1f}%  spill vol {s['spill_vol'].mean():7.0f}  "
              f"flood days {s['flood_days'].mean():4.1f}")
    write_tables(summaries, "cascade")
    tuned_label = [k for k in summaries if k.startswith("Tuned")][0]
    gain = summaries["PPO"]["ret"] - summaries[tuned_label]["ret"]
    order = np.argsort(gain)
    plot_trajectories(trajs, int(order[-1]), "cascade_trajectory_best_gain", consts)
    plot_trajectories(trajs, int(order[len(order) // 2]), "cascade_trajectory_median", consts)
    plot_trajectories(trajs, int(order[0]), "cascade_trajectory_worst", consts)
    print("trajectory seeds: best", seeds[order[-1]], "median", seeds[order[len(order) // 2]],
          "worst", seeds[order[0]])
    plot_box(summaries)
    plot_learning_curve(args.log)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="results/ppo_cascade.pt")
    p.add_argument("--log", default="results/train_log_cascade.txt")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args())
