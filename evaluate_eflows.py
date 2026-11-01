"""
Evaluate operating policies for the environmental-flow problem (ohq_eflows_env).

    python evaluate_eflows.py --tune                      # tune the rule baselines
    python evaluate_eflows.py --select "results/ppo_eflows_ffreward_u*.pt" --scenario ffreward
    python evaluate_eflows.py --compare                   # test-period comparison, tables, figures

Windows are 3 water years. Tuning and checkpoint selection use training-period
windows; the comparison uses WY2017-2025 (three windows starting 2017, 2020,
2023), each started from the observed Oct 1 storage. The forecast errors are
seeded per window, so every policy sees the same forecasts.

Every run reports, per water year: irrigation delivered (% of demand), flood
days (river > Q_safe), energy, and the functional-flow score. The common
yardstick J_ff (mean reward per year under the ffreward weights: supply +
flood + functional flows) is computed for every policy, whatever scenario it
was operated under.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import glob
import json
import shutil
from multiprocessing import Pool

import numpy as np

import functional_flows as ff

RESULTS = "results"
FIG = "report_eflows/figures"
VALID_STARTS = (1990, 1996, 2002, 2008, 2012)     # training-period windows (tuning, selection)
TEST_STARTS = (2017, 2020, 2023)
HIST_STARTS = (1989, 1992, 1995, 1998, 2001, 2004, 2007)   # river gauge at Snelling ends 2010
MAY1 = 212


def make_policy(spec, env):
    kind = spec[0]
    if kind == "rule":
        from ohq_eflows_env import sop_rule
        kw = spec[1]
        return lambda obs: sop_rule(env, **kw)
    if kind == "ppo":
        import torch
        from train_ppo import ActorCritic
        model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
        model.load_state_dict(torch.load(spec[1]))
        return lambda obs: model.act(obs[None], deterministic=True)[0]
    raise ValueError(kind)


def run_window(job):
    """job = (scenario, policy spec, start water year, seed). Returns daily records."""
    from ohq_eflows_env import EflowsEnv
    scenario, spec, start, seed = job
    env = EflowsEnv(scenario=scenario, **(spec[2] if len(spec) > 2 else {}))
    S0 = env.oct1_storage.get(start, 0.6 * env.S_cap)
    obs, _ = env.reset(seed=seed, options={"start_year": start, "storage": S0})
    policy = make_policy(spec, env)
    rec, done = [], False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        info["reward"] = r
        rec.append(info)
        done = term or trunc
    out = {k: np.array([x[k] for x in rec]) for k in rec[0] if k != "year_type"}
    out["year_type"] = np.array([x["year_type"] for x in rec])
    out["may1_type"] = [env.declared[k][MAY1] for k in range(env.episode_years)]
    out["targets"] = env.targets
    out["S0"] = S0
    env.close()
    return out


def year_metrics(run):
    """Per-water-year metrics of one window."""
    rows = []
    for k, wy in enumerate(np.unique(run["wy"])):
        s = run["wy"] == wy
        typ = run["may1_type"][k]
        score, comp = ff.ff_score(run["river"][s], run["targets"][typ])
        rows.append(dict(
            wy=int(wy), type=typ, ff=score, comp=comp,
            supply=100.0 * run["canal"][s].sum() / run["demand"][s].sum(),
            shortage_hm3=run["deficit"][s].sum() * 86400 / 1e6,
            flood_days=int(np.sum(run["river"][s] > 170.0)),
            energy_gwh=run["power"][s].sum() * 24 / 1000.0,
            river_hm3=run["river"][s].sum() * 86400 / 1e6,
            J_ff=float(np.sum(run["r_supply"][s] + run["r_flood"][s] + run["r_ff"][s])),
            reward=float(np.sum(run["reward"][s])),
            r_supply=float(np.sum(run["r_supply"][s])), r_flood=float(np.sum(run["r_flood"][s])),
            r_ff=float(np.sum(run["r_ff"][s])), r_hp=float(np.sum(run["r_hp"][s])),
            end_storage=float(run["storage"][s][-1])))
    return rows


def summarize(rows):
    keys = ("ff", "supply", "shortage_hm3", "flood_days", "energy_gwh", "river_hm3", "J_ff", "reward",
            "r_supply", "r_flood", "r_ff", "r_hp")
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["min_supply"] = float(min(r["supply"] for r in rows))
    for t in ff.YEAR_TYPES:
        v = [r["ff"] for r in rows if r["type"] == t]
        out[f"ff_{t}"] = float(np.mean(v)) if v else float("nan")
    return out


# ------------------------------------------------------------------ tuning
RULE_GRID = [dict(hedge_storage=hs, hedge_fraction=hf, drawdown_days=dd)
             for hs in (0.2, 0.3, 0.4, 0.5) for hf in (0.5, 0.7, 0.85) for dd in (5.0, 15.0)]


def tune(pool):
    """Best rule parameters per scenario, by the scenario's own mean reward
    per year on the validation windows."""
    best = {}
    for scenario in ("minflow", "prescribed", "ffreward"):
        jobs = [(scenario, ("rule", kw), y, 0) for kw in RULE_GRID for y in VALID_STARTS]
        runs = pool.map(run_window, jobs, chunksize=1)
        n = len(VALID_STARTS)
        table = []
        for i, kw in enumerate(RULE_GRID):
            rows = sum((year_metrics(r) for r in runs[i * n:(i + 1) * n]), [])
            table.append(dict(kw=kw, **summarize(rows)))
        table.sort(key=lambda r: -r["reward"])
        best[scenario] = table[0]["kw"]
        print(f"{scenario}: best {table[0]['kw']}  reward/yr {table[0]['reward']:.2f}  "
              f"supply {table[0]['supply']:.1f}%  ff {table[0]['ff']:.3f}", flush=True)
    json.dump(best, open(os.path.join(RESULTS, "eflows_rule_params.json"), "w"), indent=1)
    return best


def rule_params():
    return json.load(open(os.path.join(RESULTS, "eflows_rule_params.json")))


# --------------------------------------------------------------- selection
def select(pool, pattern, scenario, out, env_kwargs):
    ckpts = sorted(glob.glob(pattern))
    rule = ("rule", rule_params()[scenario if scenario != "multi" else "ffreward"])
    specs = [rule] + [("ppo", c, env_kwargs) for c in ckpts]
    jobs = [(scenario, sp, y, 0) for sp in specs for y in VALID_STARTS]
    runs = pool.map(run_window, jobs, chunksize=1)
    n = len(VALID_STARTS)
    table = []
    for i, sp in enumerate(specs):
        rows = sum((year_metrics(r) for r in runs[i * n:(i + 1) * n]), [])
        s = summarize(rows)
        table.append(dict(policy=sp[1] if sp[0] == "ppo" else "rule", **s))
        print(f"{os.path.basename(str(table[-1]['policy'])):34s} reward/yr {s['reward']:8.2f}  "
              f"supply {s['supply']:5.1f}%  ff {s['ff']:.3f}  flood d/yr {s['flood_days']:.1f}", flush=True)
    best = max(table[1:], key=lambda r: r["reward"])
    json.dump(dict(table=table, best=best["policy"]),
              open(os.path.join(RESULTS, f"eflows_{scenario}_selection.json"), "w"), indent=1)
    shutil.copy(best["policy"], out)
    # Remember how the policy's actions are interpreted (e.g. residual=True).
    json.dump(env_kwargs, open(out.replace(".pt", "_env.json"), "w"))
    print(f"best: {best['policy']} (reward/yr {best['reward']:.2f}) -> {out}")


# --------------------------------------------------------------- comparison
def natural_rows(starts):
    """The unimpaired flow itself as the river (no reservoir, no supply)."""
    from ohq_eflows_env import EflowsEnv
    env = EflowsEnv(scenario="ffreward", backend=object())
    rows = []
    for start in starts:
        for wy in range(start, start + 3):
            typ = env.classify(env.volume[wy])
            score, comp = ff.ff_score(env.years[wy], env.targets[typ])
            rows.append(dict(wy=wy, type=typ, ff=score, comp=comp))
    return rows


def historical_rows():
    """Observed river flow near Snelling (below the diversions), WY1989-2009."""
    from ohq_eflows_env import EflowsEnv
    env = EflowsEnv(scenario="ffreward", backend=object())
    d = np.load("data/merced_daily.npz")
    river = ff.split_water_years(d["date"], d["river"])
    rows = []
    for start in HIST_STARTS:
        for wy in range(start, start + 3):
            q = river.get(wy)
            if q is None or np.isnan(q).all():
                continue
            if np.isnan(q).any():
                q = np.interp(np.arange(365), np.flatnonzero(~np.isnan(q)), q[~np.isnan(q)])
            typ = env.classify(env.volume[wy])
            score, comp = ff.ff_score(q, env.targets[typ])
            rows.append(dict(wy=wy, type=typ, ff=score, comp=comp, river_hm3=q.sum() * 86400 / 1e6))
    return rows


def policies():
    rp = rule_params()
    pol = [("MIF rule", "minflow", ("rule", rp["minflow"])),
           ("Prescribed FF rule", "prescribed", ("rule", rp["prescribed"]))]
    for name, scen in (("PPO, FF in reward", "ffreward"), ("PPO, FF + hydropower", "multi")):
        path = os.path.join(RESULTS, f"ppo_eflows_{scen}.pt")
        if os.path.exists(path):
            side = path.replace(".pt", "_env.json")
            kw = json.load(open(side)) if os.path.exists(side) else {}
            pol.append((name, scen, ("ppo", path, kw)))
    return pol


def compare(pool):
    pol = policies()
    out = {}
    for starts, label in ((TEST_STARTS, "test"), (HIST_STARTS, "hist")):
        jobs = [(scen, spec, y, 0) for _, scen, spec in pol for y in starts]
        runs = pool.map(run_window, jobs, chunksize=1)
        n = len(starts)
        res = {}
        for i, (name, scen, spec) in enumerate(pol):
            rr = runs[i * n:(i + 1) * n]
            rows = sum((year_metrics(r) for r in rr), [])
            res[name] = dict(summary=summarize(rows), years=rows, runs=rr)
        out[label] = res
    out["natural"] = natural_rows(TEST_STARTS)
    out["natural_hist"] = natural_rows(HIST_STARTS)
    out["historical"] = historical_rows()
    return out


def fmt(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_tables(out):
    os.makedirs("report_eflows", exist_ok=True)
    lines = []
    for name, r in out["test"].items():
        s = r["summary"]
        lines.append(f"{name} & {fmt(s['supply'])} & {fmt(s['min_supply'])} & {fmt(s['flood_days'])} & "
                     f"{fmt(s['energy_gwh'], 0)} & {fmt(s['river_hm3'], 0)} & {fmt(s['ff'], 2)} & "
                     f"{fmt(s['ff_dry'], 2)} & {fmt(s['ff_moderate'], 2)} & {fmt(s['ff_wet'], 2)} & "
                     f"{fmt(s['J_ff'])} \\\\")
    nat = out["natural"]
    nff = lambda t: np.mean([r["ff"] for r in nat if r["type"] == t] or [np.nan])
    lines.append(f"Unimpaired flow (no dam) & -- & -- & -- & -- & -- & {np.mean([r['ff'] for r in nat]):.2f} & "
                 f"{fmt(nff('dry'), 2)} & {fmt(nff('moderate'), 2)} & {fmt(nff('wet'), 2)} & -- \\\\")
    open("report_eflows/table_test.tex", "w").write("\n".join(lines) + "\n")

    lines = []
    for name, r in out["test"].items():
        for y in r["years"]:
            c = y["comp"]
            lines.append(f"{name} & {y['wy']} & {y['type']} & {fmt(y['supply'])} & {y['flood_days']} & "
                         f"{fmt(c.get('FA'), 2)} & {fmt(c.get('WB'), 2)} & {fmt(c.get('PK'), 2)} & "
                         f"{fmt(c.get('SP'), 2)} & {fmt(c.get('DS'), 2)} & {fmt(y['ff'], 2)} \\\\")
        lines.append("\\midrule")
    open("report_eflows/table_years.tex", "w").write("\n".join(lines[:-1]) + "\n")

    hist, sim = out["historical"], out["hist"]["MIF rule"]["years"]
    lines = []
    for h, s in zip(hist, sim):
        lines.append(f"{h['wy']} & {h['type']} & {h['river_hm3']:.0f} & {s['river_hm3']:.0f} & "
                     f"{h['ff']:.2f} & {s['ff']:.2f} \\\\")
    open("report_eflows/table_hist.tex", "w").write("\n".join(lines) + "\n")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    names = list(out["test"])
    colors = dict(zip(names, ["tab:gray", "tab:blue", "tab:red", "tab:purple"]))
    # Hydrographs of each policy, test windows.
    for w, start in enumerate(TEST_STARTS):
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        r0 = out["test"][names[0]]["runs"][w]
        days = np.arange(len(r0["river"]))
        ax[0].semilogy(days, np.maximum(r0["inflow"], 0.1), color="k", lw=0.8, label="unimpaired inflow")
        for name in names:
            r = out["test"][name]["runs"][w]
            ax[0].semilogy(days, np.maximum(r["river"], 0.1), color=colors[name], lw=0.9, label=name)
            ax[1].plot(days, r["storage"], color=colors[name], lw=1.0)
            ax[2].plot(days, r["canal"], color=colors[name], lw=0.9)
        ax[2].plot(days, r0["demand"], "k--", lw=0.8, label="demand")
        ax[0].set_ylabel("river flow [m$^3$/s]")
        ax[1].set_ylabel("storage [hm$^3$]")
        ax[2].set_ylabel("canal delivery [m$^3$/s]")
        ax[2].set_xlabel(f"day since Oct 1, {start - 1}")
        for k in range(1, 3):
            ax[0].axvline(365 * k, color="0.7", lw=0.6)
        ax[0].legend(fontsize=7, ncol=3, loc="upper right")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{FIG}/hydrograph_{start}.pdf")
        plt.close(fig)
    # Trade-off: irrigation supply vs functional-flow score.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in names:
        s = out["test"][name]["summary"]
        ax.scatter(s["supply"], s["ff"], s=60, color=colors[name], label=name, zorder=3)
    nat = np.mean([r["ff"] for r in out["natural"]])
    ax.axhline(nat, color="k", ls=":", lw=0.8, label="unimpaired flow")
    ax.set_xlabel("irrigation delivered [% of demand], WY2017-2025")
    ax.set_ylabel("functional-flow score")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/tradeoff.pdf")
    plt.close(fig)


def plot_training(scenario="ffreward"):
    """Training curve (reward per simulated day, stochastic policy) and the
    deterministic validation reward of every saved checkpoint."""
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = os.path.join(RESULTS, f"train_log_eflows_{scenario}.txt")
    sel = os.path.join(RESULTS, f"eflows_{scenario}_selection.json")
    if not (os.path.exists(log) and os.path.exists(sel)):
        return
    u, r = [], []
    for line in open(log):
        m = re.search(r"update\s+(\d+)/.*reward/step\s+(-?[\d.]+)", line)
        if m:
            u.append(int(m.group(1)))
            r.append(float(m.group(2)))
    u, r = np.array(u), np.array(r)
    table = json.load(open(sel))["table"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    k = 10
    ax[0].plot(u, r, color="0.8", lw=0.6)
    if len(r) > k:
        ax[0].plot(u[k - 1:], np.convolve(r, np.ones(k) / k, "valid"), color="tab:red")
    ax[0].set_xlabel("PPO update")
    ax[0].set_ylabel("reward per day (training, with noise)")
    ax[0].grid(alpha=0.3)
    rule = table[0]["reward"]
    cks = [(int(re.search(r"_u(\d+)", t["policy"]).group(1)), t["reward"], t["ff"]) for t in table[1:]]
    ax[1].plot([c[0] for c in cks], [c[1] for c in cks], "o-", color="tab:red", label="PPO checkpoint")
    ax[1].axhline(rule, color="tab:blue", ls="--", label="tuned prescriptive rule")
    ax[1].set_xlabel("PPO update (0 = cloned rule)")
    ax[1].set_ylabel("validation reward per year")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/training_{scenario}.pdf")
    plt.close(fig)


def main(args):
    with Pool(args.workers) as pool:
        if args.tune:
            tune(pool)
        if args.select:
            select(pool, args.select, args.scenario, args.out or f"{RESULTS}/ppo_eflows_{args.scenario}.pt",
                   json.loads(args.env_kwargs))
        if args.compare:
            out = compare(pool)
            for label in ("test", "hist"):
                print(f"--- {label}")
                for name, r in out[label].items():
                    s = r["summary"]
                    print(f"{name:24s} supply {s['supply']:5.1f}% (min {s['min_supply']:5.1f})  "
                          f"flood d/yr {s['flood_days']:4.1f}  GWh/yr {s['energy_gwh']:5.0f}  "
                          f"river hm3/yr {s['river_hm3']:5.0f}  FF {s['ff']:.3f}  J_ff {s['J_ff']:7.2f}")
            print("natural FF (test)", np.mean([r["ff"] for r in out["natural"]]))
            print("historical FF (WY1989-2009)", np.mean([r["ff"] for r in out["historical"]]))
            write_tables(out)
            plot(out)
            for scen in ("ffreward", "multi"):
                plot_training(scen)
            slim = {lab: {n: dict(summary=r["summary"], years=r["years"]) for n, r in out[lab].items()}
                    for lab in ("test", "hist")}
            slim.update(natural=out["natural"], historical=out["historical"])
            json.dump(slim, open(os.path.join(RESULTS, "eflows_comparison.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true")
    p.add_argument("--select", default="")
    p.add_argument("--scenario", default="ffreward")
    p.add_argument("--out", default="")
    p.add_argument("--env-kwargs", default="{}", help="env options of the PPO policies, e.g. '{\"residual\": true}'")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--workers", type=int, default=30)
    main(p.parse_args())
