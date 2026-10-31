"""
Verification figures for the report:
  1. OHQ output time stamping (the one-step lag in the current OHQLib build)
  2. Per-step cost breakdown of the OHQ backend
  3. Example stochastic inflow realizations
  4. Daily mass balance over an episode

    python diagnostics.py
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ohq_reservoir_env import OHQReservoirEnv, OHQ_RESOURCES, _import_ohq, rule_curve_policy

FIG = "report/figures"


def raw_run(workdir, dt, grow, Qin=1000.0, Qout=150.0, S0=1000.0):
    """1-day run with constant inflow; returns raw (label, storage) rows."""
    ohq = _import_ohq()
    R, W = OHQ_RESOURCES, workdir + "/"
    with open(W + "const.txt", "w") as f:
        f.write(f"0, {Qin}\n10, {Qin}\n")
    s = ohq.System()
    s.set_default_template_path(R)
    s.set_working_folder(W)
    s.set_silent(True)
    s.load_template(R + "main_components.json")
    s.add_template(R + "Reservoir_operation.json")
    for k, v in {"simulation_start_time": "0", "simulation_end_time": "1",
                 "initial_time_step": f"{dt}", "n_threads": "1",
                 "max_timestep_increase_factor": "50" if grow else "1",
                 "alloutputfile": W + "raw.txt"}.items():
        s.set_system_value(k, v)
    s.create_block("Reservoir", "Reservoir", {"Storage": f"{S0}", "inflow": W + "const.txt"})
    s.create_block("River", "fixed_head", {"head": "0", "Storage": "0"})
    s.create_link("outlet", "Reservoir_link_rule", "Reservoir", "River",
                  {"Q_min": "0", "Q_max": f"{Qout}", "S_min": "0", "S_max": "50"})
    s.finalize()
    s.solve(True, False)
    s.write_outputs()
    d = np.loadtxt(W + "raw.txt", delimiter=",", skiprows=1, ndmin=2)
    return d[:, 0], d[:, 1]


def fig_stamping(env):
    W = env.backend.workdir
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    exact = lambda t: 1000 + 850 * t
    for a, grow, title in [(ax[0], True, "default: dt grows up to 50$\\times$"),
                           (ax[1], False, "dt held fixed (used in the env)")]:
        lab, S = raw_run(W, 0.05, grow)
        a.plot(lab, S - exact(lab), "o-", ms=3, label="row stamp as written")
        a.plot(lab[1:], S[:-1] - exact(lab[1:]), "s-", ms=3, label="stamp = next row's stamp")
        a.axhline(0, color="k", lw=0.8)
        a.set_title(title, fontsize=10)
        a.set_xlabel("time [d]")
        a.grid(alpha=0.3)
    ax[0].set_ylabel(r"$S_{OHQ} - S_{exact}$ [m$^3$]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/stamping_check.pdf")
    plt.close(fig)


def fig_timing(env):
    ohq = _import_ohq()
    R, W = OHQ_RESOURCES, env.backend.workdir + "/"
    n = 50
    T = dict(construct_and_templates=0.0, build_model=0.0, solve=0.0, write_and_read=0.0)
    for _ in range(n):
        t = time.perf_counter()
        s = ohq.System()
        s.set_default_template_path(R)
        s.set_working_folder(W)
        s.set_silent(True)
        s.load_template(R + "main_components.json")
        s.add_template(R + "Reservoir_operation.json")
        T["construct_and_templates"] += time.perf_counter() - t
        t = time.perf_counter()
        for k, v in {"simulation_start_time": "10", "simulation_end_time": "11",
                     "initial_time_step": "0.05", "n_threads": "1",
                     "max_timestep_increase_factor": "1", "alloutputfile": W + "timing.txt"}.items():
            s.set_system_value(k, v)
        s.create_block("Reservoir", "Reservoir", {"Storage": "20000", "inflow": env.backend.inflow_file})
        s.create_block("River", "fixed_head", {"head": "0", "Storage": "0"})
        s.create_block("SpillSink", "fixed_head", {"head": "0", "Storage": "0"})
        s.create_link("outlet", "Reservoir_link_rule", "Reservoir", "River",
                      {"Q_min": "0", "Q_max": "1500", "S_min": "0", "S_max": "200"})
        s.create_link("spillway", "Reservoir_link_rule", "Reservoir", "SpillSink",
                      {"Q_min": "0", "Q_max": "50000", "S_min": "50000", "S_max": "52500"})
        s.finalize()
        T["build_model"] += time.perf_counter() - t
        t = time.perf_counter()
        s.solve(True, False)
        T["solve"] += time.perf_counter() - t
        t = time.perf_counter()
        s.write_outputs()
        np.loadtxt(W + "timing.txt", delimiter=",", skiprows=1)
        T["write_and_read"] += time.perf_counter() - t
    T = {k: 1e3 * v / n for k, v in T.items()}
    fig, ax = plt.subplots(figsize=(6, 2.6))
    labels = ["System() + templates", "build blocks/links", "solve (20 steps)", "write + parse output"]
    ax.barh(labels, list(T.values()), color=["C1", "C2", "C0", "C4"])
    for i, v in enumerate(T.values()):
        ax.text(v, i, f" {v:.2f} ms", va="center", fontsize=9)
    ax.set_xlabel("wall time per RL step [ms]")
    ax.set_xlim(0, max(T.values()) * 1.35)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{FIG}/timing_breakdown.pdf")
    plt.close(fig)
    return T


def fig_forcing(env):
    fig, ax = plt.subplots(3, 1, figsize=(9, 5.5), sharex=True)
    for a, seed in zip(ax, [3, 17, 42]):
        env.reset(seed=seed)
        m = env.inflow_t <= env.episode_days
        a.plot(env.inflow_t[m], env.inflow_q[m], color="0.3", lw=1, label="inflow")
        a.step(np.arange(env.episode_days), env.demand[:env.episode_days], where="post",
               color="C0", lw=1, label="demand $D(t)$")
        a.axhline(env.Q_safe, color="r", ls="--", lw=0.8, label=r"$Q_{safe}$")
        a.set_ylabel("m$^3$/d")
        a.set_title(f"seed {seed}", fontsize=9, loc="left")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="upper right", ncol=3)
    ax[-1].set_xlabel("day")
    fig.tight_layout()
    fig.savefig(f"{FIG}/forcing_examples.pdf")
    plt.close(fig)


def fig_mass_balance(env):
    env.reset(seed=1)
    err, inflow_vol = [], []
    while True:
        S0, d = env.storage, env.day
        _, _, term, _, info = env.step(rule_curve_policy(env))
        m = (env.inflow_t >= d - 1e-9) & (env.inflow_t <= d + 1 + 1e-9)
        vin = np.trapezoid(env.inflow_q[m], dx=env.dt)
        err.append(S0 + vin - info["release"] - info["spill"] - info["storage"])
        inflow_vol.append(vin)
        if term:
            break
    err, inflow_vol = np.array(err), np.array(inflow_vol)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(np.arange(len(err)), 100 * err / inflow_vol, color="C0")
    ax.set_xlabel("day")
    ax.set_ylabel("daily balance error\n[% of inflow volume]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/mass_balance.pdf")
    plt.close(fig)
    return np.abs(err).max(), err.sum(), inflow_vol.sum()


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    env = OHQReservoirEnv()
    print("rows lag one step:", env.backend.rows_lag_one_step)
    env.reset(seed=0)  # writes an inflow file used by the timing test
    fig_stamping(env)
    T = fig_timing(env)
    print("timing [ms]:", {k: round(v, 2) for k, v in T.items()}, "total", round(sum(T.values()), 2))
    fig_forcing(env)
    mx, cum, vin = fig_mass_balance(env)
    print(f"mass balance: max |daily err| {mx:.1f} m3, cumulative {cum:.1f} m3 "
          f"of {vin:.0f} m3 inflow ({100 * cum / vin:.3f}%)")
    env.close()

    # Numeric summaries pulled into report/main.tex
    vin_tex = f"{vin:,.0f}".replace(",", "{,}")
    with open("report/mass_balance_summary.tex", "w") as f:
        f.write(f"the largest daily residual is {mx:.0f}~m$^3$, while the cumulative "
                f"residual over the season is {cum:.1f}~m$^3$ out of "
                f"{vin_tex}~m$^3$ of inflow ({100 * abs(cum) / vin:.3f}\\%).\n")
    total = sum(T.values())
    overhead = total - T["solve"]
    with open("report/timing_summary.tex", "w") as f:
        f.write(
            f"One step costs about {total:.1f}~ms, but the numerical solve (20 implicit "
            f"steps of 0.05~d) accounts for only {T['solve']:.1f}~ms. The remaining "
            f"{overhead:.1f}~ms ({100 * overhead / total:.0f}\\%) is overhead: constructing a "
            f"\\code{{System}} and parsing the JSON templates ({T['construct_and_templates']:.1f}~ms), "
            f"building blocks and links ({T['build_model']:.1f}~ms), and writing and parsing the "
            f"output file ({T['write_and_read']:.1f}~ms). The overhead is a fixed per-step cost, "
            f"independent of how hard the physics is. A codegen-compiled model that is built once "
            f"and only has its state reset between steps would remove nearly all of it, giving "
            f"roughly a {total / T['solve']:.0f}$\\times$ speed-up for this model even before any "
            f"gain in the solver itself.\n")
