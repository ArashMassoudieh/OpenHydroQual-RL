"""
Figures and tables that describe the environmental-flow problem setup
(report_eflows): natural hydrographs and functional-flow targets by year
type, the prescribed functional-flow hydrographs, demand / minimum flow /
flood rule curve, and the target table.

    python plot_eflows_setup.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import functional_flows as ff
from ohq_eflows_env import (EflowsEnv, TRAIN_YEARS, demand_climatology, min_instream_flow,
                            top_of_conservation)

FIG = "report_eflows/figures"
MONTHS = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]
MSTART = [0, 31, 61, 92, 123, 151, 182, 212, 243, 273, 304, 335]


def month_axis(ax):
    ax.set_xticks(MSTART)
    ax.set_xticklabels(MONTHS)
    ax.set_xlim(0, 364)


def main():
    os.makedirs(FIG, exist_ok=True)
    env = EflowsEnv(backend=object())
    T = env.targets
    days = np.arange(365)
    typ = {y: env.classify(env.volume[y]) for y in env.years}

    # 1. Natural hydrographs by year type with the targets.
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, t in zip(axs, ff.YEAR_TYPES):
        ys = [y for y in TRAIN_YEARS if typ[y] == t]
        X = np.array([env.years[y] for y in ys])
        for x in X:
            ax.semilogy(days, np.maximum(x, 0.05), color="0.75", lw=0.5)
        ax.semilogy(days, np.median(X, 0), color="k", lw=1.2, label="median unimpaired")
        ax.semilogy(days, np.maximum(env.prescribed[t], 0.05), color="tab:red", lw=1.4,
                    label="prescribed FF hydrograph")
        ax.semilogy(days, min_instream_flow(t), color="tab:blue", lw=1.0, ls="--", label="minimum instream flow")
        ax.set_title(f"{t} years ({len(ys)})")
        month_axis(ax)
        ax.set_ylim(0.05, 2000)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("flow [m$^3$/s]")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/natural_targets.pdf")
    plt.close(fig)

    # 2. Demand, flood rule curve.
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(days, demand_climatology(), color="tab:green")
    ax[0].set_ylabel("irrigation demand [m$^3$/s]")
    month_axis(ax[0])
    ax[0].grid(alpha=0.3)
    ax[1].plot(days, [top_of_conservation(d, env.S_cap) for d in days], color="tab:orange",
               label="top of conservation pool")
    ax[1].axhline(env.S_cap, color="k", lw=0.8, label="capacity")
    ax[1].axhline(env.S_inactive, color="k", ls=":", lw=0.8, label="inactive pool")
    ax[1].set_ylabel("storage [hm$^3$]")
    ax[1].legend(fontsize=7, loc="lower right")
    month_axis(ax[1])
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/demand_rulecurve.pdf")
    plt.close(fig)

    # 3. Annual unimpaired volume and year types.
    fig, ax = plt.subplots(figsize=(10, 3))
    ys = sorted(env.years)
    col = {"dry": "tab:orange", "moderate": "tab:olive", "wet": "tab:blue"}
    ax.bar(ys, [env.volume[y] for y in ys], color=[col[typ[y]] for y in ys])
    for v in env.terciles:
        ax.axhline(v, color="k", lw=0.6, ls=":")
    ax.axhline(demand_climatology().sum() * 86400 / 1e6, color="tab:green", lw=1.0,
               label="annual irrigation demand")
    ax.axvline(2016.5, color="k", lw=1.0)
    ax.text(2016.8, 2700, "test", fontsize=8)
    ax.text(2014.6, 2700, "train", fontsize=8)
    ax.set_ylabel("unimpaired runoff [hm$^3$/yr]")
    ax.set_xlabel("water year")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in col.values()]
    ax.legend(handles + [ax.lines[-2]], list(col) + ["annual demand"], fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/annual_volumes.pdf")
    plt.close(fig)

    # 4. Target table.
    rows = []
    for key, label, unit in (("Q_FA", "Fall pulse magnitude $Q_{FA}$", "m$^3$/s"),
                             ("D_FA", "Fall pulse duration $D_{FA}$", "d"),
                             ("Q_WB", "Wet-season baseflow $Q_{WB}$", "m$^3$/s"),
                             ("Q_PK", "Peak flow magnitude $Q_{PK}$", "m$^3$/s"),
                             ("D_PK", "Peak flow duration $D_{PK}$", "d"),
                             ("Q_SP", "Spring pulse magnitude $Q_{SP}$", "m$^3$/s"),
                             ("D_SP", "Spring pulse duration $D_{SP}$", "d"),
                             ("ROC_max", "Max.\\ recession rate $r_{\\max}$", "1/d"),
                             ("Q_DS", "Dry-season baseflow $Q_{DS}$", "m$^3$/s")):
        vals = " & ".join(f"{T[t][key]:.2f}" if isinstance(T[t][key], float) else f"{T[t][key]}"
                          for t in ff.YEAR_TYPES)
        rows.append(f"{label} & {unit} & {vals} \\\\")
    vol = [env.prescribed[t].sum() * 86400 / 1e6 for t in ff.YEAR_TYPES]
    rows.append("\\midrule")
    rows.append("Prescribed hydrograph volume & hm$^3$/yr & " + " & ".join(f"{v:.0f}" for v in vol) + " \\\\")
    mif = [min_instream_flow(t).sum() * 86400 / 1e6 for t in ff.YEAR_TYPES]
    rows.append("Minimum instream flow volume & hm$^3$/yr & " + " & ".join(f"{v:.0f}" for v in mif) + " \\\\")
    nat = []
    for t in ff.YEAR_TYPES:
        nat.append(np.mean([ff.ff_score(env.years[y], T[t])[0] for y in TRAIN_YEARS if typ[y] == t]))
    rows.append("FF score of the unimpaired flow & -- & " + " & ".join(f"{v:.2f}" for v in nat) + " \\\\")
    open("report_eflows/table_targets.tex", "w").write("\n".join(rows) + "\n")
    lo, hi = env.terciles
    open("report_eflows/setup_numbers.tex", "w").write(
        f"\\newcommand{{\\terclo}}{{{lo:.0f}}}\n\\newcommand{{\\tercHi}}{{{hi:.0f}}}\n"
        f"\\newcommand{{\\demandvol}}{{{demand_climatology().sum() * 86400 / 1e6:.0f}}}\n"
        f"\\newcommand{{\\meanrunoff}}{{{np.mean([env.volume[y] for y in TRAIN_YEARS]):.0f}}}\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
