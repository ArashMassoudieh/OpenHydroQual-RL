"""
Figures, tables and diagnostics for the St. Francis RL paper (stfrancis/paper/).
Uses the tuned rule / orifice results and the evaluated PPO checkpoints.

python make_paper.py
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import run_full, out_index, Kernel, KERNELS   # noqa: E402
from evaluate_sf import rollout                            # noqa: E402
from forecast import load_rain                             # noqa: E402

RES = os.path.join(HERE, "results")
PAP = os.path.join(HERE, "paper")
FIG = os.path.join(PAP, "figures")
os.makedirs(FIG, exist_ok=True)
DT = 1 / 24
QLIM = 50.0
C = {"passive": "#8a8a85", "orifice": "#2a78d6", "rule": "#eb6834", "ppo": "#1baf7a"}
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
CASES = {0: "Case 1: excess only", 1: "Case 2: excess + residence time"}
PPO_SHOW = {0: "ppo_sf_R0.pt", 1: "ppo_sf_R1.pt"}   # the final policy of each complete training run

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 1.6, "savefig.bbox": "tight", "savefig.dpi": 200})


def load(f):
    return json.load(open(os.path.join(RES, f)))


ref = load("passive.json")
names = ["t", "pond", "eng", "native", "cb", "agg", "outflow", "weir", "valve"]


def kernel_cols(kind):
    k = Kernel(KERNELS[kind])
    ix = out_index(k)
    return [1 + ix[n] for n in ("pond", "outflow", "weir", "valve")]


def _record(job):
    kind, R, arg = job
    if kind == "passive":
        r, rec = run_full("passive", record=True, dt=DT)
    elif kind == "rule":
        r, rec = run_full("rule", arg, record=True, dt=DT)
    elif kind == "orifice":
        r, rec = run_full("rl", policy=lambda t, o: arg, dt=DT, record=True)
    else:
        out, rec = rollout(R, arg, record=True)
        # rollout record: t, opening, u_rule, outflow, weir, valve, pond, dEx, dAge
        return {"Excess": out["Excess"], "Age_Load": out["Age_Load"],
                "t": rec[:, 0], "pond": rec[:, 6], "outflow": rec[:, 3], "weir": rec[:, 4],
                "valve": rec[:, 5], "opening": rec[:, 1], "u_rule": rec[:, 2]}
    cols = kernel_cols("rule" if kind == "rule" else "rl" if kind == "orifice" else "passive")
    d = {"Excess": r["Excess"], "Age_Load": r["Age_Load"], "t": rec[:, 0]}
    for n, c in zip(("pond", "outflow", "weir", "valve"), cols):
        d[n] = rec[:, c]
    if kind == "orifice":
        d["opening"] = np.full(len(d["t"]), arg)
    return d


def stats(d, R):
    q = d["outflow"]
    ex, ag = d["Excess"] / ref["Excess"], d["Age_Load"] / ref["Age_Load"]
    s = {"J": ex + R * ag, "Excess": d["Excess"], "Excess_rel": ex, "Age_rel": ag,
         "peak": float(q.max()), "hours_above": int((q > QLIM).sum()),
         "weir_vol": float(d["weir"].sum() * DT), "out_vol": float(q.sum() * DT),
         "pond_mean": float(d["pond"].mean())}
    if "opening" in d:
        s["closed_frac"] = float((d["opening"] < 1e-9).mean())
        s["open_mean"] = float(d["opening"].mean())
    return s


def main():
    jobs, keys = [("passive", 0, None)], [("passive", None)]
    for R in (0, 1):
        tag = f"R{R}"
        jobs += [("rule", R, load(f"rule_{tag}.json")["params"]), ("orifice", R, load(f"orifice_{tag}.json")["opening"]),
                 ("ppo", R, os.path.join(RES, PPO_SHOW[R])), ("ppo", R, None)]
        keys += [("rule", R), ("orifice", R), ("ppo", R), ("rule_hourly", R)]
    with Pool(6) as pool:
        recs = pool.map(_record, jobs)
    D = dict(zip(keys, recs))

    # ------------------------------------------------------------- stats
    S = {}
    for R in (0, 1):
        S[R] = {"passive": stats(D[("passive", None)], R)}
        for k in ("orifice", "rule", "rule_hourly", "ppo"):
            S[R][k] = stats(D[(k, R)], R)
        S[R]["ppo"]["checkpoint"] = PPO_SHOW[R]
        ev = load(f"eval_{tag_of(R)}.json")
        S[R]["checkpoints"] = [(int(e["policy"].split("_u")[1][:4]), e["full"]["J"], e["full"]["Excess_rel"],
                                e["full"]["Age_rel"]) for e in ev if e["policy"].startswith("ppo:")]
        S[R]["ppo"]["update"] = final_update(R)
        S[R]["checkpoints"].append((S[R]["ppo"]["update"], S[R]["ppo"]["J"], S[R]["ppo"]["Excess_rel"],
                                    S[R]["ppo"]["Age_rel"]))
        S[R]["rule_params"] = load(f"rule_{tag_of(R)}.json")["params"]
        S[R]["orifice_opening"] = load(f"orifice_{tag_of(R)}.json")["opening"]
    json.dump(S, open(os.path.join(RES, "paper_stats.json"), "w"), indent=1)
    for R in (0, 1):
        print(CASES[R])
        for k in ("passive", "orifice", "rule", "rule_hourly", "ppo"):
            print(f"  {k:12s}", {a: round(b, 4) if isinstance(b, float) else b for a, b in S[R][k].items()})

    fig_bars(S)
    fig_sweep(S)
    fig_training(S)
    fig_tradeoff(S)
    fig_storm(D, S)
    tables(S)


def final_update(R):
    """Number of the last PPO update, from the training log."""
    last = [ln for ln in open(os.path.join(RES, "logs", f"train_R{R}.txt")) if ln.startswith("update")][-1]
    return int(last.split()[1].split("/")[0])


def tag_of(R):
    return f"R{R}"


def fig_bars(S):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for ax, R in zip(axs, (0, 1)):
        labels = ["Fixed orifice", "Tuned rule", "PPO (final)"]
        vals = [S[R]["orifice"]["J"], S[R]["rule"]["J"], S[R]["ppo"]["J"]]
        cols = [C["orifice"], C["rule"], C["ppo"]]
        y = np.arange(3)[::-1]
        ax.barh(y, vals, color=cols, height=0.55)
        for yi, v in zip(y, vals):
            ax.text(v * 1.01, yi, f"{v:.3f}", va="center", ha="left", color=INK, fontsize=8)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_xlabel("J (passive = %d; lower is better)" % (1 + R))
        ax.set_title(CASES[R], fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "J_bars.pdf"))


def fig_sweep(S):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, R in zip(axs, (0, 1)):
        sw = np.array(load(f"orifice_{tag_of(R)}.json")["sweep"])
        ax.plot(sw[:, 0] * 1000, sw[:, 1], color=C["orifice"], marker="o", ms=3, label="Fixed orifice (sweep)")
        ax.axhline(S[R]["rule"]["J"], color=C["rule"], lw=1.6, label="Tuned rule")
        ax.axhline(S[R]["ppo"]["J"], color=C["ppo"], lw=1.6, ls="--", label="PPO (final)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("orifice opening (x 10$^{-3}$)")
        ax.set_ylabel("J")
        ax.set_title(CASES[R], fontsize=9, color=INK, loc="left")
    axs[0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "orifice_sweep.pdf"))


def fig_training(S):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, R in zip(axs, (0, 1)):
        c = np.array(S[R]["checkpoints"])
        ax.plot(c[:, 0], c[:, 1], color=C["ppo"], marker="o", ms=3.5, label="PPO checkpoint")
        ax.axhline(S[R]["rule"]["J"], color=C["rule"], label="Tuned rule")
        ax.axhline(S[R]["orifice"]["J"], color=C["orifice"], label="Fixed orifice")
        ax.set_xlabel("PPO update (4,320 steps each)")
        ax.set_ylabel("J, whole record")
        ax.set_title(CASES[R], fontsize=9, color=INK, loc="left")
    axs[1].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training.pdf"))


def fig_tradeoff(S):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sw0 = np.array(load("orifice_R0.json")["sweep"])
    # orifice sweep in (Excess_rel, Age_rel): rerun-free from the R0 and R1 sweeps: J0 = ex, J1 = ex + ag
    sw1 = np.array(load("orifice_R1.json")["sweep"])
    ex, ag = sw0[:, 1], sw1[:, 1] - sw0[:, 1]
    keep = ex < 0.6
    ax.plot(ex[keep], ag[keep], color=C["orifice"], marker="o", ms=2.5, lw=1.2, label="Fixed orifice (sizes)")
    for R, mk in ((0, "s"), (1, "D")):
        ax.plot(S[R]["rule"]["Excess_rel"], S[R]["rule"]["Age_rel"], marker=mk, ms=7, color=C["rule"], ls="",
                label=f"Tuned rule, {['case 1', 'case 2'][R]}")
    for R, mk in ((0, "v"), (1, "^")):
        ax.plot(S[R]["ppo"]["Excess_rel"], S[R]["ppo"]["Age_rel"], marker=mk, ms=7, color=C["ppo"], ls="",
                label=f"Final PPO, {['case 1', 'case 2'][R]}")
    xs = np.linspace(0.05, 0.35, 50)
    for Jc in (S[1]["rule"]["J"], S[1]["ppo"]["J"]):
        ax.plot(xs, Jc - xs, color=INK2, lw=0.7, ls=":")
    ax.set_xlim(0.05, 0.35)
    ax.set_ylim(0.08, 0.6)
    ax.set_xlabel("Excess / passive")
    ax.set_ylabel("young-water load / passive")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tradeoff.pdf"))


def fig_storm(D, S):
    s, e, dep = load_rain()
    t = D[("passive", None)]["t"]
    # largest 24 h rain total
    tm = 0.5 * (s + e)
    w = np.array([dep[(tm >= a) & (tm < a + 1)].sum() for a in np.arange(t[0], t[-1] - 1, 0.25)])
    a0 = t[0] + 0.25 * int(np.argmax(w))
    lo, hi = a0 - 2.5, a0 + 4.5
    for R in (0, 1):
        fig, axs = plt.subplots(4, 1, figsize=(7.0, 6.4), sharex=True,
                                gridspec_kw={"height_ratios": [0.7, 1.4, 1.1, 1.0]})
        m = (tm >= lo) & (tm <= hi)
        axs[0].bar(tm[m] - lo, dep[m] * 1000, width=1 / 24, color=INK2)
        axs[0].set_ylabel("rain (mm/h)")
        series = [("passive", D[("passive", None)], "Passive (free draining)"),
                  ("orifice", D[("orifice", R)], "Fixed orifice"),
                  ("rule", D[("rule", R)], "Tuned rule"),
                  ("ppo", D[("ppo", R)], "PPO (final policy)")]
        for key, d, lab in series:
            mm = (d["t"] >= lo) & (d["t"] <= hi)
            axs[1].plot(d["t"][mm] - lo, np.maximum(d["outflow"][mm], 0.1), color=C[key], label=lab, lw=1.4)
            axs[2].plot(d["t"][mm] - lo, d["pond"][mm], color=C[key], lw=1.4)
            if "opening" in d:
                axs[3].plot(d["t"][mm] - lo, d["opening"][mm] * 1000, color=C[key], lw=1.4)
        dr = D[("rule_hourly", R)]
        mm = (dr["t"] >= lo) & (dr["t"] <= hi)
        axs[3].plot(dr["t"][mm] - lo, dr["opening"][mm] * 1000, color=C["rule"], lw=1.4)
        axs[1].axhline(QLIM, color=INK, lw=0.8, ls="--")
        axs[1].text(0.05, QLIM * 1.08, "Q$_{lim}$ = 50 m$^3$/d", fontsize=7.5, color=INK)
        axs[1].set_ylabel("outflow (m$^3$/d)")
        axs[1].set_yscale("log")
        axs[1].set_ylim(0.1, 2000)
        axs[1].legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
        axs[2].set_ylabel("ponded volume (m$^3$)")
        axs[3].set_ylabel("opening (x10$^{-3}$)")
        axs[3].set_xlabel(f"days from t = {lo:.1f} (Excel date; largest 24-h storm of the record)")
        axs[0].set_title(CASES[R], fontsize=9, color=INK, loc="left")
        fig.tight_layout(h_pad=1.6)
        fig.savefig(os.path.join(FIG, f"storm_R{R}.pdf"))
    S["storm_start"] = lo


def tables(S):
    rows = []
    for R in (0, 1):
        b = min(S[R]["checkpoints"], key=lambda c: c[1])
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{CASES[R]}}}}}\\\\")
        for k, lab in (("passive", "Passive (free draining)"), ("orifice", "Best fixed orifice"),
                       ("rule", "Tuned forecast rule"), ("ppo", "PPO (final policy)")):
            s = S[R][k]
            rows.append(f"{lab} & {s['J']:.3f} & {s['Excess']:.0f} & {s['Excess_rel']:.3f} & {s['Age_rel']:.3f} & "
                        f"{s['peak']:.0f} & {s['hours_above']} & {s['weir_vol']:.0f}\\\\")
        rows.append("\\addlinespace")
    open(os.path.join(PAP, "table_main.tex"), "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
