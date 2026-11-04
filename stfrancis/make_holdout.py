"""
Held-out-year figure for the paper: controllers tuned/trained on 2021-2023, scored on 2020
(results/eval_R{0,1}_ho2020.json from evaluate_sf.py --split ho2020).

python make_holdout.py
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from make_paper import C, CASES, FIG, RES


def rows(R):
    t = json.load(open(os.path.join(RES, f"eval_R{R}_ho2020.json")))
    ckpt = sorted((int(m.group(1)), r) for r in t if (m := re.search(r"_u(\d{4})\.pt$", r["policy"])))
    return {"rule": t[1], "orifice": t[3], "ckpt": ckpt}


def main():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for col, R in enumerate((0, 1)):
        d = rows(R)
        u = [k for k, _ in d["ckpt"]]
        for row, (w, lab) in enumerate((("train", "2021–2023 (training)"), ("test", "2020 (held out)"))):
            ax = axs[row, col]
            ax.plot(u, [r[w]["J"] for _, r in d["ckpt"]], "o-", ms=3, color=C["ppo"], label="PPO checkpoints")
            ax.plot(u[-1], d["ckpt"][-1][1][w]["J"], "o", ms=7, color=C["ppo"], mec="white", mew=1.2)
            ax.axhline(d["rule"][w]["J"], color=C["rule"], lw=1.2, ls="--", label="tuned rule")
            ax.axhline(d["orifice"][w]["J"], color=C["orifice"], lw=1.2, ls=":", label="best fixed orifice")
            if R == 1:   # case 1 has J = 0 for the rule and the orifice in 2021-2023: linear axis
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1] if w == "train" else [0.4, 0.6, 1, 1.5])
            if row == 0:
                ax.set_title(CASES[R], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"$J$, {lab}")
            if row == 1:
                ax.set_xlabel("PPO update")
    axs[1, 1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "holdout_2020.pdf"))


if __name__ == "__main__":
    main()
