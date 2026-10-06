"""Figure for Experiment B: success rate versus median runtime for each p_ls (drawn from ablation_B_runs.csv)."""
import os, csv, collections
import numpy as np
import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
FIG = paths.FIG
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

rows = list(csv.DictReader(open(os.path.join(DATA, "ablation_B_runs.csv"))))
ps = sorted({float(r["p_ls"]) for r in rows})
assert len(rows) == 1680 and len(ps) == 7
col = {6: "#0072B2", 7: "#D55E00", 8: "#009E73"}
mk = {6: "o", 7: "s", 8: "^"}
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for n in (6, 7, 8):
    xs, ys = [], []
    for p in ps:
        r = [x for x in rows if float(x["p_ls"]) == p and int(x["n"]) == n]
        assert len(r) == 80
        xs.append(np.median([float(x["runtime"]) for x in r])); ys.append(100 * np.mean([int(x["success"]) for x in r]))
    ax.plot(xs, ys, marker=mk[n], color=col[n], lw=1.5, ms=5, label=f"$n={n}$")
# label p on the n=6 curve (top curve)
r7 = []
for p in ps:
    r = [x for x in rows if float(x["p_ls"]) == p and int(x["n"]) == 6]
    r7.append((np.median([float(x["runtime"]) for x in r]), 100 * np.mean([int(x["success"]) for x in r]), p))
for x, y, p in r7:
    ax.annotate(f"{p:g}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color="0.3")
ax.set_xscale("log"); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20))
ax.set_xlabel("median runtime per run (s), log scale"); ax.set_ylabel("success rate (%)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Success versus runtime for $p_{ls}$ (labels on $n=6$)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ablation_B_success_vs_time.png"), dpi=200)
print("saved")
