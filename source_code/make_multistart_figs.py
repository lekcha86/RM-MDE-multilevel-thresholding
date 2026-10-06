"""Figures for Experiment C, drawn from multistart_C_runs.csv and multistart_C_trace.csv.
  fig_ms_success.png    : success rate versus n, per criterion (RM-MDE, C1 multistart-4S, C2 multistart-converged)
  fig_ms_psuccess.png   : P(success by t / T_target), pooled over images, n and seeds, per criterion
"""
import os, csv, collections
import numpy as np
import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
FIG = paths.FIG
TOL = 1e-8
CRITS = [("otsu", "Otsu"), ("kapur", "Kapur"), ("mcet", "MCET")]
M = ["RM-MDE", "MS-LS-4S", "MS-LS-conv"]
LAB = {"RM-MDE": "RM-MDE", "MS-LS-4S": "multistart LS (4 sweeps)", "MS-LS-conv": "multistart LS (converged)"}
STY = {"RM-MDE": dict(color="#0072B2", ls="-", marker="o", lw=2.0),
       "MS-LS-4S": dict(color="#D55E00", ls="--", marker="s", lw=1.6),
       "MS-LS-conv": dict(color="#009E73", ls=":", marker="^", lw=1.8)}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

runs = list(csv.DictReader(open(os.path.join(DATA, "multistart_C_runs.csv"))))
for r in runs:
    r["n"] = int(r["n"]); r["seed"] = int(r["seed"]); r["success"] = int(r["success"]); r["T_target"] = float(r["T_target"])
assert len(runs) == 2160
Tt = {(r["image"], r["criterion"], r["n"], r["seed"]): r["T_target"] for r in runs if r["method"] == "RM-MDE"}

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, (c, cn) in zip(axs, CRITS):
    for m in M:
        ys = []
        for n in (6, 7, 8):
            x = [r["success"] for r in runs if r["criterion"] == c and r["n"] == n and r["method"] == m]
            assert len(x) == 80
            ys.append(100 * np.mean(x))
        ax.plot([6, 7, 8], ys, label=LAB[m], ms=5, **STY[m])
    ax.set_title(cn); ax.set_xticks([6, 7, 8]); ax.set_xlabel("number of thresholds $n$"); ax.set_ylim(0, 100)
axs[0].set_ylabel("success rate (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "fig_ms_success.png"), dpi=200); plt.close(fig)

tr = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(DATA, "multistart_C_trace.csv"))):
    tr[(r["image"], r["criterion"], int(r["n"]), int(r["seed"]), r["method"])].append((float(r["elapsed_time"]), float(r["best_gap"])))
grid = np.linspace(0.02, 1.0, 99)
G = collections.defaultdict(list)
for (im, c, n, s, m), pts in tr.items():
    t = np.array([p[0] for p in pts]) / Tt[(im, c, n, s)]; g = np.array([p[1] for p in pts])
    idx = np.searchsorted(t, grid, side="right") - 1
    v = np.where(idx >= 0, g[np.clip(idx, 0, None)], np.inf)     # nothing found yet -> not successful
    G[(c, m)].append(v)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, (c, cn) in zip(axs, CRITS):
    for m in M:
        a = np.array(G[(c, m)]); assert a.shape[0] == 240
        s = dict(STY[m]); s.pop("marker")
        ax.plot(grid, 100 * np.mean(a < TOL, axis=0), label=LAB[m], **s)
    ax.set_title(cn); ax.set_xlabel("elapsed time $t/T_{\\mathrm{target}}$"); ax.set_ylim(0, 100)
axs[0].set_ylabel("P(success by $t$) (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "fig_ms_psuccess.png"), dpi=200); plt.close(fig)

# consistency: the curve value at t = 1 must equal the table success rate
for c, _ in CRITS:
    for m in M:
        fig_sr = 100 * np.mean(np.array(G[(c, m)])[:, -1] < TOL)
        tab = 100 * np.mean([r["success"] for r in runs if r["criterion"] == c and r["method"] == m])
        print(f"{c} {m}: curve at t=1 {fig_sr:.1f}  table {tab:.1f}")
