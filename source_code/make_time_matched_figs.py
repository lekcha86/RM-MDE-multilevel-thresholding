"""Figures for Experiment R2 (time-matched PSO/DE/RM-MDE), drawn only from the raw CSV files.
Fig A : success rate at the matched budget versus n          -> fig_tm_success.png
Fig B : mean objective gap versus normalized time t/T_target -> fig_tm_gap.png
Fig C : P(success by t/T_target)                              -> fig_tm_psuccess.png
Consistency checks against the 720 paired runs are printed (and assert-ed).
"""
import os, csv, collections, json
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
STY = {"RM-MDE": dict(color="#0072B2", ls="-", marker="o", lw=2.0),
       "PSO": dict(color="#D55E00", ls="--", marker="s", lw=1.6),
       "DE": dict(color="#009E73", ls=":", marker="^", lw=1.6)}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

runs = list(csv.DictReader(open(os.path.join(DATA, "time_matched_DE_PSO.csv"))))
for r in runs:
    r["n"] = int(r["n"]); r["seed"] = int(r["seed"]); r["budget_mult"] = int(r["budget_mult"]); r["success"] = int(r["success"])
    r["T_target"] = float(r["T_target"]); r["objective_gap"] = float(r["objective_gap"])
key = lambda r: (r["image"], r["criterion"], r["n"], r["seed"])
Tt = {key(r): r["T_target"] for r in runs if r["method"] == "RM-MDE"}
assert len(Tt) == 720

# ---------- Fig A: success rate vs n ----------
def sr(crit, n, m, k):
    x = [r["success"] for r in runs if r["criterion"] == crit and r["n"] == n and r["method"] == m and r["budget_mult"] == k]
    assert len(x) == 80
    return 100 * np.mean(x)

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, (c, cn) in zip(axs, CRITS):
    for m, k, lab, ls in [("RM-MDE", 1, "RM-MDE", None), ("PSO", 1, "PSO, 1×", None), ("PSO", 3, "PSO, 3×", "-."),
                          ("DE", 1, "DE, 1× (= 2×, 3×)", None)]:
        s = dict(STY[m])
        if ls: s["ls"] = ls; s["alpha"] = 0.6
        ax.plot([6, 7, 8], [sr(c, n, m, k) for n in (6, 7, 8)], label=lab, ms=5, **s)
    ax.set_title(cn); ax.set_xticks([6, 7, 8]); ax.set_xlabel("number of thresholds $n$"); ax.set_ylim(0, 100)
axs[0].set_ylabel("success rate (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "fig_tm_success.png"), dpi=200); plt.close(fig)

# ---------- traces -> step functions on a normalized-time grid ----------
tr = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(DATA, "time_trace_DE_PSO_RMMDE.csv"))):
    tr[(r["image"], r["criterion"], int(r["n"]), int(r["seed"]), r["method"])].append((float(r["elapsed_time"]), float(r["best_gap"])))
grid = np.linspace(0.05, 3.0, 119)
G = {}   # (crit, method) -> array runs x grid of best-so-far gap; NaN beyond a method's own horizon
for (im, c, n, s, m), pts in tr.items():
    t = np.array([p[0] for p in pts]) / Tt[(im, c, n, s)]; g = np.array([p[1] for p in pts])
    idx = np.searchsorted(t, grid, side="right") - 1
    v = np.where(idx >= 0, g[np.clip(idx, 0, None)], np.nan)
    v[idx < 0] = g[0]                       # before the first recorded point: use the initial best
    if m == "RM-MDE":
        v[grid > 1.0 + 1e-9] = np.nan       # RM-MDE has a single fixed budget (its own run)
    G.setdefault((c, m), []).append(v)
G = {k: np.array(v) for k, v in G.items()}
for k, v in G.items():
    assert v.shape[0] == 240, (k, v.shape)

# ---------- Fig B: mean gap vs normalized time ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, (c, cn) in zip(axs, CRITS):
    for m in ("RM-MDE", "PSO", "DE"):
        y = np.nanmean(G[(c, m)], axis=0)
        s = dict(STY[m]); s.pop("marker")
        ax.plot(grid, y, label=m, **s)
    ax.set_yscale("log"); ax.set_title(cn); ax.set_xlabel("elapsed time $t/T_{\\mathrm{target}}$")
    ax.axvline(1.0, color="0.4", lw=0.8)
axs[0].set_ylabel("mean relative objective gap")
axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_tm_gap.png"), dpi=200); plt.close(fig)

# ---------- Fig C: P(success by t) ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for ax, (c, cn) in zip(axs, CRITS):
    for m in ("RM-MDE", "PSO", "DE"):
        p = 100 * np.nanmean(G[(c, m)] < TOL, axis=0)
        if m == "RM-MDE":
            p = np.where(grid <= 1.0 + 1e-9, p, np.nan)
        s = dict(STY[m]); s.pop("marker")
        ax.plot(grid, p, label=m, **s)
    ax.set_title(cn); ax.set_xlabel("elapsed time $t/T_{\\mathrm{target}}$"); ax.set_ylim(0, 100)
    ax.axvline(1.0, color="0.4", lw=0.8)
axs[0].set_ylabel("P(success by $t$) (%)")
axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_tm_psuccess.png"), dpi=200); plt.close(fig)

# ---------- consistency checks ----------
D = json.load(open(os.path.join(DATA, "time_matched_DE_PSO.json")))
ref = json.load(open(os.path.join(DATA, "exact_optima_dp.json")))
assert len(D["runs"]) == len(runs) == 5040
for r in D["runs"]:
    g = ref[f"{r['criterion']}|{r['image']}|{r['n']}"]["g_dp"]
    assert r["objective"] - g >= -1e-9 * abs(g), r   # no run may beat the certified optimum
for m, k in [("RM-MDE", 1), ("PSO", 1), ("PSO", 3), ("DE", 1)]:
    tot = 100 * np.mean([r["success"] for r in runs if r["method"] == m and r["budget_mult"] == k])
    print(f"{m} {k}x overall SR {tot:.1f}")
# figure (step at time <= t) versus table (first generation completing >= t) at t = 1 and 3
for c, _ in CRITS:
    for m in ("PSO", "DE"):
        for tt in (1.0, 3.0):
            i = int(np.argmin(abs(grid - tt)))
            fig_sr = 100 * np.mean(G[(c, m)][:, i] < TOL)
            tab_sr = 100 * np.mean([r["success"] for r in runs if r["criterion"] == c and r["method"] == m and r["budget_mult"] == int(tt)])
            print(f"{c} {m} t={tt:g}: figure {fig_sr:.1f}  table {tab_sr:.1f}")
