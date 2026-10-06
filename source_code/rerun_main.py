"""Consolidated re-run of the main-study tables and figures after the baseline corrections (ISSA, SDE, FSIM removed).

Baselines (all NP = 20, G = 60): PSO (constriction, ring); DE rand/1/bin (F = 0.5, Cr = 0.9); SDE with the parameter setting reported by
Ali et al. (2014), F = Cr = 0.25 fixed; ISSA = corrected one-dimensional adaptation (gopt.ISSA); RM-MDE (p_ls = 0.15).
Reference for success: DP-certified optimum (gap < 1e-8). One script, one reference, one seed rule, so that every column is consistent.

  T2   : Cameraman, Coins, Astronaut; Otsu, Kapur, MCET; n = 2..8; seeds 0..14  -> success rate and mean time per method
  CONV : MCET, Cameraman, n = 5 (six classes): mean relative gap per generation, seeds 0..14
  Q    : MCET, n = 5, eight images, seeds 0..9: PSNR and SSIM of the class-mean reconstruction, all five methods
Outputs (data_results): main_rerun_table2.json, main_rerun_conv.json, main_rerun_quality.json
Figures (figures): criteria_scalability.png, conv_Cameraman.png (same file names as before)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, sys
import numpy as np
import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import criteria as cr, gopt
from runtime_scaling import dp_solve
from ablation import all_images

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
FIG = paths.FIG
L, TOL = 256, 1e-8
CRITS = ("otsu", "kapur", "mcet")
cr.L = L
METHODS = {
    "PSO": lambda P, n, s: gopt.PSO(P, n, seed=s),
    "DE": lambda P, n, s: gopt.DE(P, n, seed=s),
    "SDE": lambda P, n, s: gopt.SDE(P, n, F=0.25, Cr=0.25, seed=s, dither=False),
    "ISSA": lambda P, n, s: gopt.ISSA(P, n, seed=s),
    "RM-MDE": lambda P, n, s: gopt.MDE(P, n, seed=s),
}
STY = {"PSO": dict(color="#1f77b4", ls="--", marker="s", lw=1.0, ms=3.5),
       "DE": dict(color="#2ca02c", ls="--", marker="^", lw=1.0, ms=3.5),
       "SDE": dict(color="#ff7f0e", ls=":", marker="v", lw=1.2, ms=3.5),
       "ISSA": dict(color="#9467bd", ls="-.", marker="D", lw=1.0, ms=3.0),
       "RM-MDE": dict(color="#dc143c", ls="-", marker="o", lw=2.4, ms=4.5)}


def seg_recon(img, T):
    thr = [t - 1 for t in T]; edges = [-1] + thr + [255]
    out = np.zeros(img.shape, float)
    for k in range(len(edges) - 1):
        m = (img >= edges[k] + 1) & (img <= edges[k + 1])
        if m.any():
            out[m] = img[m].mean()
    return out


def table2():
    base = all_images(); imgs = {k: base[k] for k in ("Cameraman", "Coins", "Astronaut")}
    res = {c: {n: {m: dict(succ=[], t=[]) for m in METHODS} for n in range(2, 9)} for c in CRITS}
    for c in CRITS:
        for n in range(2, 9):
            for im, img in imgs.items():
                h = cr.histogram(img); P = cr.Problem(h, c)
                g = -dp_solve(h, c, n, L)[0]
                for m, fn in METHODS.items():
                    for s in range(15):
                        T, f, t, _ = fn(P, n, s)
                        gap = abs(f - g) / abs(g); assert f - g >= -1e-9 * abs(g)
                        res[c][n][m]["succ"].append(bool(gap < TOL)); res[c][n][m]["t"].append(t)
            print("T2", c, n, {m: round(100 * np.mean(res[c][n][m]["succ"])) for m in METHODS}, flush=True)
    out = {c: {str(n): {m: dict(SR=100 * float(np.mean(v["succ"])), time_ms=1000 * float(np.mean(v["t"])), runs=len(v["succ"]))
                        for m, v in res[c][n].items()} for n in res[c]} for c in res}
    json.dump(out, open(os.path.join(DATA, "main_rerun_table2.json"), "w"), indent=1)
    # scalability figure
    names = {"otsu": "Otsu (between-class variance)", "kapur": "Kapur entropy", "mcet": "Minimum cross entropy"}
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
    for ax, c in zip(axs, CRITS):
        for m in METHODS:
            ax.plot(range(2, 9), [out[c][str(n)][m]["SR"] for n in range(2, 9)], label=m, **STY[m])
        ax.set_title(names[c], fontsize=9); ax.set_xlabel("# thresholds n"); ax.set_ylim(-5, 105); ax.grid(alpha=0.3)
    axs[0].set_ylabel("Success rate (%)  [reach global optimum]")
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "criteria_scalability.png"), dpi=200); plt.close(fig)
    return out


def conv():
    ref = json.load(open(os.path.join(DATA, "exact_optima_dp.json")))
    img = all_images()["Cameraman"]; P = cr.Problem(cr.histogram(img), "mcet")
    g = ref["mcet|Cameraman|5"]["g_dp"]
    curves = {}
    for m, fn in METHODS.items():
        H = np.array([fn(P, 5, s)[3] for s in range(15)])
        curves[m] = (np.mean(np.maximum((H - g) / abs(g), 1e-16), axis=0)).tolist()
    json.dump(curves, open(os.path.join(DATA, "main_rerun_conv.json"), "w"))
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for m, y in curves.items():
        s = dict(STY[m]); s.pop("marker"); s.pop("ms")
        ax.plot(range(1, len(y) + 1), y, label=m, **s)
    ax.set_yscale("log"); ax.set_xlabel("Generation"); ax.set_ylabel("Mean relative gap to $g^*$")
    ax.set_title("Cameraman, minimum cross entropy, 6 classes (n=5)", fontsize=8.5); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "conv_Cameraman.png"), dpi=200); plt.close(fig)
    return curves


def quality():
    imgs = all_images(); out = {}
    for im, img in imgs.items():
        P = cr.Problem(cr.histogram(img), "mcet"); out[im] = {}
        for m, fn in METHODS.items():
            ps, ss = [], []
            for s in range(10):
                T = fn(P, 5, s)[0]; rec = seg_recon(img, T)
                ps.append(peak_signal_noise_ratio(img.astype(float), rec, data_range=255))
                ss.append(structural_similarity(img.astype(float), rec, data_range=255))
            out[im][m] = dict(PSNR=float(np.mean(ps)), SSIM=float(np.mean(ss)))
        print("Q", im, {m: round(out[im][m]["PSNR"], 2) for m in METHODS}, flush=True)
    out["mean"] = {m: dict(PSNR=float(np.mean([out[i][m]["PSNR"] for i in imgs])), SSIM=float(np.mean([out[i][m]["SSIM"] for i in imgs]))) for m in METHODS}
    json.dump(out, open(os.path.join(DATA, "main_rerun_quality.json"), "w"), indent=1)
    print("quality mean", json.dumps(out["mean"]))


if __name__ == "__main__":
    quality(); conv(); table2()
