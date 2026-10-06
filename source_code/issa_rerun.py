"""Re-run every ISSA-dependent result with the corrected ISSA (gopt.ISSA), NP = 20.

  T2   : the Table-2 design: Cameraman, Coins, Astronaut; Otsu, Kapur, MCET; n = 2..8; seeds 0..14;
         G = 60 (main controlled budget) and G = 30 (the application budget reported by Wu and Yuan, as supplied by the author;
         to be checked against the PDF). Reference: DP-certified optimum, success gap < 1e-8.
  S120 : the statistical-study design: eight images, three criteria, n = 4..8, seeds 0..9, G = 60. Per-instance mean gap
         and success rate replace the ISSA entries of stats_{otsu,kapur,mcet}.json; the Friedman / Wilcoxon summary
         of the old Table is first REPRODUCED from the stored values (pipeline check) and then recomputed.
  Q    : MCET, n = 5, eight images, seeds 0..9: PSNR and SSIM of ISSA (FSIM cannot be recomputed here: the FSIM code
         used for the earlier table is not in the repository).
Outputs (data_results): issa_rerun_table2.json, issa_rerun_stats120.json, issa_rerun_quality.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, time
import numpy as np
import paths
from scipy import stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import criteria as cr, gopt
from runtime_scaling import dp_solve
from ablation import all_images

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
L, TOL = 256, 1e-8
CRITS = ("otsu", "kapur", "mcet")
cr.L = L


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
    res = {}
    for G in (60, 30):
        rows = []
        for c in CRITS:
            for n in range(2, 9):
                succ, times, gaps = [], [], []
                for im, img in imgs.items():
                    h = cr.histogram(img); P = cr.Problem(h, c)
                    g = -dp_solve(h, c, n, L)[0]
                    for s in range(15):
                        T, f, t, _ = gopt.ISSA(P, n, G=G, seed=s)
                        gap = abs(f - g) / abs(g); assert f - g >= -1e-9 * abs(g)
                        succ.append(gap < TOL); times.append(t); gaps.append(gap)
                rows.append(dict(criterion=c, n=n, G=G, SR=100 * float(np.mean(succ)), time_ms=1000 * float(np.mean(times)),
                                 gap_mean=float(np.mean(gaps)), runs=len(succ)))
                print("T2", G, c, n, f"{rows[-1]['SR']:.0f}", flush=True)
        res[f"G{G}"] = rows
    json.dump(res, open(os.path.join(DATA, "issa_rerun_table2.json"), "w"), indent=1)


def friedman_summary(table):
    """table: dict method -> array over instances of mean gap. Replicates the original summary."""
    meths = list(table)
    M = np.array([table[m] for m in meths]).T
    chi, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    ranks = np.mean([stats.rankdata(r) for r in M], axis=0)
    out = dict(chi2=float(chi), p=float(p), ranks=dict(zip(meths, ranks.tolist())), wilcoxon={})
    ref = table["MDE"]
    for m in meths:
        if m == "MDE":
            continue
        d = table[m] - ref
        pv = stats.wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue
        out["wilcoxon"][m] = dict(p=float(pv), W=int((d > 0).sum()), T=int((d == 0).sum()), L=int((d < 0).sum()))
    return out


def stats120():
    imgs = all_images(); G = 60
    J = {c: json.load(open(os.path.join(DATA, f"stats_{c}.json"))) for c in CRITS}
    old = {m: [] for m in ("PSO", "DE", "SDE", "ISSA", "MDE")}; oldsr = {m: [] for m in old}
    new_gap, new_sr, keys = [], [], []
    for c in CRITS:
        for im in J[c]["images"]:
            P = cr.Problem(cr.histogram(imgs[im]), c)
            for n in range(4, 9):
                e = J[c][im][str(n)]; g = e["gstar"]
                for m in old:
                    old[m].append(e[m]["mean_relgap"]); oldsr[m].append(e[m]["SR"])
                fs = np.array([gopt.ISSA(P, n, G=G, seed=s)[1] for s in range(10)])
                gaps = (fs - g) / abs(g); assert (gaps >= -1e-9).all()
                new_gap.append(float(gaps.mean())); new_sr.append(100 * float(np.mean(gaps < TOL))); keys.append(f"{c}|{im}|{n}")
        print("S120", c, flush=True)
    oldt = {m: np.array(v) for m, v in old.items()}
    newt = dict(oldt); newt["ISSA"] = np.array(new_gap)
    res = dict(n_instances=len(keys),
               reproduction_of_old_summary=friedman_summary(oldt),
               old_mean_SR={m: float(np.mean(v)) for m, v in oldsr.items()},
               new_summary=friedman_summary(newt),
               new_ISSA_mean_SR=float(np.mean(new_sr)), new_ISSA_mean_gap=float(np.mean(new_gap)),
               old_ISSA_mean_gap=float(np.mean(oldt["ISSA"])),
               new_ISSA_instances=dict(zip(keys, zip(new_gap, new_sr))))
    json.dump(res, open(os.path.join(DATA, "issa_rerun_stats120.json"), "w"), indent=1)
    print("old (reproduced):", json.dumps(res["reproduction_of_old_summary"]))
    print("new:", json.dumps(res["new_summary"]))
    print("ISSA mean SR new", res["new_ISSA_mean_SR"], "mean gap old/new", res["old_ISSA_mean_gap"], res["new_ISSA_mean_gap"])


def quality():
    imgs = all_images(); out = {}
    old = json.load(open(os.path.join(DATA, "quality_mcet_n5.json")))
    for im, img in imgs.items():
        P = cr.Problem(cr.histogram(img), "mcet"); ps, ss = [], []
        for s in range(10):
            T = gopt.ISSA(P, 5, seed=s)[0]
            rec = seg_recon(img, T)
            ps.append(peak_signal_noise_ratio(img.astype(float), rec, data_range=255))
            ss.append(structural_similarity(img.astype(float), rec, data_range=255))
        out[im] = dict(PSNR=float(np.mean(ps)), SSIM=float(np.mean(ss)), old_PSNR=old[im]["ISSA"]["PSNR"], old_SSIM=old[im]["ISSA"]["SSIM"])
        print("Q", im, f"PSNR new {out[im]['PSNR']:.2f} old {out[im]['old_PSNR']:.2f}", flush=True)
    out["mean"] = {k: float(np.mean([out[i][k] for i in imgs])) for k in ("PSNR", "SSIM", "old_PSNR", "old_SSIM")}
    json.dump(out, open(os.path.join(DATA, "issa_rerun_quality.json"), "w"), indent=1)
    print("quality mean", out["mean"])


if __name__ == "__main__":
    stats120(); quality(); table2()
