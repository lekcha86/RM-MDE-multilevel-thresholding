"""Re-run the SDE-dependent results with the SDE of Ali et al. (2014) parameter setting, F = Cr = 0.25 fixed (no randomized F),
under the common budget NP = 20, G = 60 (NOT their NP = 10 D / 200 iterations).

  T2   : Table-2 design: Cameraman, Coins, Astronaut; Otsu, Kapur, MCET; n = 2..8; seeds 0..14; DP-certified reference.
  S120 : the statistical study: 8 images x 3 criteria x n = 4..8 x seeds 0..9. Per-instance mean gap / success rate replace the SDE
         entries (and, with issa_rerun_stats120.json, the ISSA entries); the Friedman / Wilcoxon summary is recomputed from scratch.
  CHK  : RM-MDE / DE / PSO entries of the stored study are checked against fresh runs of gopt (seeds 0..9) on a subset, to confirm
         that the stored RM-MDE results are unchanged and that the seeds are 0..9.
Outputs (data_results): sde_rerun_table2.json, sde_rerun_stats120.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json
import numpy as np
import paths
from scipy import stats
import criteria as cr, gopt
from runtime_scaling import dp_solve
from ablation import all_images
from issa_rerun import friedman_summary

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
L, TOL = 256, 1e-8
CRITS = ("otsu", "kapur", "mcet")
cr.L = L


def sde_paper(P, n, seed):
    return gopt.SDE(P, n, F=0.25, Cr=0.25, seed=seed, dither=False)


def table2():
    base = all_images(); imgs = {k: base[k] for k in ("Cameraman", "Coins", "Astronaut")}
    rows = []
    for c in CRITS:
        for n in range(2, 9):
            succ, times, gaps = [], [], []
            for im, img in imgs.items():
                h = cr.histogram(img); P = cr.Problem(h, c)
                g = -dp_solve(h, c, n, L)[0]
                for s in range(15):
                    T, f, t, _ = sde_paper(P, n, s)
                    gap = abs(f - g) / abs(g); assert f - g >= -1e-9 * abs(g)
                    assert abs(P.obj(list(T)) - f) <= 1e-9 * abs(f)          # position / fitness consistent
                    succ.append(gap < TOL); times.append(t); gaps.append(gap)
            rows.append(dict(criterion=c, n=n, SR=100 * float(np.mean(succ)), time_ms=1000 * float(np.mean(times)),
                             gap_mean=float(np.mean(gaps)), runs=len(succ)))
            print("T2", c, n, f"{rows[-1]['SR']:.0f}", flush=True)
    json.dump(rows, open(os.path.join(DATA, "sde_rerun_table2.json"), "w"), indent=1)


def stats120():
    imgs = all_images()
    J = {c: json.load(open(os.path.join(DATA, f"stats_{c}.json"))) for c in CRITS}
    issa_new = json.load(open(os.path.join(DATA, "issa_rerun_stats120.json")))["new_ISSA_instances"]
    old = {m: [] for m in ("PSO", "DE", "SDE", "ISSA", "MDE")}
    new_sde, new_sde_sr, issa_gap, keys = [], [], [], []
    chk = []
    for c in CRITS:
        for im in J[c]["images"]:
            P = cr.Problem(cr.histogram(imgs[im]), c)
            for n in range(4, 9):
                e = J[c][im][str(n)]; g = e["gstar"]
                for m in old:
                    old[m].append(e[m]["mean_relgap"])
                fs = np.array([sde_paper(P, n, s)[1] for s in range(10)])
                gaps = (fs - g) / abs(g); assert (gaps >= -1e-9).all()
                new_sde.append(float(gaps.mean())); new_sde_sr.append(100 * float(np.mean(gaps < TOL)))
                k = f"{c}|{im}|{n}"; keys.append(k); issa_gap.append(issa_new[k][0])
                if n in (6, 8) and im in ("Cameraman", "Moon", "Clock"):       # CHK subset (18 instances)
                    row = {}
                    for m, fn in (("MDE", lambda s: gopt.MDE(P, n, seed=s)[1]), ("DE", lambda s: gopt.DE(P, n, seed=s)[1]), ("PSO", lambda s: gopt.PSO(P, n, seed=s)[1])):
                        gg = np.array([(fn(s) - g) / abs(g) for s in range(10)])
                        row[m] = (float(gg.mean()), e[m]["mean_relgap"])
                    chk.append((k, row))
        print("S120", c, flush=True)
    chk_ok = {m: sum(abs(r[m][0] - r[m][1]) <= 1e-12 + 1e-9 * abs(r[m][1]) for _, r in chk) for m in ("MDE", "DE", "PSO")}
    oldt = {m: np.array(v) for m, v in old.items()}
    cur = dict(oldt)                                                             # stored study (SDE randomized-F, legacy ISSA)
    mid = dict(oldt); mid["ISSA"] = np.array(issa_gap)                           # corrected ISSA only
    fin = dict(mid); fin["SDE"] = np.array(new_sde)                              # corrected ISSA + SDE with F = Cr = 0.25  (new main)
    sr = lambda a: float(np.mean(a))
    res = dict(n_instances=len(keys), seeds="0..9",
               check_stored_vs_fresh=dict(instances=len(chk), identical_mean_gap=chk_ok),
               summary_stored=friedman_summary(cur), summary_ISSA_corrected=friedman_summary(mid), summary_new_main=friedman_summary(fin),
               new_SDE_mean_SR=sr(new_sde_sr), new_SDE_mean_gap=float(np.mean(new_sde)), old_SDE_mean_gap=float(np.mean(oldt["SDE"])),
               new_SDE_instances=dict(zip(keys, zip(new_sde, new_sde_sr))))
    json.dump(res, open(os.path.join(DATA, "sde_rerun_stats120.json"), "w"), indent=1)
    print("CHK (fresh run identical to stored, of", len(chk), "instances):", chk_ok)
    print("stored   :", json.dumps(res["summary_stored"]["ranks"]), "chi2", round(res["summary_stored"]["chi2"], 1))
    print("ISSA fix :", json.dumps(res["summary_ISSA_corrected"]["ranks"]), "chi2", round(res["summary_ISSA_corrected"]["chi2"], 1))
    print("NEW MAIN :", json.dumps(res["summary_new_main"]["ranks"]), "chi2", round(res["summary_new_main"]["chi2"], 1), "p", res["summary_new_main"]["p"])
    print("Wilcoxon (new main):", json.dumps(res["summary_new_main"]["wilcoxon"]))
    print("SDE new mean SR", res["new_SDE_mean_SR"], "mean gap old/new", res["old_SDE_mean_gap"], res["new_SDE_mean_gap"])


if __name__ == "__main__":
    stats120(); table2()
