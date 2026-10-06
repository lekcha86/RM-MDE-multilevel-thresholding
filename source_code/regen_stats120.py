"""Regenerate the PSO, DE and RM-MDE entries of the 120-instance study (stats_otsu.json, stats_kapur.json, stats_mcet.json).

The script that produced these files in the first version is not in the code bundle. This script recomputes, for every
(criterion, image, n = 4..8), the mean relative gap and the success rate over seeds 0..9 against the stored reference value gstar
(which equals the DP optimum), and compares with the stored numbers.
Output: data_results/regen_stats120.json (differences) and a console summary.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json
import numpy as np
import paths
import criteria as cr, gopt
from ablation import all_images

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
cr.L = 256
TOL = 1e-8
imgs = all_images()
FN = {"PSO": lambda P, n, s: gopt.PSO(P, n, seed=s), "DE": lambda P, n, s: gopt.DE(P, n, seed=s), "MDE": lambda P, n, s: gopt.MDE(P, n, seed=s)}
rows, worst = [], {m: 0.0 for m in FN}
same = {m: 0 for m in FN}; total = 0
for c in ("otsu", "kapur", "mcet"):
    J = json.load(open(os.path.join(DATA, f"stats_{c}.json")))
    for im in J["images"]:
        P = cr.Problem(cr.histogram(imgs[im]), c)
        for n in range(4, 9):
            g = J[im][str(n)]["gstar"]; total += 1
            for m, fn in FN.items():
                f = np.array([fn(P, n, s)[1] for s in range(10)])
                gap = (f - g) / abs(g)
                mg, sr = float(gap.mean()), 100 * float(np.mean(gap < TOL))
                st = J[im][str(n)][m]
                d = abs(mg - st["mean_relgap"]); worst[m] = max(worst[m], d)
                ok = d <= 1e-12 + 1e-9 * abs(st["mean_relgap"]) and abs(sr - st["SR"]) < 1e-9
                same[m] += int(ok)
                rows.append(dict(criterion=c, image=im, n=n, method=m, regenerated_mean_gap=mg, stored_mean_gap=st["mean_relgap"], regenerated_SR=sr, stored_SR=st["SR"], identical=bool(ok)))
    print(c, "done", flush=True)
out = dict(instances=total, identical_instances=same, max_abs_diff_mean_gap=worst, rows=rows)
json.dump(out, open(os.path.join(DATA, "regen_stats120.json"), "w"), indent=1)
print("instances:", total, "| identical to stored:", same, "| max |difference| of mean gap:", worst)
