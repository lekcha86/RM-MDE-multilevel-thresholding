"""SDE sensitivity (Reviewer 1.9: baseline settings; deviation from Ali et al. 2014).

PROTOCOL (fixed before running)
  SDE variants (all NP = 20, G = 60, oppositional init, tournament-best base, dynamic in-place update):
    SDE-current : F = 0.5, Cr = 0.9, randomized scale F (0.5 + 0.5 u)   <- the main-study baseline
    SDE-fixedF  : F = 0.5, Cr = 0.9, F fixed (isolates the randomization)
    SDE-paper   : F = Cr = 0.25, F fixed                                  <- the settings of Ali et al. (2014); their NP = 10 D and
                                                                             200 iterations are NOT used, to keep the common budget
  Context at the same budget: DE (F = 0.5, Cr = 0.9), PSO (main-study settings), RM-MDE.
  Instances: 8 images x 3 criteria x n = 6, 7, 8, seeds 0..9 (720 runs per method); reference = DP optimum; success gap < 1e-8.
  Question: do the baseline comparison and the ranking change when the source paper's settings are used?
Outputs (data_results): sde_sensitivity.json, sde_sensitivity_runs.csv   (the original results are not touched)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv, time
import numpy as np
import paths
import criteria as cr, gopt
from ablation import all_images

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = paths.DATA
CRITS, NS, SEEDS, TOL = ("otsu", "kapur", "mcet"), (6, 7, 8), list(range(10)), 1e-8
METHODS = {
    "SDE-current": lambda P, n, s: gopt.SDE(P, n, seed=s),
    "SDE-fixedF": lambda P, n, s: gopt.SDE(P, n, seed=s, dither=False),
    "SDE-paper": lambda P, n, s: gopt.SDE(P, n, F=0.25, Cr=0.25, seed=s, dither=False),
    "DE": lambda P, n, s: gopt.DE(P, n, seed=s),
    "PSO": lambda P, n, s: gopt.PSO(P, n, seed=s),
    "RM-MDE": lambda P, n, s: gopt.MDE(P, n, seed=s),
}


def main():
    ref = json.load(open(os.path.join(OUT, "exact_optima_dp.json")))
    imgs = all_images(); cr.L = 256
    runs = []; t0 = time.time()
    for im, img in imgs.items():
        h = cr.histogram(img)
        for c in CRITS:
            P = cr.Problem(h, c)
            for n in NS:
                g = ref[f"{c}|{im}|{n}"]["g_dp"]
                for m, fn in METHODS.items():
                    for s in SEEDS:
                        f = fn(P, n, s)[1]
                        gap = abs(f - g) / abs(g); assert f - g >= -1e-9 * abs(g)
                        runs.append(dict(image=im, criterion=c, n=n, seed=s, method=m, relative_gap=gap, success=int(gap < TOL)))
        print(im, "done", f"{time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "sde_sensitivity_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(runs[0])); w.writeheader(); w.writerows(runs)
    summ = {}
    for m in METHODS:
        r = [x for x in runs if x["method"] == m]
        summ[m] = dict(runs=len(r), SR=100 * float(np.mean([x["success"] for x in r])), gap_mean=float(np.mean([x["relative_gap"] for x in r])),
                       gap_median=float(np.median([x["relative_gap"] for x in r])),
                       by_criterion={c: 100 * float(np.mean([x["success"] for x in r if x["criterion"] == c])) for c in CRITS},
                       by_n={n: 100 * float(np.mean([x["success"] for x in r if x["n"] == n])) for n in NS})
    idx = {(x["method"], x["image"], x["criterion"], x["n"], x["seed"]): x["relative_gap"] for x in runs}
    def wtl(a, b):
        w = t = l = 0
        for (m, im, c, n, s), g in idx.items():
            if m != a: continue
            h = idx[(b, im, c, n, s)]
            if abs(g - h) < 1e-8: t += 1
            elif g < h: w += 1
            else: l += 1
        return w, t, l
    pair = {f"{a} vs {b}": wtl(a, b) for a in ("RM-MDE",) for b in ("SDE-current", "SDE-fixedF", "SDE-paper", "DE", "PSO")}
    pair.update({f"{a} vs DE": wtl(a, "DE") for a in ("SDE-current", "SDE-fixedF", "SDE-paper")})
    pair.update({f"{a} vs PSO": wtl(a, "PSO") for a in ("SDE-current", "SDE-fixedF", "SDE-paper")})
    order = sorted(summ, key=lambda m: -summ[m]["SR"])
    json.dump(dict(protocol=__doc__, summary=summ, paired_WTL_first_strictly_better_tie_worse=pair, ranking_by_SR=order),
              open(os.path.join(OUT, "sde_sensitivity.json"), "w"), indent=1)
    for m in METHODS:
        s_ = summ[m]; print(f"{m:12s} SR {s_['SR']:5.1f}  mean gap {s_['gap_mean']:.2e}  by crit {', '.join(f'{c} {v:.0f}' for c, v in s_['by_criterion'].items())}  by n {', '.join(f'{n} {v:.0f}' for n, v in s_['by_n'].items())}")
    print("ranking by SR:", order)
    for k, v in pair.items(): print(f"{k:28s} W/T/L {v}")


if __name__ == "__main__":
    main()
