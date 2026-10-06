"""PSO / DE parameter sensitivity (Reviewer 1.9: were the competitors tuned?).

PROTOCOL (fixed before running; baselines only, RM-MDE is not re-run)
  Instances : 8 images x 3 criteria x n = 6, 7, 8 x seeds 0..9 = 720 runs per configuration; reference = DP optimum;
              success = gap < 1e-8. RM-MDE reference = the stored RM-MDE results of time_matched_DE_PSO.csv (same instances and seeds).
  Part 1 (fixed generations, as in the main study)
    DE  : F in {0.3, 0.5, 0.8} x Cr in {0.2, 0.5, 0.9}, NP = 20, G = 60                       (9 configurations; (0.5, 0.9) is the main one)
    PSO : ring + constriction (main), global-best + constriction, ring + inertia weight (the constriction coefficients written in inertia-weight form: w = 0.7298, c = 1.49618)  (NP = 20, G = 60)
    Population / generations at equal evaluations (1200): NP = 40, G = 30 and NP = 10, G = 120 for DE (F = 0.5, Cr = 0.9) and for PSO (main variant).
  Part 2 (seed-matched wall-clock budget, as in the matched-budget experiment)
    the DE configuration and the PSO configuration with the highest success in Part 1 are re-run until they have used at least the RM-MDE
    elapsed time T_target stored for the same (image, criterion, n, seed).
  Aim: not to find the best setting, but to see whether the superiority of RM-MDE changes when the baseline parameters vary within a reasonable range.
Outputs (data_results): baseline_sensitivity.json, baseline_sensitivity_runs.csv
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv, time
import numpy as np
import paths
import criteria as cr, gopt
from ablation import all_images
from time_matched import pso_t, de_t

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
CRITS, NS, SEEDS, TOL = ("otsu", "kapur", "mcet"), (6, 7, 8), list(range(10)), 1e-8

CONFIGS = {}
for F in (0.3, 0.5, 0.8):
    for Cr in (0.2, 0.5, 0.9):
        CONFIGS[f"DE F={F} Cr={Cr}"] = ("DE", lambda P, n, s, F=F, Cr=Cr: gopt.DE(P, n, NP=20, G=60, F=F, Cr=Cr, seed=s))
for v in ("ring-constriction", "gbest-constriction", "ring-inertia"):
    CONFIGS[f"PSO {v}"] = ("PSO", lambda P, n, s, v=v: gopt.PSO(P, n, NP=20, G=60, seed=s, variant=v))
for NP, G in ((40, 30), (10, 120)):
    CONFIGS[f"DE NP={NP} G={G}"] = ("DE", lambda P, n, s, NP=NP, G=G: gopt.DE(P, n, NP=NP, G=G, F=0.5, Cr=0.9, seed=s))
    CONFIGS[f"PSO NP={NP} G={G}"] = ("PSO", lambda P, n, s, NP=NP, G=G: gopt.PSO(P, n, NP=NP, G=G, seed=s))


def main():
    ref = json.load(open(os.path.join(DATA, "exact_optima_dp.json")))
    imgs = all_images(); cr.L = 256
    tm = list(csv.DictReader(open(os.path.join(DATA, "time_matched_DE_PSO.csv"))))
    Tt = {(r["image"], r["criterion"], int(r["n"]), int(r["seed"])): float(r["T_target"]) for r in tm if r["method"] == "RM-MDE"}
    rm_gap = {(r["image"], r["criterion"], int(r["n"]), int(r["seed"])): float(r["objective_gap"]) for r in tm if r["method"] == "RM-MDE"}
    assert len(Tt) == 720
    runs = []; t0 = time.time()
    probs = {(im, c): cr.Problem(cr.histogram(img), c) for im, img in imgs.items() for c in CRITS}
    def record(cfg, im, c, n, s, f, rt, part):
        g = ref[f"{c}|{im}|{n}"]["g_dp"]; gap = abs(f - g) / abs(g); assert f - g >= -1e-9 * abs(g)
        runs.append(dict(part=part, config=cfg, image=im, criterion=c, n=n, seed=s, relative_gap=gap, success=int(gap < TOL), runtime=rt))
    # ---- Part 1
    for k, (cfg, (kind, fn)) in enumerate(CONFIGS.items()):
        for im in imgs:
            for c in CRITS:
                for n in NS:
                    for s in SEEDS:
                        _, f, rt, _ = fn(probs[(im, c)], n, s)
                        record(cfg, im, c, n, s, f, rt, "fixed-G")
        print(f"part1 {k+1}/{len(CONFIGS)} {cfg}  SR {100*np.mean([r['success'] for r in runs if r['config']==cfg and r['part']=='fixed-G']):.1f}  {time.time()-t0:.0f}s", flush=True)
    # ---- Part 2: best DE grid config and best PSO variant, seed-matched wall clock
    def sr(cfg): return np.mean([r["success"] for r in runs if r["config"] == cfg and r["part"] == "fixed-G"])
    bestDE = max([c for c in CONFIGS if c.startswith("DE F=")], key=sr)
    bestPSO = max([c for c in CONFIGS if c.startswith("PSO ") and "NP=" not in c], key=sr)
    print("best DE:", bestDE, " best PSO:", bestPSO, flush=True)
    for cfg in (bestDE, bestPSO):
        for im in imgs:
            for c in CRITS:
                for n in NS:
                    for s in SEEDS:
                        T = Tt[(im, c, n, s)]
                        if cfg.startswith("DE"):
                            F = float(cfg.split("F=")[1].split()[0]); Cr = float(cfg.split("Cr=")[1])
                            tr = de_t(probs[(im, c)], n, T, s, F=F, Cr=Cr)
                        else:
                            tr = pso_t(probs[(im, c)], n, T, s, variant=cfg.split()[1])
                        record(cfg, im, c, n, s, tr[-1][1], tr[-1][0], "matched-time")
        print("part2", cfg, f"{time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(DATA, "baseline_sensitivity_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(runs[0])); w.writeheader(); w.writerows(runs)
    # ---- summaries
    summ = []
    for part in ("fixed-G", "matched-time"):
        for cfg in dict.fromkeys(r["config"] for r in runs if r["part"] == part):
            r = [x for x in runs if x["config"] == cfg and x["part"] == part]
            gaps = np.array([x["relative_gap"] for x in r])
            wtl = [0, 0, 0]
            for x in r:
                d = x["relative_gap"] - rm_gap[(x["image"], x["criterion"], x["n"], x["seed"])]
                wtl[0 if d > 1e-8 else (2 if d < -1e-8 else 1)] += 1          # RM-MDE better / tie / worse
            summ.append(dict(part=part, config=cfg, runs=len(r), SR=100 * float(np.mean([x["success"] for x in r])), gap_mean=float(gaps.mean()),
                             gap_median=float(np.median(gaps)), runtime_mean=float(np.mean([x["runtime"] for x in r])),
                             by_criterion={c: 100 * float(np.mean([x["success"] for x in r if x["criterion"] == c])) for c in CRITS},
                             by_n={n: 100 * float(np.mean([x["success"] for x in r if x["n"] == n])) for n in NS},
                             RMMDE_better_tie_worse=wtl))
    rm_sr = 100 * float(np.mean([r["gap"] < TOL for r in [dict(gap=v) for v in rm_gap.values()]]))
    json.dump(dict(protocol=__doc__, RM_MDE_reference=dict(SR=rm_sr, gap_mean=float(np.mean(list(rm_gap.values())))), best_DE=bestDE, best_PSO=bestPSO, summary=summ),
              open(os.path.join(DATA, "baseline_sensitivity.json"), "w"), indent=1)
    print(f"RM-MDE reference SR {rm_sr:.1f}")
    for s_ in summ:
        print(f"{s_['part']:12s} {s_['config']:28s} SR {s_['SR']:5.1f}  mean gap {s_['gap_mean']:.2e}  median {s_['gap_median']:.1e}  RM-MDE b/t/w {s_['RMMDE_better_tie_worse']}")
    print("saved", time.time() - t0)


if __name__ == "__main__":
    main()
