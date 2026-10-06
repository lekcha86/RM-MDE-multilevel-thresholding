"""Experiment C: RM-MDE versus time-matched multistart exact local search (Reviewer 2, comment 5).

PROTOCOL (fixed before running)
  images      : the eight of the main study; criteria otsu, kapur, mcet (MCET with x = i); n = 6, 7, 8; seeds 0..9
  instances   : 8 x 3 x 3 x 10 = 720 paired runs
  reference   : DP-certified optimum (exact_optima_dp.json); success |J-J*|/|J*| < 1e-8
  budget      : wall-clock. For each (image, criterion, n, seed) RM-MDE is run in its standard setting
                (NP=20, G=60, p_ls=0.15, 4-sweep cap); its runtime T_target (diversity bookkeeping excluded)
                is the budget of both multistart methods. Each multistart method always completes the
                start in progress when the budget expires (so it is granted at least T_target).
  C1 (main)   : repeat { random feasible start (seeded rng, seed = run seed) -> exact coordinate local search,
                at most 4 sweeps (the SAME primitive and cap as RM-MDE) } until the budget is used; keep the best.
  C2 (supp.)  : the same, but each local search runs until no move improves (cap 1000 sweeps).
  Recorded    : success, gap, runtime, number of starts, LS sweeps / moves / probes, time-to-success,
                best-so-far trace. threads = 1.
Outputs (data_results): multistart_C.json, multistart_C_runs.csv, multistart_C_trace.csv
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, csv, time
import numpy as np
import paths
import criteria as cr
from gopt import rand_sol
from ablation import memetic, local_search_counted, all_images
from runtime_scaling import hardware

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = paths.DATA
NS, SEEDS, TOL = (6, 7, 8), list(range(10)), 1e-8
CRITS = ("otsu", "kapur", "mcet")


def multistart(prob, n, seed, budget, sweeps):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best, tr = np.inf, []
    st = dict(starts=0, sweeps=0, moves=0, probes=0)
    while True:
        T0 = rand_sol(rng, n)
        T, sw, mv, pr = local_search_counted(prob, T0, sweeps)
        f = prob.obj(T)
        st["starts"] += 1; st["sweeps"] += sw; st["moves"] += mv; st["probes"] += pr
        t = time.perf_counter() - t0
        if f < best:
            best = float(f); tr.append((t, best))
        if t >= budget:
            break
    tr.append((t, best))
    return best, t, tr, st


def first_success(tr, gstar):
    for t, f in tr:
        if abs(f - gstar) / abs(gstar) < TOL:
            return t
    return float("nan")


def main():
    ref = json.load(open(os.path.join(OUT, "exact_optima_dp.json")))
    imgs = all_images(); cr.L = 256
    runs, trace = [], []
    # warm-up
    P0 = cr.Problem(cr.histogram(imgs["Cameraman"]), "otsu"); memetic(P0, 6, 99, "rand", 0.15); multistart(P0, 6, 99, 0.05, 4)
    total = len(imgs) * len(CRITS) * len(NS) * len(SEEDS); k = 0; t_start = time.time()
    for im, img in imgs.items():
        h = cr.histogram(img)
        for crit in CRITS:
            P = cr.Problem(h, crit)
            for n in NS:
                gstar = ref[f"{crit}|{im}|{n}"]["g_dp"]
                for seed in SEEDS:
                    r = memetic(P, n, seed, "rand", 0.15)
                    Tt = r["runtime"]
                    mde_tr = [(t, h_) for t, h_ in zip(r["times"], r["hist"])]
                    out = {"RM-MDE": dict(obj=r["obj"], t=Tt, tr=mde_tr, starts=float("nan"), sweeps=r["LS_sweeps"],
                                          moves=r["LS_moves"], probes=r["LS_probes"], ls_calls=r["LS_calls"])}
                    for name, sw in (("MS-LS-4S", 4), ("MS-LS-conv", 1000)):
                        f, t, tr, st = multistart(P, n, seed, Tt, sw)
                        out[name] = dict(obj=f, t=t, tr=tr, starts=st["starts"], sweeps=st["sweeps"], moves=st["moves"],
                                         probes=st["probes"], ls_calls=st["starts"])
                    for m, o in out.items():
                        gap = abs(o["obj"] - gstar) / abs(gstar)
                        assert o["obj"] - gstar >= -1e-9 * abs(gstar)
                        runs.append(dict(image=im, criterion=crit, n=n, seed=seed, method=m, T_target=Tt, elapsed_time=o["t"],
                                         objective=o["obj"], optimal_objective=gstar, relative_gap=gap, success=int(gap < TOL),
                                         starts=o["starts"], LS_calls=o["ls_calls"], LS_sweeps=o["sweeps"], LS_moves=o["moves"],
                                         LS_probes=o["probes"], time_to_success=first_success(o["tr"], gstar)))
                        best = np.inf
                        for t, f in o["tr"]:
                            if f < best:
                                best = f; trace.append((im, crit, n, seed, m, t, float(f), abs(f - gstar) / abs(gstar)))
                    k += 1
            print(f"{im} {crit} done {k}/{total} {time.time()-t_start:.0f}s", flush=True)
    cols = list(runs[0])
    with open(os.path.join(OUT, "multistart_C_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(runs)
    with open(os.path.join(OUT, "multistart_C_trace.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "criterion", "n", "seed", "method", "elapsed_time", "best_objective", "best_gap"])
        w.writerows(trace)
    summ = []
    for crit in list(CRITS) + ["all"]:
        for n in list(NS) + ["all"]:
            for m in ("RM-MDE", "MS-LS-4S", "MS-LS-conv"):
                r = [x for x in runs if x["method"] == m and (crit == "all" or x["criterion"] == crit) and (n == "all" or x["n"] == n)]
                ts = [x["time_to_success"] for x in r if x["success"]]
                summ.append(dict(criterion=crit, n=n, method=m, runs=len(r), SR=100 * float(np.mean([x["success"] for x in r])),
                                 gap_mean=float(np.mean([x["relative_gap"] for x in r])),
                                 gap_median=float(np.median([x["relative_gap"] for x in r])),
                                 time_median=float(np.median([x["elapsed_time"] for x in r])),
                                 starts_mean=float(np.mean([x["starts"] for x in r])),
                                 sweeps_mean=float(np.mean([x["LS_sweeps"] for x in r])),
                                 moves_mean=float(np.mean([x["LS_moves"] for x in r])),
                                 probes_mean=float(np.mean([x["LS_probes"] for x in r])),
                                 time_to_success_median=float(np.median(ts)) if ts else float("nan")))
    proto = dict(images=list(imgs), criteria=list(CRITS), n=list(NS), seeds=SEEDS, reference="DP-certified optimum",
                 tolerance=TOL, budget="RM-MDE runtime for the same (instance, seed); multistart completes the start in progress",
                 C1="random start + exact LS (<=4 sweeps), repeated", C2="random start + exact LS to convergence (cap 1000), repeated",
                 threads=1, hardware=hardware())
    json.dump(dict(protocol=proto, summary=summ), open(os.path.join(OUT, "multistart_C.json"), "w"), indent=1)
    print("saved", time.time() - t_start, "s")


if __name__ == "__main__":
    main()
