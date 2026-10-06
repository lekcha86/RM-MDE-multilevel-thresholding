"""Ablation runner for Reviewer 2, comment 5 (Experiments A and B).

One configurable memetic-DE loop; every variant shares the objective, repair, bounds, initialization,
NP=20, G=60, F=0.5, Cr=0.9, the 4-sweep local-search cap, the seed list and the DP-certified
reference. Only the factor under study changes.

PROTOCOL (fixed before running)
  criterion  : MCET (formulation A, x = i). Images: the eight of the main study. n = 6, 7, 8. Seeds 0..9.
  reference  : DP-certified optimum (data_results/exact_optima_dp.json); success: |J-J*|/|J*| < 1e-8.
  Experiment A (donor isolation): donor in {rand/1, best/1, SDE donor} x LS in {none, p_ls = 0.15}.
      rand/1 : x_r1 + F (x_r2 - x_r3)
      best/1 : x_best + F (x_r1 - x_r2)                       (best of the current population)
      SDE    : tournament-best of three random members as base, randomized F_i = F (0.5 + 0.5 u);
               this isolates the SDE DONOR only (no opposition-based initialization, synchronous update),
               so it is not the full SDE of the main study.
  Experiment B (p_ls sensitivity, donor rand/1): p_ls in {0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.00}.
      p_ls is the probability of applying the local search to an ACCEPTED trial; p_ls = 1 refines every
      accepted trial.
Recorded per run: objective, gap, success, runtime (diversity bookkeeping excluded), generations,
  accepted trials, LS_calls, LS_sweeps, LS_moves, LS_probes, final diversity, best-by-generation, diversity traces.
Outputs (data_results): ablation_<exp>.json, ablation_<exp>_runs.csv, ablation_<exp>_trace.csv
Usage: python ablation.py check | A | B
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, csv, time
import numpy as np
import paths
import criteria as cr
import gopt
from gopt import repair, rand_sol

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = paths.DATA
NS, SEEDS, TOL = (6, 7, 8), list(range(10)), 1e-8
L = 256


def local_search_counted(prob, T, max_sweeps=4):
    """Identical to gopt.local_search, plus work counters (verified equal in `check`)."""
    T = list(map(int, T)); n = len(T)
    bounds = [1] + T + [L + 1]
    improved, sweeps, moves, probes = True, 0, 0, 0
    while improved and sweeps < max_sweeps:
        improved = False; sweeps += 1
        for k in range(1, n + 1):
            lo, hi = bounds[k - 1] + 1, bounds[k + 1] - 1
            if lo > hi:
                continue
            base = prob.seg_S(bounds[k - 1], bounds[k]) + prob.seg_S(bounds[k], bounds[k + 1])
            best_pos, best_gain = bounds[k], 0.0
            for cand in range(lo, hi + 1):
                if cand == bounds[k]:
                    continue
                probes += 1
                new = prob.seg_S(bounds[k - 1], cand) + prob.seg_S(cand, bounds[k + 1])
                if new - base > best_gain + 1e-9:
                    best_gain, best_pos = new - base, cand
            if best_pos != bounds[k]:
                bounds[k] = best_pos; improved = True; moves += 1
    return np.array(bounds[1:n + 1]), sweeps, moves, probes


def diversity(X):
    c = X.mean(axis=0)
    return float(np.mean(np.linalg.norm(X - c, axis=1))), float(np.mean(X.std(axis=0)))


def memetic(prob, n, seed, donor="rand", pls=0.0, NP=20, G=60, F=0.5, Cr=0.9, ls_sweeps=4):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter(); tdiv = 0.0
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)])
    st = dict(accepted=0, LS_calls=0, LS_sweeps=0, LS_moves=0, LS_probes=0)
    hist = [float(fit.min())]
    times = [time.perf_counter() - t0 - tdiv]
    t1 = time.perf_counter(); d = diversity(X); tdiv += time.perf_counter() - t1
    dtr = [d]
    for g in range(G):
        nX, nf = X.copy(), fit.copy()
        for i in range(NP):
            idxs = [j for j in range(NP) if j != i]
            if donor == "rand":
                r1, r2, r3 = rng.choice(idxs, 3, replace=False)
                V = X[r1] + F * (X[r2] - X[r3])
            elif donor == "best":
                b = int(np.argmin(fit)); r1, r2 = rng.choice(idxs, 2, replace=False)
                V = X[b] + F * (X[r1] - X[r2])
            elif donor in ("sde", "sde_fixed"):
                trio = list(rng.choice(idxs, 3, replace=False))
                base = trio[int(np.argmin(fit[trio]))]; oth = [t for t in trio if t != base]
                Fi = F * (0.5 + 0.5 * rng.random()) if donor == "sde" else F      # sde_fixed: no randomized scale factor
                V = X[base] + Fi * (X[oth[0]] - X[oth[1]])
            else:
                raise ValueError(donor)
            U = X[i].copy(); jr = rng.integers(n)
            for j in range(n):
                if rng.random() <= Cr or j == jr:
                    U[j] = V[j]
            Ti = repair(U, n); f = prob.obj(Ti)
            if f <= fit[i]:
                st["accepted"] += 1
                if pls >= 1.0 or (0.0 < pls < 1.0 and rng.random() < pls):
                    Ti, sw, mv, pr = local_search_counted(prob, Ti, ls_sweeps); f = prob.obj(Ti)
                    st["LS_calls"] += 1; st["LS_sweeps"] += sw; st["LS_moves"] += mv; st["LS_probes"] += pr
                nX[i], nf[i] = Ti.astype(float), f
        X, fit = nX, nf
        hist.append(float(fit.min()))
        times.append(time.perf_counter() - t0 - tdiv)
        t1 = time.perf_counter(); dtr.append(diversity(X)); tdiv += time.perf_counter() - t1
    runtime = time.perf_counter() - t0 - tdiv
    bi = int(np.argmin(fit))
    return dict(T=repair(X[bi], n).tolist(), obj=float(fit[bi]), runtime=runtime, generations=G, hist=hist, times=times, div=dtr, **st)


EXPERIMENTS = {
    "A": [("rand/1", "rand", 0.0), ("rand/1+LS", "rand", 0.15),
          ("best/1", "best", 0.0), ("best/1+LS", "best", 0.15),
          ("SDE-donor", "sde", 0.0), ("SDE-donor+LS", "sde", 0.15)],
    # A2: the synergetic donor without the randomized scale factor (which Ali et al. 2014 do not use), at the common
    # F = 0.5, Cr = 0.9, and with the source paper's F = Cr = 0.25
    "A2": [("SDE-donor fixedF", "sde_fixed", 0.0), ("SDE-donor fixedF+LS", "sde_fixed", 0.15),
           ("SDE-donor paper(F=Cr=.25)", "sde_fixed", 0.0, dict(F=0.25, Cr=0.25)),
           ("SDE-donor paper(F=Cr=.25)+LS", "sde_fixed", 0.15, dict(F=0.25, Cr=0.25))],
    "B": [("rand/1 p=0", "rand", 0.0), ("rand/1 p=0.05", "rand", 0.05), ("rand/1 p=0.10", "rand", 0.10),
          ("rand/1 p=0.15", "rand", 0.15), ("rand/1 p=0.25", "rand", 0.25),
          ("rand/1 p=0.50", "rand", 0.5), ("rand/1 p=1.00", "rand", 1.0)],
}


def all_images():
    from skimage import data, color, util
    g = lambda x: util.img_as_ubyte(color.rgb2gray(x))
    return {"Cameraman": data.camera(), "Coins": data.coins(), "Moon": data.moon(), "Astronaut": g(data.astronaut()),
            "Coffee": g(data.coffee()), "Cat": g(data.chelsea()), "Clock": data.clock(),
            "Histology": g(data.immunohistochemistry())}


def check():
    """Step 3: variants must be consistent with the main-study code and with each other."""
    cr.L = L
    P = cr.Problem(cr.histogram(all_images()["Coins"]), "mcet")
    # (i) counted local search == gopt.local_search on random starts
    rng = np.random.default_rng(0); bad = 0
    for _ in range(100):
        n = int(rng.integers(3, 9)); T = rand_sol(rng, n)
        a = gopt.local_search(P, T, 4); b = local_search_counted(P, T, 4)[0]
        bad += int(not np.array_equal(a, b))
    print("local-search copies differ from gopt.local_search in", bad, "of 100 random starts")
    # (ii) rand/1, p=0 reproduces gopt.DE; rand/1, p=.15 reproduces gopt.MDE (same seed)
    for n in (6, 8):
        for s in (0, 3):
            a = memetic(P, n, s, "rand", 0.0)["obj"]; b = gopt.DE(P, n, seed=s)[1]
            c = memetic(P, n, s, "rand", 0.15)["obj"]; d = gopt.MDE(P, n, seed=s)[1]
            print(f"n={n} seed={s}  DE {'equal' if a == b else 'DIFF'}  RM-MDE {'equal' if c == d else 'DIFF'}")
    # (iii) all variants return feasible ordered thresholds and a finite objective on the same instance
    for name, donor, p in EXPERIMENTS["A"] + EXPERIMENTS["B"]:
        r = memetic(P, 7, 1, donor, p)
        T = r["T"]
        assert all(T[k] < T[k + 1] for k in range(len(T) - 1)) and 2 <= T[0] and T[-1] <= L and np.isfinite(r["obj"])
        print(f"{name:16s} obj {r['obj']:.6e}  LS_calls {r['LS_calls']:3d} sweeps {r['LS_sweeps']:3d} moves {r['LS_moves']:3d} "
              f"accepted {r['accepted']:4d}  t {r['runtime']:.3f}s  div {r['div'][0][0]:.1f}->{r['div'][-1][0]:.1f}")


def run(exp):
    ref = json.load(open(os.path.join(OUT, "exact_optima_dp.json")))
    variants = EXPERIMENTS[exp]
    imgs = all_images(); cr.L = L
    runs, trace = [], []
    total = len(imgs) * len(NS) * len(SEEDS) * len(variants); k = 0; ts = time.time()
    for im, img in imgs.items():
        P = cr.Problem(cr.histogram(img), "mcet")
        for n in NS:
            gstar = ref[f"mcet|{im}|{n}"]["g_dp"]
            for v in variants:
                name, donor, p = v[:3]; kw = v[3] if len(v) > 3 else {}
                for s in SEEDS:
                    r = memetic(P, n, s, donor, p, **kw)
                    gap = abs(r["obj"] - gstar) / abs(gstar)
                    assert r["obj"] - gstar >= -1e-9 * abs(gstar)        # nothing may beat the certified optimum
                    rid = len(runs)
                    runs.append(dict(run_id=rid, image=im, criterion="mcet", n=n, seed=s, variant=name, donor=donor, p_ls=p,
                                     objective=r["obj"], optimal_objective=gstar, relative_gap=gap, success=int(gap < TOL),
                                     runtime=r["runtime"], generations=r["generations"], accepted=r["accepted"],
                                     LS_calls=r["LS_calls"], LS_sweeps=r["LS_sweeps"], LS_moves=r["LS_moves"],
                                     LS_probes=r["LS_probes"], final_diversity_l2=r["div"][-1][0], final_diversity_coord=r["div"][-1][1]))
                    for g, (h, d) in enumerate(zip(r["hist"], r["div"])):
                        trace.append((rid, g, h, (h - gstar) / abs(gstar), d[0], d[1]))
                    k += 1
            print(f"{im} n={n} done  {k}/{total}  {time.time()-ts:.0f}s", flush=True)
    cols = list(runs[0])
    with open(os.path.join(OUT, f"ablation_{exp}_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(runs)
    with open(os.path.join(OUT, f"ablation_{exp}_trace.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["run_id", "generation", "best_objective", "best_gap", "diversity_l2", "diversity_coord"])
        w.writerows(trace)
    summ = []
    for v in variants:
        name, donor, p = v[:3]
        for n in list(NS) + ["all"]:
            r = [x for x in runs if x["variant"] == name and (n == "all" or x["n"] == n)]
            summ.append(dict(variant=name, n=n, runs=len(r), SR=100 * float(np.mean([x["success"] for x in r])),
                             gap_mean=float(np.mean([x["relative_gap"] for x in r])),
                             gap_median=float(np.median([x["relative_gap"] for x in r])),
                             runtime_median=float(np.median([x["runtime"] for x in r])),
                             LS_calls_mean=float(np.mean([x["LS_calls"] for x in r])),
                             LS_calls_per_accepted=float(np.mean([x["LS_calls"] / max(x["accepted"], 1) for x in r])),
                             time_median_successful=float(np.median([x["runtime"] for x in r if x["success"]] or [float("nan")])),
                             LS_sweeps_mean=float(np.mean([x["LS_sweeps"] for x in r])),
                             LS_moves_mean=float(np.mean([x["LS_moves"] for x in r])),
                             LS_probes_mean=float(np.mean([x["LS_probes"] for x in r])),
                             final_div_l2_mean=float(np.mean([x["final_diversity_l2"] for x in r]))))
    proto = dict(experiment=exp, variants=[v[0] for v in variants], images=list(imgs), n=list(NS), seeds=SEEDS,
                 criterion="mcet (x=i)", reference="DP-certified optimum", tolerance=TOL, NP=20, G=60, F=0.5, Cr=0.9,
                 ls_sweep_cap=4, threads=1)
    json.dump(dict(protocol=proto, summary=summ), open(os.path.join(OUT, f"ablation_{exp}.json"), "w"), indent=1)
    print("saved", exp, time.time() - ts, "s")


if __name__ == "__main__":
    m = sys.argv[1]
    check() if m == "check" else run(m)
