"""Experiment R2: wall-clock-matched comparison of PSO, DE and RM-MDE.

PROTOCOL (fixed before running; do not change mid-experiment)
  images      : Cameraman, Coins, Moon, Astronaut, Coffee, Cat, Clock, Histology
  criteria    : otsu, kapur, mcet
  thresholds  : n = 6, 7, 8
  seeds       : 0..9, the SAME seed value for all three methods within an instance
  reference   : DP-certified optimum (data_results/exact_optima_dp.json); never a method's own result
  success     : |J* - J| / |J*| < 1e-8
  budget      : wall-clock. RM-MDE runs its standard setting (NP=20, G=60, p_ls=0.15, F=.5, Cr=.9,
                4 sweeps). Its elapsed time for that (instance, seed) is T_target. PSO and DE (NP=20,
                same settings as the main study) then run with a generation loop until they have used
                at least 3*T_target. Reported budgets: 1x (the headline matched budget), 2x, 3x.
                The result "at budget B" is the best-so-far after the first generation that completes
                at or after B, i.e. each baseline is granted AT LEAST B (favours the baselines).
  threads     : 1 (single-threaded, runs executed sequentially)
Outputs (data_results): time_matched_DE_PSO.json / .csv, time_trace_DE_PSO_RMMDE.csv
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, time, csv
import numpy as np
import paths
import criteria as cr
from gopt import repair, rand_sol, local_search
from runtime_scaling import hardware, images as _img3   # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = paths.DATA
NS, SEEDS, MULTS = (6, 7, 8), list(range(10)), (1, 2, 3)
CRITS = ("otsu", "kapur", "mcet")
TOL = 1e-8


def all_images():
    from skimage import data, color, util
    g = lambda x: util.img_as_ubyte(color.rgb2gray(x))
    return {"Cameraman": data.camera(), "Coins": data.coins(), "Moon": data.moon(), "Astronaut": g(data.astronaut()),
            "Coffee": g(data.coffee()), "Cat": g(data.chelsea()), "Clock": data.clock(),
            "Histology": g(data.immunohistochemistry())}


# ---- exact copies of the loops in gopt.py, with wall-clock control and per-generation timestamps ----
def pso_t(prob, n, stop_t, seed, NP=20, max_gen=10**9, variant='ring-constriction'):
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    K, c1, c2 = 0.729, 2.05, 2.05                              # default: constriction, ring topology
    gbest = variant.startswith('gbest'); inertia = variant.endswith('inertia')
    if inertia:
        K, c1, c2 = 0.7298, 1.49618, 1.49618                     # the constriction coefficients written as an inertia weight (w = K)
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    V = rng.uniform(-1, 1, size=(NP, n))
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)])
    pb, pbf = X.copy(), fit.copy()
    tr = [(time.perf_counter() - t0, pbf.min())]
    while tr[-1][0] < stop_t and len(tr) - 1 < max_gen:
        for i in range(NP):
            nb = list(range(NP)) if gbest else [(i - 1) % NP, i, (i + 1) % NP]
            lb = nb[int(np.argmin(pbf[nb]))]
            if inertia:
                V[i] = K * V[i] + c1 * rng.random(n) * (pb[i] - X[i]) + c2 * rng.random(n) * (pb[lb] - X[i])
            else:
                V[i] = K * (V[i] + c1 * rng.random(n) * (pb[i] - X[i]) + c2 * rng.random(n) * (pb[lb] - X[i]))
            Ti = repair(X[i] + V[i], n); X[i] = Ti.astype(float)
            f = prob.obj(Ti)
            if f < pbf[i]:
                pb[i], pbf[i] = X[i].copy(), f
        tr.append((time.perf_counter() - t0, pbf.min()))
    return tr


def de_t(prob, n, stop_t, seed, NP=20, F=0.5, Cr=0.9, max_gen=10**9):
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)])
    tr = [(time.perf_counter() - t0, fit.min())]
    while tr[-1][0] < stop_t and len(tr) - 1 < max_gen:
        nX, nf = X.copy(), fit.copy()
        for i in range(NP):
            idxs = [j for j in range(NP) if j != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            V = X[r1] + F * (X[r2] - X[r3]); U = X[i].copy(); jr = rng.integers(n)
            for j in range(n):
                if rng.random() <= Cr or j == jr:
                    U[j] = V[j]
            Ti = repair(U, n); f = prob.obj(Ti)
            if f <= fit[i]:
                nX[i], nf[i] = Ti.astype(float), f
        X, fit = nX, nf
        tr.append((time.perf_counter() - t0, fit.min()))
    return tr


def mde_t(prob, n, seed, NP=20, G=60, F=0.5, Cr=0.9, pls=0.15):
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)])
    tr = [(time.perf_counter() - t0, fit.min())]
    for g in range(G):
        nX, nf = X.copy(), fit.copy()
        for i in range(NP):
            idxs = [j for j in range(NP) if j != i]
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            V = X[r1] + F * (X[r2] - X[r3]); U = X[i].copy(); jr = rng.integers(n)
            for j in range(n):
                if rng.random() <= Cr or j == jr:
                    U[j] = V[j]
            Ti = repair(U, n); f = prob.obj(Ti)
            if f <= fit[i]:
                if rng.random() < pls:
                    Ti = local_search(prob, Ti, max_sweeps=4); f = prob.obj(Ti)
                nX[i], nf[i] = Ti.astype(float), f
        X, fit = nX, nf
        tr.append((time.perf_counter() - t0, fit.min()))
    return tr


def at_budget(tr, B):
    """best-so-far after the first generation completing at or after B (else the last point)."""
    for t, f in tr:
        if t >= B:
            return t, f
    return tr[-1]


def compress(tr):
    out, best = [], np.inf
    for t, f in tr:
        if f < best:
            out.append((t, f)); best = f
    if out[-1][0] != tr[-1][0]:
        out.append((tr[-1][0], best))
    return out


def clopper(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo * 100), float(hi * 100)


def main():
    ref = json.load(open(os.path.join(OUT, "exact_optima_dp.json")))
    imgs = all_images()
    proto = dict(images=list(imgs), criteria=list(CRITS), thresholds=list(NS), seeds=SEEDS,
                 reference="DP-certified optimum (exact_optima_dp.json)", success_tolerance=TOL,
                 budget_type="wall-clock; T_target = RM-MDE elapsed for the same (instance, seed); baselines run to >=3*T_target",
                 reported_budget_multiples=list(MULTS), threads=1, hardware=hardware(),
                 settings=dict(NP=20, G_rmmde=60, p_ls=0.15, F=0.5, Cr=0.9, max_sweeps=4))
    runs, trace_rows = [], []
    # warm-up (imports, caches)
    cr.L = 256
    P0 = cr.Problem(cr.histogram(imgs["Cameraman"]), "otsu"); mde_t(P0, 6, 99); pso_t(P0, 6, 0.05, 99); de_t(P0, 6, 0.05, 99)
    total = len(imgs) * len(CRITS) * len(NS) * len(SEEDS); done = 0; t_start = time.time()
    for im, img in imgs.items():
        h = cr.histogram(img)
        for crit in CRITS:
            P = cr.Problem(h, crit)
            for n in NS:
                gstar = ref[f"{crit}|{im}|{n}"]["g_dp"]
                for seed in SEEDS:
                    tr_m = mde_t(P, n, seed); Tt = tr_m[-1][0]
                    trs = {"RM-MDE": tr_m,
                           "PSO": pso_t(P, n, MULTS[-1] * Tt, seed),
                           "DE": de_t(P, n, MULTS[-1] * Tt, seed)}
                    for m, tr in trs.items():
                        for k in MULTS:
                            B = k * Tt
                            if m == "RM-MDE":
                                t, f = tr[-1]
                                if k != 1:
                                    continue      # RM-MDE has one fixed budget (its own full run)
                            else:
                                t, f = at_budget(tr, B)
                            gap = abs(f - gstar) / abs(gstar)
                            runs.append(dict(image=im, criterion=crit, n=n, seed=seed, method=m, budget_mult=k,
                                             T_target=Tt, elapsed_time=t, objective=float(f), objective_gap=float(gap),
                                             success=int(abs(f - gstar) / abs(gstar) < TOL)))
                        for t, f in compress(tr):
                            trace_rows.append((im, crit, n, seed, m, t, float(f), abs(f - gstar) / abs(gstar)))
                    done += 1
                if done % 30 == 0:
                    print(f"{done}/{total}  {time.time()-t_start:.0f}s", flush=True)
    # ---- outputs ----
    cols = ["image", "criterion", "n", "seed", "method", "budget_mult", "T_target", "elapsed_time", "objective",
            "objective_gap", "success"]
    with open(os.path.join(OUT, "time_matched_DE_PSO.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader(); [w.writerow(r) for r in runs]
    with open(os.path.join(OUT, "time_trace_DE_PSO_RMMDE.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "criterion", "n", "seed", "method", "elapsed_time", "best_objective", "best_gap"])
        w.writerows(trace_rows)
    summ = []
    for crit in CRITS:
        for n in NS:
            for m, k in [("RM-MDE", 1)] + [(m, k) for m in ("PSO", "DE") for k in MULTS]:
                r = [x for x in runs if x["criterion"] == crit and x["n"] == n and x["method"] == m and x["budget_mult"] == k]
                s = sum(x["success"] for x in r)
                lo, hi = clopper(s, len(r))
                gaps = np.array([x["objective_gap"] for x in r])
                summ.append(dict(criterion=crit, n=n, method=m, budget_mult=k, runs=len(r), successes=s,
                                 SR=100 * s / len(r), SR_ci95=[lo, hi], gap_median=float(np.median(gaps)),
                                 gap_mean=float(gaps.mean()), mean_time=float(np.mean([x["elapsed_time"] for x in r]))))
    json.dump(dict(protocol=proto, summary=summ, runs=runs), open(os.path.join(OUT, "time_matched_DE_PSO.json"), "w"), indent=1)
    print("saved; total", time.time() - t_start, "s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # sanity: the time-aware copies reproduce gopt.py for the same seed (same RNG stream), G=60
        import gopt
        cr.L = 256
        for crit in CRITS:
            P = cr.Problem(cr.histogram(all_images()["Coins"]), crit)
            r = [(mde_t(P, 6, 3)[-1][1], gopt.MDE(P, 6, seed=3)[1]),
                 (pso_t(P, 6, 1e9, 3, max_gen=60)[-1][1], gopt.PSO(P, 6, seed=3)[1]),
                 (de_t(P, 6, 1e9, 3, max_gen=60)[-1][1], gopt.DE(P, 6, seed=3)[1])]
            print(crit, ["equal" if abs(a - b) <= 1e-12 * abs(b) else f"DIFF {a} {b}" for a, b in r])
    else:
        main()
