"""
Generic, criterion-agnostic optimizers operating on a criteria.Problem.
All MINIMIZE prob.obj(T). Generation-based, CPU-time measured.
Local search MAXIMIZES prob.score via O(1)-delta moves (shared by all criteria).
"""
import numpy as np, time, math
from criteria import L

# ---------------- encoding / repair ----------------
def repair(vec, n):
    v = np.clip(np.round(vec).astype(int), 2, L)
    v = np.sort(v)
    for k in range(1, n):
        if v[k] <= v[k - 1]:
            v[k] = min(v[k - 1] + 1, L)
    for k in range(n - 1, 0, -1):
        if v[k] <= v[k - 1]:
            v[k - 1] = max(v[k] - 1, 2)
    return v

def rand_sol(rng, n):
    return repair(rng.uniform(2, L, size=n), n)

# ---------------- exact O(1)-delta local search ----------------
def local_search(prob, T, max_sweeps=8):
    T = list(map(int, T)); n = len(T)
    bounds = [1] + T + [L + 1]
    improved, sweeps = True, 0
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
                new = prob.seg_S(bounds[k - 1], cand) + prob.seg_S(cand, bounds[k + 1])
                if new - base > best_gain + 1e-9:
                    best_gain, best_pos = new - base, cand
            if best_pos != bounds[k]:
                bounds[k] = best_pos; improved = True
    return np.array(bounds[1:n + 1])

# ---------------- exhaustive (validation, small n) ----------------
def exhaustive(prob, n):
    from itertools import combinations
    best_v, best_T = np.inf, None
    for c in combinations(range(2, L + 1), n):
        v = prob.obj(list(c))
        if v < best_v:
            best_v, best_T = v, list(c)
    return np.array(best_T), best_v

# ---------------- PSO (constriction, lbest ring) ----------------
def PSO(prob, n, NP=20, G=60, seed=0, variant='ring-constriction'):
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    K, c1, c2 = 0.729, 2.05, 2.05                              # default: constriction, ring topology
    gbest = variant.startswith('gbest'); inertia = variant.endswith('inertia')
    if inertia:
        K, c1, c2 = 0.7298, 1.49618, 1.49618                     # the constriction coefficients written as an inertia weight (w = K)
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    V = rng.uniform(-1, 1, size=(NP, n))
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)])
    pb, pbf = X.copy(), fit.copy(); hist = []
    for g in range(G):
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
        hist.append(pbf.min())
    bi = int(np.argmin(pbf)); return repair(pb[bi], n), pbf[bi], time.perf_counter() - t0, hist

# ---------------- DE rand/1/bin (deferred) ----------------
def DE(prob, n, NP=20, G=60, F=0.5, Cr=0.9, seed=0):
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)]); hist = []
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
                nX[i], nf[i] = Ti.astype(float), f
        X, fit = nX, nf; hist.append(fit.min())
    bi = int(np.argmin(fit)); return repair(X[bi], n), fit[bi], time.perf_counter() - t0, hist

# ---------------- SDE (OBL + tournament-best base + dynamic update) ----------------
def SDE(prob, n, NP=20, G=60, F=0.5, Cr=0.9, seed=0, dither=True):
    rng = np.random.default_rng(seed); t0 = time.perf_counter(); lo, hi = 2.0, float(L)
    P = rng.uniform(lo, hi, size=(NP, n)); OP = lo + hi - P
    cand = [repair(c, n) for c in np.vstack([P, OP])]
    cf = np.array([prob.obj(T) for T in cand]); o = np.argsort(cf)[:NP]
    X = np.array([cand[i].astype(float) for i in o]); fit = cf[o].copy(); hist = []
    for g in range(G):
        for i in range(NP):
            idxs = [j for j in range(NP) if j != i]
            trio = list(rng.choice(idxs, 3, replace=False))
            base = trio[int(np.argmin(fit[trio]))]; oth = [t for t in trio if t != base]
            Fi = F * (0.5 + 0.5 * rng.random()) if dither else F   # dither (randomized F) is NOT in Ali et al. (2014); dither=False gives their fixed F
            V = X[base] + Fi * (X[oth[0]] - X[oth[1]]); U = X[i].copy(); jr = rng.integers(n)
            for j in range(n):
                if rng.random() <= Cr or j == jr:
                    U[j] = V[j]
            Ti = repair(U, n); f = prob.obj(Ti)
            if f <= fit[i]:
                X[i], fit[i] = Ti.astype(float), f
        hist.append(fit.min())
    bi = int(np.argmin(fit)); return repair(X[bi], n), fit[bi], time.perf_counter() - t0, hist

# ---------------- ISSA (sparrow search + Levy + cosine inertia) ----------------
def _levy(rng, n, beta=1.5):
    sg = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
          (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sg, n); v = rng.normal(0, 1, n)
    return u / np.abs(v) ** (1 / beta)

# ---------------- ISSA (corrected one-dimensional adaptation of Wu & Yuan, 2022) ----------------
def ISSA(prob, n, NP=20, G=60, seed=0, PD=0.3, SD=0.2, ST=0.6, wmax=1.5, wmin=0.5):
    """Improved sparrow search algorithm adapted to the ordered-threshold vector of this study.

    Follows the update equations of Wu and Yuan (2022) with rank i = 1..NP in the sorted population:
      discoverers (best PD*NP): X exp(-i/(alpha G)), alpha ~ U(0,1], if R2 < ST;  else X + Q L, Q ~ N(0,1) scalar, L = ones
      entrants:  i > NP/2:  Q exp((X_worst - X)/i^2) (Q scalar);
                 otherwise: X_P + w |X - X_P| A+ L, A a +-1 row, A+ = A^T (A A^T)^-1, X_P = best discoverer position
                 after its update, w = wmax - (wmax - wmin) cos(pi t / G)       (cosine nonlinear inertia weight)
      vigilant (SD*NP distinct random sparrows): X_best + Levy |X - X_best| if f > f_best (Levy flight),
                 else X + K |X - X_worst| / (f - f_worst + eps), K ~ U(-1, 1)
    Greedy acceptance: a candidate replaces its sparrow only if its fitness is better; a rejected move leaves
    both the position and the fitness unchanged, so position and fitness are always consistent.
    Adaptation (not in the original, which targets 2-D entropy thresholding with a threshold pair): positions
    are threshold vectors in bin units, evaluated through the common repair (round, clip, sort, separate) and
    the common objective; the returned vector is checked to attain the reported fitness."""
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)])
    nprod = max(1, int(PD * NP)); nsd = max(1, int(SD * NP)); ones = np.ones(n); hist = []

    def accept(i, cand):
        Ti = repair(cand, n); f = prob.obj(Ti)
        if f < fit[i]:
            X[i], fit[i] = Ti.astype(float), f

    for t in range(G):
        order = np.argsort(fit); X, fit = X[order], fit[order]
        Xbest, Xworst, fbest, fworst = X[0].copy(), X[-1].copy(), fit[0], fit[-1]
        w = wmax - (wmax - wmin) * np.cos(np.pi * t / G)
        for i in range(nprod):                                       # discoverers
            if rng.random() < ST:
                cand = X[i] * np.exp(-(i + 1) / (rng.random() * G + 1e-9))
            else:
                cand = X[i] + rng.normal() * ones
            accept(i, cand)
        XP = X[int(np.argmin(fit[:nprod]))].copy()
        for i in range(nprod, NP):                                   # entrants
            if (i + 1) > NP / 2:
                cand = rng.normal() * np.exp((Xworst - X[i]) / (i + 1) ** 2)
            else:
                A = rng.choice([-1.0, 1.0], n); Aplus = A / n
                cand = XP + w * float(np.abs(X[i] - XP) @ Aplus) * ones
            accept(i, cand)
        for i in rng.choice(NP, nsd, replace=False):                 # vigilant sparrows
            if fit[i] > fbest:
                cand = Xbest + _levy(rng, n) * np.abs(X[i] - Xbest)
            else:
                cand = X[i] + rng.uniform(-1, 1) * np.abs(X[i] - Xworst) / (fit[i] - fworst + 1e-12)
            accept(i, cand)
        hist.append(fit.min())
    bi = int(np.argmin(fit)); T = repair(X[bi], n); f = float(fit[bi])
    if abs(prob.obj(T) - f) > 1e-9 * max(1.0, abs(f)):
        raise RuntimeError("ISSA invariant violated: returned thresholds do not attain the stored fitness")
    return T, f, time.perf_counter() - t0, hist


# ---------------- MDE (PROPOSED): memetic DE + exact local search ----------------
def MDE(prob, n, NP=20, G=60, F=0.5, Cr=0.9, seed=0, pls=0.15):
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)]); hist = []
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
        X, fit = nX, nf; hist.append(fit.min())
    bi = int(np.argmin(fit)); return repair(X[bi], n), fit[bi], time.perf_counter() - t0, hist

# The method set and settings used in the study are defined in rerun_main.py (METHODS).
