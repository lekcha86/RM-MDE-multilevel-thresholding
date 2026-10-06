"""DEFECTIVE first-version ISSA, kept ONLY for audit. Do not use for any reported result.

Defect: in the discoverer and entrant branches a rejected move was not undone, so the positions drifted away from the stored fitness
values and the returned thresholds did not attain the reported fitness (720 of 720 audit runs). The corrected implementation is
source_code/gopt.py :: ISSA.
"""
import os, sys, time, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source_code"))
from gopt import repair, rand_sol, _levy


def ISSA_legacy(prob, n, NP=20, G=60, seed=0, PD=0.3, SD=0.2, ST=0.6):
    """LEGACY, KNOWN-DEFECTIVE version used for the first submission; kept only for audit (issa_audit.py).
    Defect: in the discoverer and entrant branches the new position is written into X[i] before it is
    evaluated and is NOT restored when rejected, so positions drift away from the stored fitness values and
    the returned thresholds do not attain the reported fitness. Do not use; see ISSA below."""
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)]); hist = []
    nprod = max(1, int(PD * NP)); nsd = max(1, int(SD * NP))
    for g in range(G):
        order = np.argsort(fit); X, fit = X[order], fit[order]
        Xbest, Xworst = X[0].copy(), X[-1].copy(); fbest = fit[0]
        w = 1.5 - (1.5 - 0.5) * np.cos(np.pi * g / G)      # cosine nonlinear inertia
        for i in range(NP):
            if i < nprod:                                    # discoverers
                if rng.random() < ST:
                    X[i] = X[i] * np.exp(-(i + 1) / (rng.random() * G + 1e-9))
                else:
                    X[i] = X[i] + rng.normal(size=n)
            else:                                            # entrants
                if i > NP / 2:
                    X[i] = rng.normal(size=n) * np.exp((Xworst - X[i]) / (i + 1) ** 2)
                else:
                    X[i] = Xbest + w * np.abs(X[i] - Xbest) * rng.choice([-1, 1], n)
            Ti = repair(X[i], n); f = prob.obj(Ti)
            if f < fit[i]:
                X[i], fit[i] = Ti.astype(float), f
        for _ in range(nsd):                                 # vigilant (Levy)
            i = rng.integers(NP)
            if fit[i] > fbest:
                Xi = Xbest + _levy(rng, n) * np.abs(X[i] - Xbest)
            else:
                Xi = X[i] + _levy(rng, n) * np.abs(X[i] - Xworst) / (fit[i] - fit[-1] + 1e-12)
            Ti = repair(Xi, n); f = prob.obj(Ti)
            if f < fit[i]:
                X[i], fit[i] = Ti.astype(float), f
        hist.append(fit.min())
    bi = int(np.argmin(fit)); return repair(X[bi], n), fit[bi], time.perf_counter() - t0, hist
