"""ISSA implementation audit (Reviewer 2, comment 3).

Compares, on the same instances and seeds, with NP = 20, G = 60 and the DP-certified reference:
  as-run   : gopt.ISSA exactly as used in the main study
  fixed    : gopt.ISSA with ONE change, greedy acceptance made real (a rejected move is undone)
  paper    : ISSA written from the update equations of Wu and Yuan (2022) as extracted from the full text
             (discoverers X exp(-i/(alpha G)) or X + Q with a scalar Q; entrants Q exp((Xworst - X)/i^2) or
              X_P + w |X - X_P| A+ L with w = wmax - (wmax - wmin) cos(pi t / G), wmax = 1.5, wmin = 0.5;
              vigilant: Xbest + Levy |X - Xbest| or X + K |X - Xworst| / (f - f_worst + eps), K ~ U(-1, 1);
              greedy acceptance; PD = 0.3, vigilant fraction 0.2, ST = 0.6)
Also reports how often the threshold vector returned by `as-run` is not the one that attains its reported fitness.
Usage: python issa_audit.py
"""
import os
import json, math, sys
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source_code"))
import criteria as cr
import gopt
from issa_legacy import ISSA_legacy
from gopt import repair, rand_sol, _levy
from ablation import all_images

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
L = 256
CRITS = ("otsu", "kapur", "mcet")
NS, SEEDS, TOL = (6, 7, 8), list(range(10)), 1e-8


def issa_fixed(prob, n, seed, NP=20, G=60, PD=0.3, SD=0.2, ST=0.6):
    """gopt.ISSA with the rejected-move bug removed (nothing else changed)."""
    rng = np.random.default_rng(seed)
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)])
    nprod = max(1, int(PD * NP)); nsd = max(1, int(SD * NP))
    for g in range(G):
        order = np.argsort(fit); X, fit = X[order], fit[order]
        Xbest, Xworst = X[0].copy(), X[-1].copy(); fbest = fit[0]
        w = 1.5 - (1.5 - 0.5) * np.cos(np.pi * g / G)
        for i in range(NP):
            old = X[i].copy()
            if i < nprod:
                if rng.random() < ST:
                    X[i] = X[i] * np.exp(-(i + 1) / (rng.random() * G + 1e-9))
                else:
                    X[i] = X[i] + rng.normal(size=n)
            else:
                if i > NP / 2:
                    X[i] = rng.normal(size=n) * np.exp((Xworst - X[i]) / (i + 1) ** 2)
                else:
                    X[i] = Xbest + w * np.abs(X[i] - Xbest) * rng.choice([-1, 1], n)
            Ti = repair(X[i], n); f = prob.obj(Ti)
            if f < fit[i]:
                X[i], fit[i] = Ti.astype(float), f
            else:
                X[i] = old
        for _ in range(nsd):
            i = rng.integers(NP)
            if fit[i] > fbest:
                Xi = Xbest + _levy(rng, n) * np.abs(X[i] - Xbest)
            else:
                Xi = X[i] + _levy(rng, n) * np.abs(X[i] - Xworst) / (fit[i] - fit[-1] + 1e-12)
            Ti = repair(Xi, n); f = prob.obj(Ti)
            if f < fit[i]:
                X[i], fit[i] = Ti.astype(float), f
    bi = int(np.argmin(fit))
    return repair(X[bi], n), float(fit[bi])


def issa_paper(prob, n, seed, NP=20, G=60, PD=0.3, SD=0.2, ST=0.6, wmax=1.5, wmin=0.5):
    rng = np.random.default_rng(seed)
    X = np.array([rand_sol(rng, n).astype(float) for _ in range(NP)])
    fit = np.array([prob.obj(repair(X[i], n)) for i in range(NP)])
    nprod = max(1, int(PD * NP)); nsd = max(1, int(SD * NP)); ones = np.ones(n)

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
                cand = X[i] + rng.normal() * ones                  # scalar Q, L = ones
            accept(i, cand)
        XP = X[int(np.argmin(fit[:nprod]))].copy()                   # best discoverer after its update
        for i in range(nprod, NP):                                   # entrants
            if (i + 1) > NP / 2:
                cand = rng.normal() * np.exp((Xworst - X[i]) / (i + 1) ** 2)
            else:
                A = rng.choice([-1.0, 1.0], n); Aplus = A / n          # A+ = A^T (A A^T)^-1 for a 1 x n sign row
                cand = XP + w * float(np.abs(X[i] - XP) @ Aplus) * ones
            accept(i, cand)
        for _ in range(nsd):                                         # vigilant sparrows
            i = rng.integers(NP)
            if fit[i] > fbest:
                cand = Xbest + _levy(rng, n) * np.abs(X[i] - Xbest)
            else:
                K = rng.uniform(-1, 1)
                cand = X[i] + K * np.abs(X[i] - Xworst) / (fit[i] - fworst + 1e-12)
            accept(i, cand)
    bi = int(np.argmin(fit))
    return repair(X[bi], n), float(fit[bi])


def main():
    import paths
    ref = json.load(open(os.path.join(paths.DATA, "exact_optima_dp.json")))
    imgs = all_images(); cr.L = L
    res = {m: {c: {n: [] for n in NS} for c in CRITS} for m in ("as-run", "fixed", "paper")}
    mismatch = tot = 0
    for im, img in imgs.items():
        h = cr.histogram(img)
        for c in CRITS:
            P = cr.Problem(h, c)
            for n in NS:
                gstar = ref[f"{c}|{im}|{n}"]["g_dp"]
                for s in SEEDS:
                    T, f, _, _ = ISSA_legacy(P, n, seed=s)
                    tot += 1; mismatch += int(abs(P.obj(list(T)) - f) > 1e-9 * abs(f))
                    runs = {"as-run": f, "fixed": issa_fixed(P, n, s)[1], "paper": issa_paper(P, n, s)[1]}
                    for m, fv in runs.items():
                        assert fv - gstar >= -1e-9 * abs(gstar)
                        res[m][c][n].append(abs(fv - gstar) / abs(gstar))
        print(im, "done", flush=True)
    print(f"\nas-run: returned thresholds do not attain the reported fitness in {mismatch} of {tot} runs")
    out = {}
    for m in res:
        allr = []
        for c in CRITS:
            for n in NS:
                allr += res[m][c][n]
        out[m] = dict(SR=100 * float(np.mean(np.array(allr) < TOL)), mean_gap=float(np.mean(allr)), median_gap=float(np.median(allr)),
                      by_criterion={c: 100 * float(np.mean(np.concatenate([res[m][c][n] for n in NS]) < TOL)) for c in CRITS},
                      by_n={n: 100 * float(np.mean(np.concatenate([res[m][c][n] for c in CRITS]) < TOL)) for n in NS})
        print(m, json.dumps(out[m]))
    json.dump(dict(protocol=__doc__, mismatch_returned_vector=mismatch, runs=tot, results=out), open(os.path.join(OUT, "issa_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
