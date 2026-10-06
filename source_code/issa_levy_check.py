"""Sensitivity of the corrected ISSA to the reading of equation (10) of Wu and Yuan (2022).

Standard Mantegna step (used in gopt.ISSA): u ~ N(0, delta_u^2), v ~ N(0, 1), Levy = u / |v|^(1/beta).
Literal reading of eq. (10): Levy = delta_u * u / |v|^(1/beta) with u ~ N(0, delta_u^2), i.e. delta_u applied twice.
720 runs per reading (8 images x 3 criteria x n = 6, 7, 8 x seeds 0..9), NP = 20, G = 60, DP reference.
Output: data_results/issa_levy_check.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, math
import numpy as np
import paths
import criteria as cr, gopt
from ablation import all_images

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
cr.L = 256
ref = json.load(open(os.path.join(DATA, "exact_optima_dp.json")))
std = gopt._levy


def literal(rng, n, beta=1.5):
    sg = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sg, n); v = rng.normal(0, 1, n)
    return sg * u / np.abs(v) ** (1 / beta)


imgs = all_images(); out = {}
for name, fn in (("standard (delta_u once)", std), ("literal eq. (10) (delta_u twice)", literal)):
    gopt._levy = fn; gaps = []
    for im, img in imgs.items():
        h = cr.histogram(img)
        for c in ("otsu", "kapur", "mcet"):
            P = cr.Problem(h, c)
            for n in (6, 7, 8):
                g = ref[f"{c}|{im}|{n}"]["g_dp"]
                for s in range(10):
                    gaps.append(abs(gopt.ISSA(P, n, seed=s)[1] - g) / abs(g))
    gaps = np.array(gaps)
    out[name] = dict(runs=len(gaps), SR=100 * float(np.mean(gaps < 1e-8)), gap_mean=float(gaps.mean()), gap_median=float(np.median(gaps)))
    print(name, out[name], flush=True)
gopt._levy = std
json.dump(out, open(os.path.join(DATA, "issa_levy_check.json"), "w"), indent=1)
