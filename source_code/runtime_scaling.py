"""Runtime of the exact DP versus RM-MDE (Experiment R1).

Part A : L=256, n=2..8, three criteria, three images.
Part B : scaling with the number of gray levels L (synthetic histograms), fixed n.
DP is deterministic (timed as the median of DP_REP repeats, table construction included);
RM-MDE is stochastic (R seeds; median/mean/sd/min/max and success vs. the DP optimum).
Both are single-threaded NumPy/Python on the same machine. NOTE the implementation
asymmetry: the DP is vectorized NumPy, RM-MDE (gopt.py) is a pure-Python loop.

Usage: python runtime_scaling.py A|B [outfile.json]
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, time, platform, statistics as st
import numpy as np
import paths
import criteria as cr, gopt

DP_REP = 3


def set_L(L):
    cr.L = L
    gopt.L = L


def prep(h, crit, L):
    """Cumulative arrays indexed by p = 0..L (p = bin boundary b-1)."""
    hh = h[1:L + 1]
    N = hh.sum()
    idx = np.arange(1, L + 1)
    C0 = np.concatenate(([0.0], np.cumsum(hh)))
    C1 = np.concatenate(([0.0], np.cumsum(idx * hh)))
    out = {"C0": C0, "C1": C1}
    if crit == "kapur":
        p = hh / N
        with np.errstate(divide="ignore", invalid="ignore"):
            plnp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
        out["Cp"] = np.concatenate(([0.0], np.cumsum(p)))
        out["Cq"] = np.concatenate(([0.0], np.cumsum(plnp)))
    return out


def score_table(pre, crit, L):
    """W[pa, pb] = S for pa<pb (same conventions/limits as criteria.Problem.seg_S)."""
    C0, C1 = pre["C0"], pre["C1"]
    m0 = C0[None, :] - C0[:, None]
    m1 = C1[None, :] - C1[:, None]
    ok = m0 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        if crit == "otsu":
            W = np.where(ok, m1 * m1 / np.where(ok, m0, 1.0), 0.0)
        elif crit == "mcet":
            ok2 = ok & (m1 > 0)
            W = np.where(ok2, m1 * np.log(np.where(ok2, m1, 1.0) / np.where(ok2, m0, 1.0)), 0.0)
        else:
            w = pre["Cp"][None, :] - pre["Cp"][:, None]
            A = pre["Cq"][None, :] - pre["Cq"][:, None]
            okw = w > 0
            ws = np.where(okw, w, 1.0)
            W = np.where(okw, -A / ws + np.log(ws), 0.0)
    iu = np.triu(np.ones((L + 1, L + 1), bool), 1)
    return np.where(iu, W, -np.inf)


def dp_solve(h, crit, n, L):
    pre = prep(h, crit, L)
    W = score_table(pre, crit, L)
    F = np.full(L + 1, -np.inf)
    F[0] = 0.0
    args = []
    for _ in range(n + 1):
        M = F[:, None] + W
        args.append(M.argmax(axis=0))
        F = M.max(axis=0)
    p = L
    T = []
    for k in range(n, -1, -1):
        p = int(args[k][p])
        T.append(p + 1)
    return F[L], sorted(T[:-1])      # score (to maximize), bin-index thresholds


def time_dp(h, crit, n, L):
    ts = []
    for _ in range(DP_REP):
        t0 = time.perf_counter()
        val, T = dp_solve(h, crit, n, L)
        ts.append(time.perf_counter() - t0)
    return val, T, statistics_median(ts)


def statistics_median(x):
    return float(st.median(x))


def run_mde(prob, n, R):
    ts, vs = [], []
    for s in range(R):
        _, f, t, _ = gopt.MDE(prob, n, seed=s)
        ts.append(t)
        vs.append(f)
    return np.array(ts), np.array(vs)


def stats(ts):
    return dict(median=float(np.median(ts)), mean=float(np.mean(ts)), sd=float(np.std(ts, ddof=1)) if len(ts) > 1 else 0.0,
                min=float(ts.min()), max=float(ts.max()))


def hardware():
    cpu = platform.processor()
    try:
        import subprocess
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip() or cpu
        ram = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        ram = "?"
    return dict(cpu=cpu, ram_gb=ram, os=platform.platform(), python=platform.python_version(), numpy=np.__version__)


def images():
    from skimage import data, color, util
    return {"Cameraman": data.camera(), "Coins": data.coins(),
            "Astronaut": util.img_as_ubyte(color.rgb2gray(data.astronaut()))}


def requantize(img, L, seed=0):
    """Synthetic L-level image from an 8-bit one (adds sub-level noise so all bins are populated)."""
    rng = np.random.default_rng(seed)
    x = img.astype(float) / 255.0 * (L - 1) + rng.uniform(-0.5, 0.5, img.shape) * max(1.0, L / 256.0)
    return np.clip(np.round(x), 0, L - 1).astype(np.int64)


def histL(img, L):
    h = np.zeros(L + 2)
    h[1:L + 1] = np.bincount(img.ravel(), minlength=L)[:L]
    return h


def partA(R=10):
    set_L(256)
    res = dict(hardware=hardware(), R=R, rows=[])
    for iname, img in images().items():
        h = cr.histogram(img)
        for crit in ("otsu", "kapur", "mcet"):
            P = cr.Problem(h, crit)
            for n in range(2, 9):
                val, T, tdp = time_dp(h, crit, n, 256)
                g = -val
                ts, vs = run_mde(P, n, R)
                sr = float(np.mean((vs - g) / abs(g) < 1e-8) * 100)
                row = dict(image=iname, criterion=crit, n=n, dp_time=tdp, dp_T=T, mde=stats(ts), SR=sr,
                           speedup_dp_over_mde_median=tdp / float(np.median(ts)))
                res["rows"].append(row)
                print(iname, crit, n, f"DP {tdp:.3f}s  MDE {np.median(ts):.3f}s  SR {sr:.0f}", flush=True)
    return res


def partB(Ls=(64, 128, 256, 512, 1024, 2048), n=6, R=5):
    res = dict(hardware=hardware(), n=n, R=R, rows=[])
    base = images()["Cameraman"]
    for L in Ls:
        set_L(L)
        img = requantize(base, L)
        h = histL(img, L)
        for crit in ("otsu", "kapur", "mcet"):
            P = cr.Problem(h, crit)
            val, T, tdp = time_dp(h, crit, n, L)
            g = -val
            ts, vs = run_mde(P, n, R if L <= 1024 else max(2, R // 2))
            sr = float(np.mean((vs - g) / abs(g) < 1e-8) * 100)
            res["rows"].append(dict(L=L, criterion=crit, n=n, dp_time=tdp, mde=stats(ts), SR=sr,
                                    speedup_dp_over_mde_median=tdp / float(np.median(ts))))
            print(L, crit, f"DP {tdp:.3f}s  MDE {np.median(ts):.3f}s  SR {sr:.0f}", flush=True)
    return res


def validate():
    """Vectorized DP must reproduce the certified optima (exact_optima_dp.json)."""
    here = os.path.dirname(os.path.abspath(__file__))
    ref = json.load(open(os.path.join(paths.DATA, "exact_optima_dp.json")))
    from skimage import data, color, util
    g = lambda x: util.img_as_ubyte(color.rgb2gray(x))
    imgs = {"Cameraman": data.camera(), "Coins": data.coins(), "Moon": data.moon(), "Astronaut": g(data.astronaut()),
            "Coffee": g(data.coffee()), "Cat": g(data.chelsea()), "Clock": data.clock(),
            "Histology": g(data.immunohistochemistry())}
    set_L(256)
    bad = 0
    for k, v in ref.items():
        crit, im, n = k.split("|")
        val, T = dp_solve(cr.histogram(imgs[im]), crit, int(n), 256)
        if abs(-val - v["g_dp"]) > 1e-9 * abs(v["g_dp"]) or T != v["T"]:
            bad += 1
    print("validation mismatches vs exact_optima_dp.json:", bad, "of", len(ref))
    return bad


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "V":
        validate()
        sys.exit(0)
    out = sys.argv[2] if len(sys.argv) > 2 else f"runtime_{mode}.json"
    res = partA() if mode == "A" else partB()
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)
