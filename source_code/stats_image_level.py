"""Image-level statistical analysis (Reviewer 1: effect sizes, CIs, dependence; Reviewer 2, comment 6).

PROTOCOL (fixed before looking at any test result)
  Unit of analysis : the IMAGE (N = 8). Seeds, threshold counts and criteria are never treated as independent
                     observations; they are averaged within an image first.
  Primary outcome  : relative objective gap to the DP-certified optimum, |J - J*| / |J*|, MEAN over the ten seeds
                     and over n = 6, 7, 8, computed separately for each criterion. (The mean is used instead of the
                     median because the per-seed median gap of the stronger methods is exactly 0 and would produce
                     degenerate ties.) Gaps are not pooled across criteria because their scales differ.
  Secondary outcome: success rate (gap < 1e-8), per image, over seeds and n; it is on a common scale, so it is
                     also pooled over the three criteria (one value per image).
  Direction        : for a comparison "A vs B", d_i = metric_B - metric_A for the gap and metric_A - metric_B for the
                     success rate, so POSITIVE d favours A (the first-named method).
  Tests            : two-sided Wilcoxon signed-rank test on the 8 paired differences (zero differences dropped,
                     Wilcoxon convention; exact p when there are no ties/zeros). With N = 8 the smallest attainable
                     two-sided p is 2/2^8 = 0.0078.
  Effect sizes     : median of d; Hodges-Lehmann estimate (median Walsh average) with the exact distribution-free
                     confidence interval from the signed-rank distribution (achieved level reported, about 96 %);
                     matched-pairs rank-biserial correlation r = (W+ - W-)/(W+ + W-); number of images favouring A.
  Multiplicity     : Holm correction within named families only:
                       F1 baselines (time-matched, 1x budget): RM-MDE vs PSO, RM-MDE vs DE, per criterion (6 tests)
                       F2 ablation A (MCET): rand/1+LS vs best/1+LS, vs SDE-donor+LS with fixed F = 0.5, vs rand/1 (3 tests)
                       S3 donor sensitivity: vs SDE-donor+LS with randomized F, vs SDE-donor+LS with F = Cr = 0.25 (2 tests)
                       F3 multistart: RM-MDE vs C1, vs C2, per criterion (6 tests)
                       S1 sensitivity: RM-MDE vs PSO at 3x budget, per criterion (3 tests)
                       S2 pooled success rate (over criteria) for the primary comparisons of F1, F3 and F2 (separately
                          Holm-corrected within each of those groups)
  Friedman         : per criterion, methods RM-MDE, PSO, DE (1x), blocks = images; Kendall's W reported. Post hoc =
                     the paired Wilcoxon tests of F1 with Holm.
  CIs for rates    : image-level (cluster) bootstrap, 10000 resamples of the 8 images with a fixed seed; never a
                     binomial CI that treats runs as independent.
  Experiment B     : descriptive only (success profile with image-level bootstrap CI); no hypothesis tests.
Outputs (data_results): stats_image_level.json, stats_image_level_master.csv, stats_image_level_perimage.csv
"""
import os, csv, json, itertools, collections
import numpy as np
import paths
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = paths.DATA
IMAGES = ["Cameraman", "Coins", "Moon", "Astronaut", "Coffee", "Cat", "Clock", "Histology"]
CRITS = ["otsu", "kapur", "mcet"]
NS = (6, 7, 8)
BOOT, BSEED = 10000, 12345


def load(name):
    rows = list(csv.DictReader(open(os.path.join(DATA, name))))
    for r in rows:
        r["n"] = int(r["n"]); r["seed"] = int(r["seed"])
        for k in ("objective_gap", "relative_gap", "success"):
            if k in r:
                r[k] = float(r[k])
    return rows


TM = load("time_matched_DE_PSO.csv")
AB = load("ablation_A_runs.csv") + load("ablation_A2_runs.csv")
MS = load("multistart_C_runs.csv")
BB = load("ablation_B_runs.csv")


def gapof(r):
    return r["objective_gap"] if "objective_gap" in r else r["relative_gap"]


def per_image(rows, crit, pred, field):
    """image -> mean of `field` over seeds and n (rows already restricted by pred)."""
    acc = collections.defaultdict(list)
    for r in rows:
        if (crit is None or r["criterion"] == crit) and pred(r):
            acc[r["image"]].append(gapof(r) if field == "gap" else r["success"])
    for im in IMAGES:
        assert len(acc[im]) == (10 * 3 * (len(CRITS) if crit is None else 1)) or len(acc[im]) > 0, (im, len(acc[im]))
    return np.array([np.mean(acc[im]) for im in IMAGES])


def signed_rank_dist(n):
    cnt = {0: 1}
    for k in range(1, n + 1):
        new = collections.defaultdict(int)
        for s, c in cnt.items():
            new[s] += c; new[s + k] += c
        cnt = new
    return cnt


def hl_ci(d, alpha=0.05):
    n = len(d)
    cnt = signed_rank_dist(n); tot = 2 ** n
    c, cum = -1, 0
    for s in sorted(cnt):
        cum += cnt[s]
        if cum / tot <= alpha / 2:
            c = s
        else:
            break
    walsh = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    M = len(walsh)
    if c < 0:
        return float(walsh[0]), float(walsh[-1]), 1.0 - 2 / tot * 0
    lo, hi = walsh[c], walsh[M - 1 - c]
    level = 1 - 2 * sum(cnt[s] for s in cnt if s <= c) / tot
    return float(lo), float(hi), float(level)


def paired(d):
    d = np.asarray(d, float)
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz)); wp = float(ranks[nz > 0].sum()); wm = float(ranks[nz < 0].sum())
    p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if len(nz) > 0 else 1.0
    lo, hi, lev = hl_ci(d)
    return dict(n_images=len(d), median_diff=float(np.median(d)), HL=float(np.median([(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))])),
                HL_CI_low=lo, HL_CI_high=hi, CI_level=lev, rank_biserial=(wp - wm) / (wp + wm) if wp + wm > 0 else 0.0,
                A_better=int((d > 0).sum()), ties=int((d == 0).sum()), A_worse=int((d < 0).sum()), W_plus=wp, W_minus=wm, p_raw=p)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def boot_ci(v, rng):
    v = np.asarray(v, float)
    s = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(BOOT)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    rng = np.random.default_rng(BSEED)
    out = dict(protocol=__doc__, images=IMAGES, tests=[], friedman=[], rates=[], by_n=[])
    perim_rows = []

    def tm(method, mult):
        return lambda r: r["method"] == method and int(r["budget_mult"]) == mult and r["n"] in NS
    # ---------------- F1 / S1 : baselines ----------------
    metric = {}
    for c in CRITS:
        for name, pred in (("RM-MDE", tm("RM-MDE", 1)), ("PSO", tm("PSO", 1)), ("PSO3x", tm("PSO", 3)), ("DE", tm("DE", 1))):
            for f in ("gap", "succ"):
                metric[("TM", c, name, f)] = per_image(TM, c, pred, f)
                for im, v in zip(IMAGES, metric[("TM", c, name, f)]):
                    perim_rows.append(("time_matched", c, name, f, im, v))
    def comp(key_a, key_b, field):
        a, b = metric[key_a], metric[key_b]
        return (b - a) if field == "gap" else (a - b)
    fam = []
    for c in CRITS:
        for other in ("PSO", "DE"):
            d = comp(("TM", c, "RM-MDE", "gap"), ("TM", c, other, "gap"), "gap")
            fam.append(dict(family="F1", comparison=f"RM-MDE vs {other}", criterion=c, outcome="gap", **paired(d)))
    for t, p in zip(fam, holm([t["p_raw"] for t in fam])): t["p_holm"] = p
    out["tests"] += fam
    fam = []
    for c in CRITS:
        d = comp(("TM", c, "RM-MDE", "gap"), ("TM", c, "PSO3x", "gap"), "gap")
        fam.append(dict(family="S1", comparison="RM-MDE vs PSO (3x budget)", criterion=c, outcome="gap", **paired(d)))
    for t, p in zip(fam, holm([t["p_raw"] for t in fam])): t["p_holm"] = p
    out["tests"] += fam
    # Friedman per criterion
    for c in CRITS:
        cols = [metric[("TM", c, m, "gap")] for m in ("RM-MDE", "PSO", "DE")]
        chi, p = stats.friedmanchisquare(*cols)
        ranks = np.mean([stats.rankdata([cols[j][i] for j in range(3)]) for i in range(len(IMAGES))], axis=0)
        out["friedman"].append(dict(criterion=c, chi2=float(chi), df=2, p=float(p), kendalls_W=float(chi / (len(IMAGES) * 2)),
                                    mean_ranks=dict(zip(("RM-MDE", "PSO", "DE"), ranks.tolist()))))
    # ---------------- F3 : multistart ----------------
    msm = {"RM-MDE": lambda r: r["method"] == "RM-MDE", "C1": lambda r: r["method"] == "MS-LS-4S", "C2": lambda r: r["method"] == "MS-LS-conv"}
    for c in CRITS:
        for name, pred in msm.items():
            for f in ("gap", "succ"):
                metric[("MS", c, name, f)] = per_image(MS, c, pred, f)
                for im, v in zip(IMAGES, metric[("MS", c, name, f)]):
                    perim_rows.append(("multistart", c, name, f, im, v))
    fam = []
    for c in CRITS:
        for other in ("C1", "C2"):
            d = comp(("MS", c, "RM-MDE", "gap"), ("MS", c, other, "gap"), "gap")
            fam.append(dict(family="F3", comparison=f"RM-MDE vs multistart {other}", criterion=c, outcome="gap", **paired(d)))
    for t, p in zip(fam, holm([t["p_raw"] for t in fam])): t["p_holm"] = p
    out["tests"] += fam
    # ---------------- F2 : ablation A (MCET) ----------------
    ab = {v: (lambda v: lambda r: r["variant"] == v)(v) for v in ("rand/1+LS", "best/1+LS", "SDE-donor fixedF+LS", "SDE-donor+LS", "SDE-donor paper(F=Cr=.25)+LS", "rand/1")}
    for name, pred in ab.items():
        for f in ("gap", "succ"):
            metric[("AB", "mcet", name, f)] = per_image([dict(r, criterion="mcet") for r in AB], "mcet", pred, f)
            for im, v in zip(IMAGES, metric[("AB", "mcet", name, f)]):
                perim_rows.append(("ablation_A", "mcet", name, f, im, v))
    fam = []
    for other in ("best/1+LS", "SDE-donor fixedF+LS", "rand/1"):
        d = comp(("AB", "mcet", "rand/1+LS", "gap"), ("AB", "mcet", other, "gap"), "gap")
        fam.append(dict(family="F2", comparison=f"rand/1+LS vs {other}", criterion="mcet", outcome="gap", **paired(d)))
    for t, p in zip(fam, holm([t["p_raw"] for t in fam])): t["p_holm"] = p
    out["tests"] += fam
    fam = []
    for other in ("SDE-donor+LS", "SDE-donor paper(F=Cr=.25)+LS"):
        d = comp(("AB", "mcet", "rand/1+LS", "gap"), ("AB", "mcet", other, "gap"), "gap")
        fam.append(dict(family="S3", comparison=f"rand/1+LS vs {other} (sensitivity)", criterion="mcet", outcome="gap", **paired(d)))
    for t, p in zip(fam, holm([t["p_raw"] for t in fam])): t["p_holm"] = p
    out["tests"] += fam
    # ---------------- S2 : pooled success over criteria ----------------
    def pooled(key, name):
        return np.mean([metric[(key, c, name, "succ")] for c in CRITS], axis=0)
    groups = {"F1": [("RM-MDE", "PSO"), ("RM-MDE", "DE")], "F3": [("RM-MDE", "C1"), ("RM-MDE", "C2")]}
    for g, pairs in groups.items():
        fam = []
        for a, b in pairs:
            key = "TM" if g == "F1" else "MS"
            d = pooled(key, a) - pooled(key, b)
            fam.append(dict(family="S2-" + g, comparison=f"{a} vs {b}" + ("" if g == "F1" else " (multistart)"), criterion="pooled (3 criteria)", outcome="success rate", **paired(d)))
        for t, p in zip(fam, holm([t["p_raw"] for t in fam])): t["p_holm"] = p
        out["tests"] += fam
    # ---------------- image-level bootstrap CIs for success rates ----------------
    def rate(label, v):
        lo, hi = boot_ci(v, rng)
        out["rates"].append(dict(label=label, success_rate=float(100 * np.mean(v)), CI95_low=100 * lo, CI95_high=100 * hi))
    for c in CRITS + ["pooled"]:
        for name in ("RM-MDE", "PSO", "PSO3x", "DE"):
            v = pooled("TM", name) if c == "pooled" else metric[("TM", c, name, "succ")]
            rate(f"time-matched | {c} | {name}", v)
        for name in ("RM-MDE", "C1", "C2"):
            v = pooled("MS", name) if c == "pooled" else metric[("MS", c, name, "succ")]
            rate(f"multistart | {c} | {name}", v)
    for name in ("rand/1", "rand/1+LS", "best/1+LS", "SDE-donor fixedF+LS", "SDE-donor+LS", "SDE-donor paper(F=Cr=.25)+LS"):
        rate(f"ablation A (mcet) | {name}", metric[("AB", "mcet", name, "succ")])
    for p in sorted({float(r["p_ls"]) for r in BB}):
        v = per_image([dict(r, criterion="mcet") for r in BB], "mcet", lambda r, p=p: float(r["p_ls"]) == p, "succ")
        rate(f"ablation B (mcet) | p_ls={p:g}", v)
    # ---------------- by-n sensitivity for the primary comparisons (descriptive) ----------------
    for c in CRITS:
        for n in NS:
            def pm(method, mult=None, src="TM"):
                rows = TM if src == "TM" else MS
                return per_image(rows, c, (lambda r: r["method"] == method and (mult is None or int(r["budget_mult"]) == mult) and r["n"] == n), "gap")
            ref = pm("RM-MDE", 1)
            for lab, other in (("PSO", pm("PSO", 1)), ("DE", pm("DE", 1)),
                               ("multistart C1", pm("MS-LS-4S", None, "MS")), ("multistart C2", pm("MS-LS-conv", None, "MS"))):
                if lab.startswith("multistart"):
                    ref_ms = pm("RM-MDE", None, "MS"); d = other - ref_ms
                else:
                    d = other - ref
                out["by_n"].append(dict(criterion=c, n=n, comparison=f"RM-MDE vs {lab}", RMMDE_better=int((d > 0).sum()),
                                        ties=int((d == 0).sum()), RMMDE_worse=int((d < 0).sum()), p_raw=paired(d)["p_raw"]))
    # ---------------- write ----------------
    json.dump(out, open(os.path.join(DATA, "stats_image_level.json"), "w"), indent=1)
    cols = ["family", "comparison", "criterion", "outcome", "median_diff", "HL", "HL_CI_low", "HL_CI_high", "CI_level", "rank_biserial",
            "A_better", "ties", "A_worse", "p_raw", "p_holm"]
    with open(os.path.join(DATA, "stats_image_level_master.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for t in out["tests"]: w.writerow([t[k] for k in cols])
    with open(os.path.join(DATA, "stats_image_level_perimage.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["experiment", "criterion", "method", "outcome", "image", "value"]); w.writerows(perim_rows)
    # console summary
    print("=== tests ===")
    for t in out["tests"]:
        print(f"{t['family']:6s} {t['comparison'][:34]:34s} {t['criterion'][:10]:10s} {t['outcome'][:7]:7s} HL {t['HL']:+.2e} CI [{t['HL_CI_low']:+.2e},{t['HL_CI_high']:+.2e}] ({100*t['CI_level']:.1f}%) r={t['rank_biserial']:+.2f} "
              f"A>B {t['A_better']}/{t['ties']}/{t['A_worse']} p={t['p_raw']:.4f} holm={t['p_holm']:.4f}")
    print("=== Friedman ===")
    for f in out["friedman"]: print(f)
    print("=== image-level bootstrap CIs (success %) ===")
    for r in out["rates"]: print(f"{r['label']:44s} {r['success_rate']:5.1f}  [{r['CI95_low']:5.1f}, {r['CI95_high']:5.1f}]")


if __name__ == "__main__":
    main()
