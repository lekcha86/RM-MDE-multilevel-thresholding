"""MCET offset validation (Experiment R3).

PROTOCOL (fixed before running)
  Question   : does the intensity convention in the minimum-cross-entropy score change the optimum?
  Formulation A (manuscript so far): x_i = i      (bin index, i = 1..256; gray value + 1)
  Formulation B (physical gray)    : x_i = i - 1  (gray value 0..255)
  Score      : S(a,b) = m1 * ln(m1 / m0), m_k computed with x_i; segment [a,b) in bin indices.
  Convention : S = 0 when m0 <= 0 (empty class) or m1 <= 0 (a class containing only intensity 0 under B),
               i.e. the limit x ln x -> 0; the SAME rule in the DP and in RM-MDE (criteria.Problem.seg_S).
               The script counts how often m1 = 0 with m0 > 0 occurs (table cells and optimal segments).
  Instances  : 8 images x n = 2..8 = 56 per formulation. Exact DP gives the certified optimum of each.
  Comparison : thresholds are bin indices in both formulations, so vectors are compared directly
               (t_gray = t_bin - 1 only for display). Reported: objective values, relative difference,
               number of changed optima, the loss of using A's optimum under B and vice versa
               (cross-evaluation), ties (a different vector with equal objective within 1e-9).
  RM-MDE     : NP=20, G=60, p_ls=.15, 10 seeds (0..9), both formulations, success vs. that formulation's
               own DP optimum, tolerance 1e-8.
Sanity: formulation A must reproduce exact_optima_dp.json (n=4..8) exactly.
Outputs (data_results): exact_optima_mcet_gray0.json, mcet_offset_validation.json/.csv,
                           mcet_offset_rmmde_comparison.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json, csv
import numpy as np
import paths
import criteria as cr, gopt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = paths.DATA
L = 256
NS = range(2, 9)
SEEDS = list(range(10))
TOL = 1e-8


def all_images():
    from skimage import data, color, util
    g = lambda x: util.img_as_ubyte(color.rgb2gray(x))
    return {"Cameraman": data.camera(), "Coins": data.coins(), "Moon": data.moon(), "Astronaut": g(data.astronaut()),
            "Coffee": g(data.coffee()), "Cat": g(data.chelsea()), "Clock": data.clock(),
            "Histology": g(data.immunohistochemistry())}


def cumulatives(h, shift):
    """C0, C1 indexed by p = 0..L (p = bin boundary b-1). x_i = i - shift."""
    hh = h[1:L + 1]
    x = np.arange(1, L + 1) - shift
    return np.concatenate(([0.0], np.cumsum(hh))), np.concatenate(([0.0], np.cumsum(x * hh)))


def table(C0, C1):
    m0 = C0[None, :] - C0[:, None]
    m1 = C1[None, :] - C1[:, None]
    iu = np.triu(np.ones((L + 1, L + 1), bool), 1)
    ok = (m0 > 0) & (m1 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(ok, m1 * np.log(np.where(ok, m1, 1.0) / np.where(ok, m0, 1.0)), 0.0)
    zero_m1 = int(((m0 > 0) & (m1 == 0) & iu).sum())          # segments with mass but m1 == 0
    neg_m1 = int(((m0 > 0) & (m1 < 0) & iu).sum())
    return np.where(iu, W, -np.inf), zero_m1, neg_m1


def dp(W, n):
    F = np.full(L + 1, -np.inf); F[0] = 0.0; args = []
    for _ in range(n + 1):
        M = F[:, None] + W
        args.append(M.argmax(axis=0)); F = M.max(axis=0)
    p, T = L, []
    for k in range(n, -1, -1):
        p = int(args[k][p]); T.append(p + 1)
    return float(F[L]), sorted(T[:-1])        # score (maximize); thresholds as bin indices


def score_of(W, T):
    p = [0] + [t - 1 for t in T] + [L]
    return float(sum(W[p[k], p[k + 1]] for k in range(len(p) - 1)))


class ProblemGray0(cr.Problem):
    """MCET with x_i = i - 1 (physical gray values); same seg_S rule (S = 0 if m0 <= 0 or m1 <= 0)."""
    def __init__(self, h):
        super().__init__(h, "mcet")
        hh = np.zeros(L + 1); hh[1:L + 1] = h[1:L + 1]
        self.C1 = np.concatenate(([0.0], np.cumsum((np.arange(1, L + 1) - 1) * hh[1:L + 1])))


def seg_counts(C0, C1, T):
    p = [0] + [t - 1 for t in T] + [L]
    m0 = [C0[p[k + 1]] - C0[p[k]] for k in range(len(p) - 1)]
    m1 = [C1[p[k + 1]] - C1[p[k]] for k in range(len(p) - 1)]
    return sum(1 for a, b in zip(m0, m1) if a > 0 and b == 0), sum(1 for a in m0 if a <= 0)


def main():
    imgs = all_images()
    ref = json.load(open(os.path.join(OUT, "exact_optima_dp.json")))
    rows, optima_gray0 = [], {}
    tot_zero_cells = {}
    for im, img in imgs.items():
        h = cr.histogram(img)
        CA = cumulatives(h, 0)        # x = i
        CB = cumulatives(h, 1)        # x = i - 1
        WA, zA, nA = table(*CA); WB, zB, nB = table(*CB)
        tot_zero_cells[im] = dict(A_cells_m1_zero=zA, B_cells_m1_zero=zB, A_cells_m1_neg=nA, B_cells_m1_neg=nB)
        for n in NS:
            JA, TA = dp(WA, n); JB, TB = dp(WB, n)
            if n >= 4:                                    # sanity vs the certified file (formulation A)
                r = ref[f"mcet|{im}|{n}"]
                assert abs(-JA - r["g_dp"]) <= 1e-9 * abs(r["g_dp"]) and TA == r["T"], (im, n)
            JB_at_TA = score_of(WB, TA); JA_at_TB = score_of(WA, TB)
            changed = TA != TB
            tie = changed and abs(JB_at_TA - JB) <= 1e-9 * abs(JB)
            zsegA = seg_counts(*CA, TA); zsegB = seg_counts(*CB, TB)
            rows.append(dict(image=im, n=n, J_A=JA, J_B=JB, rel_diff_J=(JB - JA) / abs(JA),
                             T_A_bin=TA, T_B_bin=TB, T_A_gray=[t - 1 for t in TA], T_B_gray=[t - 1 for t in TB],
                             optimum_changed=int(changed), tie_under_B=int(tie),
                             loss_using_A_vectors_under_B=(JB - JB_at_TA) / abs(JB),
                             loss_using_B_vectors_under_A=(JA - JA_at_TB) / abs(JA),
                             B_opt_segments_m1_zero=zsegB[0], B_opt_segments_empty=zsegB[1],
                             A_opt_segments_m1_zero=zsegA[0], A_opt_segments_empty=zsegA[1]))
            optima_gray0[f"mcet_gray0|{im}|{n}"] = dict(g_dp=-JB, T=TB, T_gray=[t - 1 for t in TB])
    changed_n = sum(r["optimum_changed"] for r in rows)
    summary = dict(instances=len(rows), changed_optima=changed_n, ties_among_changed=sum(r["tie_under_B"] for r in rows),
                   max_abs_rel_objective_diff=max(abs(r["rel_diff_J"]) for r in rows),
                   max_loss_A_under_B=max(r["loss_using_A_vectors_under_B"] for r in rows),
                   max_loss_B_under_A=max(r["loss_using_B_vectors_under_A"] for r in rows),
                   instances_with_zero_m1_optimal_segment_B=sum(1 for r in rows if r["B_opt_segments_m1_zero"] > 0),
                   instances_with_empty_class_at_optimum=sum(1 for r in rows if r["B_opt_segments_empty"] > 0 or r["A_opt_segments_empty"] > 0),
                   zero_m1_table_cells=tot_zero_cells,
                   changed_list=[(r["image"], r["n"]) for r in rows if r["optimum_changed"]])
    print(json.dumps({k: v for k, v in summary.items() if k != "zero_m1_table_cells"}, indent=1))
    json.dump(optima_gray0, open(os.path.join(OUT, "exact_optima_mcet_gray0.json"), "w"))
    json.dump(dict(summary=summary, instances=rows), open(os.path.join(OUT, "mcet_offset_validation.json"), "w"), indent=1)
    with open(os.path.join(OUT, "mcet_offset_validation.csv"), "w", newline="") as fh:
        w = csv.writer(fh); keys = list(rows[0])
        w.writerow(keys); [w.writerow([json.dumps(r[k]) if isinstance(r[k], list) else r[k] for k in keys]) for r in rows]

    # ---- RM-MDE under both formulations, each against its own DP optimum ----
    cmp_rows = []
    for im, img in imgs.items():
        h = cr.histogram(img)
        PA, PB = cr.Problem(h, "mcet"), ProblemGray0(h)
        for n in NS:
            rr = [r for r in rows if r["image"] == im and r["n"] == n][0]
            for name, P, J in (("A", PA, rr["J_A"]), ("B", PB, rr["J_B"])):
                fs = np.array([gopt.MDE(P, n, seed=s)[1] for s in SEEDS])
                gaps = (fs - (-J)) / abs(J)
                assert (gaps >= -1e-9).all()
                cmp_rows.append(dict(image=im, n=n, formulation=name, SR=float(np.mean(gaps < TOL) * 100),
                                     mean_gap=float(gaps.mean())))
        print(im, "done", flush=True)
    agg = {}
    for name in "AB":
        r = [x for x in cmp_rows if x["formulation"] == name]
        agg[name] = dict(SR_all=float(np.mean([x["SR"] for x in r])),
                         SR_n6_8=float(np.mean([x["SR"] for x in r if x["n"] >= 6])),
                         mean_gap=float(np.mean([x["mean_gap"] for x in r])))
    print(agg)
    json.dump(dict(seeds=SEEDS, aggregate=agg, rows=cmp_rows), open(os.path.join(OUT, "mcet_offset_rmmde_comparison.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
