# ISSA equation audit checklist (Wu & Yuan 2022 vs `gopt.ISSA`)

Purpose: confirm, against the PDF, that `gopt.ISSA` (corrected one-dimensional adaptation) follows the paper's equations,
so that the manuscript can call it an adaptation that preserves the defining ISSA mechanisms.

**Status of the evidence.** Checked by eye against the published PDF (`References/s11042-022-13073-x.pdf`, Multimedia Tools and Applications 81:33513-33546, pages 5-8: equations (1)-(11), Table 2, Section 5 settings). Verdicts below refer to that PDF.

Notation in the code: rank i = 1..NP after sorting by fitness (code index `i`, equations use `i+1`); `G` = maximum iterations;
`t` = current iteration; `ones` = the L vector of ones.

| # | Item | Paper says (as extracted) | `gopt.ISSA` does | Verdict to confirm |
|---|------|---------------------------|------------------|--------------------|
| 1 | Discoverer branch 1 (R2 < ST) | X exp(-i / (alpha iter_max)) | `X[i] * exp(-(i+1) / (U(0,1] * G))` | MATCH (eq. 1: alpha in [0,1] random, i = rank) |
| 2 | Discoverer branch 2 (R2 >= ST) | X + Q L | `X[i] + N(0,1) * ones` (Q scalar, same shift in every coordinate) | MATCH (eq. 1: Q ~ normal, L = 1 x d ones) |
| 3 | Entrant branch 1 (i > n/2) | Q exp((X_worst - X) / i^2) | `N(0,1) * exp((Xworst - X[i]) / (i+1)^2)` (Q scalar) | MATCH (eq. 2) |
| 4 | Entrant branch 2 (otherwise) | X_P + w abs(X - X_P) A+ L (eq. 4) | `XP + w * (abs(X[i]-XP) @ A/n) * ones` | MATCH (eq. 2 and 4: X_P^{t+1} + w abs(X - X_P^{t+1}) A+ L) |
| 5 | Meaning of X_P | best position occupied by the discoverers (eq. 4 notation uses both X_P and X_p) | best discoverer position AFTER the discoverer update (`argmin(fit[:nprod])`) | MATCH in effect: the text defines X_P as the best position of the current population, written X_P^{t+1} (after the discoverer update, which precedes the entrant update in the flowchart); the best discoverer after its update is that position |
| 6 | Q scalar or vector | normally distributed random number (check whether per-dimension) | scalar | MATCH (Q is a normal random number, L = ones, so Q L is one scalar for all coordinates) |
| 7 | A+ | A is a 1 x d row of +-1 (random sign); A+ = A^T (A A^T)^-1 | `A/n` (since A A^T = n) | MATCH (A is a 1 x d row of +-1, A+ = A^T (A A^T)^-1) |
| 8 | L / Levy term | L is a 1 x d row of ones in eq. 1-4; Levy = (delta_u u) / abs(v)^(1/beta), delta_v = 1 (eq. 6-10) | `ones`; `_levy` implements the Mantegna formula with beta = 1.5 | MATCH for eq. (6)-(9); eq. (10) reads Levy = delta_u u / abs(v)^(1/beta) with u ~ N(0, delta_u^2), which taken literally scales by delta_u twice, whereas the code uses the usual Mantegna step (once). Sensitivity: 720 runs, SR 0.00% (code) vs 0.14% (literal), mean gap 2.65e-3 vs 2.31e-3: immaterial |
| 9 | Vigilant branch 1 (f > f_best) | X_best + Levy abs(X - X_best) (eq. 11, Levy replaces beta) | `Xbest + levy * abs(X[i]-Xbest)` (vector Levy) | MATCH (eq. 3 / 11; Levy replaces beta) |
| 10 | Vigilant branch 2 (f = f_best) | X + K ( abs(X - X_worst) / ((f - f_w) + eps) ), K in [-1, 1] | `X[i] + U(-1,1) * abs(X[i]-Xworst) / (fit[i]-fworst + 1e-12)` | MATCH (eq. 3 / 11: K in [-1, 1], eps small). Note the paper applies this branch when f_i = f_g; the code applies it to f_i <= f_best, which only differs for a sparrow that has just improved on the old best |
| 11 | Cosine inertia w | w = w_max - (w_max - w_min) cos(pi T / G) (eq. 5) | same, w_max = 1.5, w_min = 0.5 (bounds from the author, unverified) | MATCH (eq. 5; Table 2: w in [0.5, 1.5]) |
| 12 | Levy beta | beta = 1.5 | 1.5 | MATCH (beta usually 1.5) |
| 13 | Greedy acceptance | update only if the fitness is better, otherwise give up | `accept`: replace position and fitness only if `f < fit[i]`; otherwise both unchanged | MATCH (Step 6: update only if better, otherwise give up) |
| 14 | Global best update | best position and fitness refreshed consistently | population is re-sorted every iteration; `Xbest` = row 0; the returned vector is verified to attain the stored fitness (raises otherwise) | MATCH (population re-sorted each iteration; flowchart: recalculate fitness and sort) |

## Adaptations that are ours (not in the paper) and must be labelled as such
- one-dimensional ordered-threshold vector in bin units instead of the 2-D threshold pair (t, s) of the 2-D entropy problem;
- positions evaluated through the common repair (round, clip to [2, L], sort, separate coincident thresholds) and stored as the repaired vector;
- common objectives (Otsu, Kapur, minimum cross entropy) and the common DP-certified reference;
- vigilant sparrows chosen as `SD * NP` distinct random individuals (the paper's selection details were not checked);
- common budget NP = 20, G = 60 for the controlled comparison; NP = 20, G = 30 as a sensitivity setting
  (as reported by the author for the application in the paper, to be confirmed).

## Items that are NOT claimed
- exact reproduction of the paper's experiments;
- that the published ISSA would behave identically on the original 2-D problem.

## Outcome of the audit run (`issa_audit.py`, `issa_rerun.py`)
- legacy code: returned thresholds did not attain the reported fitness in 720 / 720 runs (acceptance defect);
- corrected code: invariant enforced; 120-instance study: mean success 0.0% -> 3.5%, mean gap 7.6e-3 -> 2.0e-3, Friedman rank 5.00 -> 4.56,
  RM-MDE still better in 120 / 120 instances.

## Settings confirmed from the PDF
- Table 2 (ISSA): ST = 0.6, PD = 0.3, PV = 0.2, w in [0.5, 1.5].
- Benchmark functions: population 50, iterations 300, 30 independent runs (section 5.1).
- Image segmentation application: maximum iterations 30, population 20 (section 5.2). ISSA's parameters are not restated in the
  application's Table 6.
- Problem solved: 2-D maximum entropy with a threshold pair (t, s); hence the code is a one-dimensional adaptation, not a reproduction.

## Other findings from the References folder (checked against the PDFs)
- Synergetic DE (Ali et al., 2014): oppositional initialization (NP best of 2NP), mutation V = X_tb + F (X_r2 - X_r3) with the best
  of three random individuals as base, dynamic single-population update; parameters in the paper: NP = 10 D, **F = Cr = 0.25**, 200 iterations.
  Our SDE uses F = 0.5, Cr = 0.9 and a **randomized scale factor F (0.5 + 0.5 u), which is not in the paper**; the ablation "synergetic donor"
  inherits that randomization. This must be stated as a deviation (or SDE rerun with F = Cr = 0.25 as a sensitivity check).
- MCET formulation (Horng, 2010, eq. 5-9, citing Li-Lee and Yin): histogram h(i), i = 1..L, t_0 = 1, t_{c+1} = L + 1,
  m0(a, b) and m1(a, b) summed over i = a..b-1 with intensity i. This supports the x = i convention used here (as a secondary source;
  the Li-Lee and Yin papers themselves are not in the folder).
- Bibliographic details confirmed from the PDFs: Xue and Shen 2020 (SSCE 8:1, 22-34), Ali et al. 2014 (ASC 17, 1-11), Akay 2013
  (ASC 13, 3066-3091), Horng 2010 (ESWA 37, 4580-4592), Mirjalili and Lewis 2016 (AES 95, 51-67), Storn and Price 1997 (JOGO 11, 341-359),
  Otsu 1979 (IEEE SMC 9, 62-66), Wu and Yuan 2022 (MTAP 81, 33513-33546; DOI 10.1007/s11042-022-13073-x), Dong et al. 2025 (Appl. Sci. 15).
- Not in the folder (not verifiable here): Yin 2007, Li and Lee 1993, Kapur et al. 1985, Kennedy and Eberhart 1995, Sezgin and Sankur 2004,
  Liao et al. 2001, Luessi et al. 2009. Moscato's PDF carries the program number 158-79; the report number 826 in the manuscript is not
  visible in it.
