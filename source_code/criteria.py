"""
Criterion-agnostic multilevel thresholding problems.
Every supported criterion is SEGMENT-ADDITIVE:
    score(T) = sum_k  S(t_{k-1}, t_k)          (to be MAXIMIZED)
and each per-segment score S(a,b) is O(1) via pre-computed cumulative arrays.
We expose obj(T) = -score(T) so all optimizers uniformly MINIMIZE.

Gray levels i = 1..L (L=256); h[i] = #pixels of value (i-1).
Segment [a,b) uses moments over i=a..b-1.
"""
import numpy as np

L = 256


def histogram(img):
    counts = np.bincount(img.ravel(), minlength=L).astype(np.float64)
    h = np.zeros(L + 2)
    h[1:L + 1] = counts
    return h


class Problem:
    """Holds pre-computed cumulatives for one image + criterion."""
    def __init__(self, h, criterion):
        self.criterion = criterion
        self.N = h[1:L + 1].sum()
        idx = np.arange(0, L + 1)
        hh = np.zeros(L + 1); hh[1:L + 1] = h[1:L + 1]
        # moment cumulatives (0..L)
        self.C0 = np.concatenate(([0.0], np.cumsum(hh[1:L + 1])))          # sum h
        self.C1 = np.concatenate(([0.0], np.cumsum(idx[1:] * hh[1:L + 1])))  # sum i*h
        if criterion == 'kapur':
            p = hh[1:L + 1] / self.N
            plnp = np.where(p > 0, p * np.log(p), 0.0)
            self.Cp = np.concatenate(([0.0], np.cumsum(p)))                 # sum p
            self.Cplnp = np.concatenate(([0.0], np.cumsum(plnp)))           # sum p ln p

    # --- O(1) per-segment score S(a,b), to be MAXIMIZED ---
    def seg_S(self, a, b):
        m0 = self.C0[b - 1] - self.C0[a - 1]
        if m0 <= 0:
            return 0.0
        m1 = self.C1[b - 1] - self.C1[a - 1]
        c = self.criterion
        if c == 'otsu':
            return m1 * m1 / m0                      # sum -> proportional to between-class var
        if c == 'mcet':
            return m1 * np.log(m1 / m0) if m1 > 0 else 0.0
        if c == 'kapur':
            w = self.Cp[b - 1] - self.Cp[a - 1]
            if w <= 0:
                return 0.0
            A = self.Cplnp[b - 1] - self.Cplnp[a - 1]
            return -A / w + np.log(w)
        raise ValueError(c)

    def score(self, thresholds):
        bounds = [1] + list(thresholds) + [L + 1]
        return sum(self.seg_S(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1))

    def obj(self, thresholds):
        """Minimization objective (lower is better) = -score."""
        return -self.score(thresholds)
