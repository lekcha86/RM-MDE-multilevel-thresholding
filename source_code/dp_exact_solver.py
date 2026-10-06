import numpy as np
from criteria import L, Problem

def score_matrix(P):
    """W[a,b] = S(a,b) for 1<=a<b<=L+1 (boundary indices), -inf otherwise. Uses P.seg_S exactly."""
    W = np.full((L + 2, L + 2), -np.inf)
    for a in range(1, L + 1):
        for b in range(a + 1, L + 2):
            W[a, b] = P.seg_S(a, b)
    return W

def dp_optimum(W, n):
    """Exact global maximum of sum S over n ordered thresholds t_1<..<t_n in [2,L]
    (n+1 segments between boundaries 1 and L+1). Returns (thresholds, score)."""
    K = n + 1
    F = np.full((K + 1, L + 2), -np.inf); arg = np.zeros((K + 1, L + 2), dtype=int)
    F[0, 1] = 0.0
    for k in range(1, K + 1):
        M = F[k - 1][:, None] + W            # M[a,b]
        arg[k] = M.argmax(axis=0)
        F[k] = M.max(axis=0)
    b = L + 1; bounds = [b]
    for k in range(K, 0, -1):
        b = arg[k, b]; bounds.append(b)
    bounds = bounds[::-1]                      # 1 ... L+1
    return np.array(bounds[1:-1]), F[K, L + 1]
