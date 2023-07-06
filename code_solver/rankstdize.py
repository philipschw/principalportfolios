import numpy as np
from scipy.stats import rankdata

def rankstdize(rawvar):
    T, N = rawvar.shape
    rankvar = np.full((T, N), np.nan)

    for t in range(T):
        valid_idx = ~np.isnan(rawvar[t, :])
        n = np.sum(valid_idx)

        rk = rankdata(rawvar[t, valid_idx])
        rankvar[t, valid_idx] = rk / n - np.nanmean(rk / n)

    return rankvar