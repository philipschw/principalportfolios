import numpy as np
from scipy.stats import linregress

def perform_eval(testassets, benchmark, yearfrac):
    N = testassets.shape[1]
    SR = np.full((N, 1), np.nan)
    SRse = np.full((N, 1), np.nan)
    IR = np.full((N, 1), np.nan)
    IRse = np.full((N, 1), np.nan)

    for j in range(N):
        ptf = testassets[:, j]
        TT = np.sum(~np.isnan(ptf))

        # ER, SR
        SR[j] = sharpe(ptf) * np.sqrt(1 / yearfrac)
        SRse[j] = np.sqrt(1 / yearfrac) * np.sqrt((1 + 0.5 * (SR[j] / np.sqrt(1 / yearfrac)) ** 2) / TT)

        # Alpha wrt own-factor and FF5
        if benchmark is not None:
            stats = linregress(benchmark, ptf)
            alpha = stats.intercept + stats.slope * benchmark
            IR[j] = sharpe(alpha) * np.sqrt(1 / yearfrac)
            IRse[j] = np.sqrt(1 / yearfrac) * np.sqrt((1 + 0.5 * (IR[j] / np.sqrt(1 / yearfrac)) ** 2) / TT)

    return SR, SRse, IR, IRse

def sharpe(returns):
    return np.mean(returns) / np.std(returns, ddof=1)