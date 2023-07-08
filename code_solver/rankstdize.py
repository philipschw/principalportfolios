# import packages
import numpy as np
from scipy.stats import rankdata

def rankstdize(rawvar: np.ndarray) -> np.ndarray:
    """
    Perform rank standardization on the input array.

    :param rawvar: Input array of shape (T, N) with N bigger or equal 1.
    :type rawvar: np.ndarray
    :return: Rank standardized array of the same shape as rawvar.
    :rtype: np.ndarray

    :Description:
        The rankstdize function takes an input array `rawvar` and performs rank standardization on each row.
        The ranks are mapped to the range [-0.5, 0.5] and centered around zero by subtracting the mean.
        The resulting array has the same shape as the input array, with NaN values preserved.

    :Example:
        >>> import numpy as np
        >>> rawvar = np.array([[3, 2, 5],
        ...                    [1, 4, np.nan]])
        >>> rankvar = rankstdize(rawvar)
        >>> print(rankvar)
        [[ 0.        , -0.33333333,  0.33333333],
         [-0.25      ,  0.25      ,      np.nan]]
    """
    T, N = rawvar.shape if rawvar.ndim > 1 else (rawvar.shape[0],1)
    rankvar = np.full((T, N), np.nan)

    for t in range(T):
        # Get the non-NaN values in the current row
        non_nan_index = ~np.isnan(rawvar[t, :])

        # Compute the number of non-NaN values
        n = np.sum(non_nan_index)

        # Map to [-0.5,0.5]
        rk = np.argsort(np.argsort(rawvar[t, non_nan_index])) + 1
        rankvar[t, non_nan_index] = rk / n - np.nanmean(rk / n)

        # Enforce np.dot(rankvar[t,:].T, rankvar[t,:]) == 1
        # rankvar[t, :] /= np.sqrt(np.nansum(rankvar[t, :]**2))

    return rankvar