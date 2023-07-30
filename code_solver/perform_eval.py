# import packages
import numpy as np
import statsmodels.api as sm

def perform_eval(testassets, benchmark, yearfrac) -> tuple:
    """
    Calculates various performance metrics for a set of test assets.

    :param testassets: A numpy.ndarray matrix containing the test assets. The matrix has shape (M, N), where M is the number of periods and N is the number of assets.
    :param benchmark: A list of numpy.ndarray representing the benchmark.
    :param yearfrac: A value indicating the fraction of a year.
    :return: Four floats (SR, SRse, IR, IRse) containing the calculated performance metrics for each asset.
    :rtype: tuple

    **Notes**:
    - `SR` stands for Sharpe Ratio.
    - `SRse` stands for Standard Error of Sharpe Ratio.
    - `IR` stands for Information Ratio.
    - `IRse` stands for Standard Error of Information Ratio.
    """
    
    N = testassets.shape[1] if len(testassets.shape) > 1 else 1
    SR = np.full((N, 1), np.nan)
    SRse = np.full((N, 1), np.nan)
    IR = np.full((N, 1), np.nan)
    IRse = np.full((N, 1), np.nan)

    for j in range(N):
        ptf = testassets[:, j] if len(testassets.shape) > 1 else testassets
        TT = np.sum(~np.isnan(ptf))

        # ER, SR
        SR[j] = sharpe(ptf) * np.sqrt(1 / yearfrac)
        SRse[j] = np.sqrt(1 / yearfrac) * np.sqrt((1 + 0.5 * np.power(SR[j] / np.sqrt(1 / yearfrac), 2)) / TT)

        # Alpha wrt own-factor and FF5
        if len(benchmark) != 0:
            # Combine ptf and benchmark into a single matrix
            data = np.hstack((ptf.reshape(-1,1), benchmark))

            # Drop rows with NaN values
            data_clean = data[~np.isnan(data).any(axis=1)]

            # Separate the ptf and benchmark columns
            ptf = data_clean[:, 0]
            benchmark_clean = data_clean[:, 1:]

            # Perform linear regression
            model = sm.OLS(ptf, sm.add_constant(benchmark_clean))
            results = model.fit()

            # Extract the alpha and correlation coefficient
            alpha = results.params[0]
            r = results.resid

            IR[j] = sharpe(alpha + r) * np.sqrt(1 / yearfrac)
            IRse[j] = np.sqrt(1 / yearfrac) * np.sqrt((1 + 0.5 * np.power(IR[j] / np.sqrt(1 / yearfrac), 2)) / TT)

    return SR, SRse, IR, IRse

def sharpe(returns):
    """
    Calculates the Sharpe Ratio for a given set of returns.

    :param returns: A numpy.ndarray or a sequence of returns.
    :type returns: numpy.ndarray
    :return: The Sharpe Ratio as a float value.
    :rtype: float

    The Sharpe Ratio measures the risk-adjusted return of an investment or trading strategy. It is calculated as the ratio of the average return to the standard deviation of the returns.

    **Note**:
    - `returns` should not contain NaN (Not a Number) values, or they will be treated as missing values.
    - The Sharpe Ratio is calculated using the formula: average return / standard deviation of returns (with degrees of freedom = 1).
    """

    return np.nanmean(returns) / np.nanstd(returns, ddof=1)