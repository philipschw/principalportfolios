# import packages
import numpy as np
from scipy.linalg import lstsq

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
        if len(benchmark) != 0:
            # Combine X and y into a single matrix
            Xy = np.column_stack((benchmark, np.atleast_2d(ptf).T))
            
            # Remove rows with NaN values
            Xy = Xy[~np.isnan(Xy).any(axis=1)]

            # Separate X and y after removing NaN values
            X = Xy[:, :-1]
            y = Xy[:, -1]

            # Add a column of ones to X for the intercept term
            X_with_intercept = np.column_stack((np.ones((X.shape[0], 1)), X))

            # Perform least squares regression
            coefficients, _, _, _ = lstsq(X_with_intercept, y, lapack_driver='gelsd')

            y_predicted = np.dot(X_with_intercept, coefficients)

            # Calculate residuals
            residuals = y - y_predicted

            # Calculate information ratio
            alpha = coefficients[0] + residuals
            IR[j] = sharpe(alpha) * np.sqrt(1 / yearfrac)
            IRse[j] = np.sqrt(1 / yearfrac) * np.sqrt((1 + 0.5 * (IR[j] / np.sqrt(1 / yearfrac)) ** 2) / TT)

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