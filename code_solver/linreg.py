# import packages
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import nnls

def linreg(y: np.ndarray, X: np.ndarray, intcpt: bool = True, nnconstraint: bool = False) -> tuple:
    """
    Perform linear regression and calculate statistics.

    NaN values are automatically handled by removing all corresponding rows in X and y if any NaN value is present.

    :param y:  Dependent variable of shape (n,).
    :type y: numpy.ndarray
    :param X: Independent variables of shape (n, p).
    :type X:  numpy.ndarray
    :param intcpt: Whether to include the intercept term, defaults to True
    :type intcpt: bool, optional
    :param nnconstraint: Whether to impose a non-negativity constraint on the estimated coefficeints, defaults to False
    :type nnconstraint: bool, optional
    :return: A tuple consiting of the estimated coefficients of shape (p+1,), t-statistics of the coefficients of shape (p+1,), and the R-squared value of the regression model.
    :rtype: tuple
    """
    if not isinstance(y, np.ndarray):
        raise TypeError('y must be a numpy.ndarray')
    if y.ndim != 1:
        raise ValueError('y must be a 1-D array')
    if not isinstance(X, np.ndarray):
        raise TypeError('X must be a numpy.ndarray')
    if X.ndim != 2:
        raise ValueError('X must be a 2-D array')
    if y.shape[0] != X.shape[0]:
        raise ValueError('The number of rows in y and X must be the same')
    if X.shape[0] < X.shape[1]:
        raise ValueError('The number of rows in X must be greater than or equal to the number of columns')
    if not isinstance(intcpt, bool):
        raise TypeError('intcpt must be a boolean')
    if not isinstance(nnconstraint, bool):
        raise TypeError('nnconstraint must be a boolean')

    # Remove rows with NaN values
    mask = np.isnan(y) | np.isnan(X).any(axis=1)
    y = y[~mask]
    X = X[~mask]

    if intcpt == True:
        # Add constant term to X (if needed)
        X = np.column_stack((np.ones(len(X)), X))

    # Perform linear regression
    if nnconstraint:
        coefficients, _ = nnls(X, y)
    else:
        coefficients, _, _, _ = lstsq(X, y, lapack_driver='gelsd')

    # Calculate R-squared
    ymean = np.mean(y)
    ypred = np.dot(X, coefficients)
    res = y - ypred
    sstot = np.sum((y - ymean) ** 2)
    ssres = np.sum(res ** 2)
    rsquared = 1 - (ssres / sstot)

    # Calculate t-statistics
    n = len(y)
    p = X.shape[1] - 1
    mse = ssres / (n - p - 1)
    covmat = np.linalg.inv(X.T @ X) * mse
    stderr = np.sqrt(np.diag(covmat))
    tstat = coefficients / stderr

    return coefficients, tstat, rsquared
