# import packages
import numpy as np

# import self-written auxiliary functions
from .predeig import predeig

def specptfs(R, S, Roos, Soos):
    """
    Calculate principal portfolios, exposure portfolios, and alpha portfolios based on training and testing data.

    Parameters
    ----------
    R : ndarray
        Training data set R with shape (T, N).
    S : ndarray
        Training data set S with shape (T, N).
    Roos : ndarray
        Testing data set Roos with shape (1, N).
    Soos : ndarray
        Testing data set Soos with shape (1, N).

    Returns
    -------
    Wfull1 : ndarray
        NxN matrix of eigenvectors for Pi'*Pi.
    Wfull2 : ndarray
        NxN matrix of eigenvectors for Pi*Pi'.
    Dfull : ndarray
        NxN diagonal matrix of singular values of Pi (ordered from large to small).
    Wsym : ndarray
        NxN matrix of eigenvectors of symmetric component Pis (ordered from large to small).
    Dsym : ndarray
        Nx1 vector of eigenvalues of symmetric component Pis (ordered from large to small).
    Wasym : ndarray
        N/2xN/2 matrix of eigenvectors of asymmetric component Pia (ordered from large to small).
    Dasym : ndarray
        N/2x1 vector of eigenvalues of asymmetric component Pia (ordered from large to small).
    PP : ndarray
        Principal portfolios based on Pi.
    PEP : ndarray
        Principal exposure portfolios based on Pis.
    PAP : ndarray
        Principal alpha portfolios based on Pia.
    """
    N = R.shape[1]
    
    Wfull1, Wfull2, Dfull, Wsym, Dsym, Wasym, Dasym = predeig(R, S)
    
    Roos[np.isnan(Roos)] = 0
    
    # Pi: Principal portfolios
    PP = np.full_like(Roos, np.nan)
    for k in range(N):
        w1 = Wfull1[:, k]
        w2 = Wfull2[:, k]
        Sw1 = np.dot(Soos, w1)
        Rw2 = np.dot(Roos, w2)
        PP[k] = Sw1 * Rw2
    
    # Pis: Principal exposure portfolios
    PEP = np.full_like(Roos, np.nan)
    for k in range(N):
        ws = Wsym[:, k]
        Sws = np.dot(Soos, ws)
        Rws = np.dot(Roos, ws)
        PEP[k] = Sws * Rws
    
    # Pia: Principal alpha portfolios
    PAP = np.full((1, Roos.shape[0]//2), np.nan)
    for k in range(N // 2):
        wa1 = np.real(Wasym[:, k])
        wa2 = np.imag(Wasym[:, k])
        Swa1 = np.dot(Soos, wa1)
        Swa2 = np.dot(Soos, wa2)
        Rwa1 = np.dot(Roos, wa1)
        Rwa2 = np.dot(Roos, wa2)
        PAP[:, k] = Swa1 * Rwa2 - Swa2 * Rwa1
    
    return Wfull1, Wfull2, Dfull, Wsym, Dsym, Wasym, Dasym, PP, PEP, PAP
