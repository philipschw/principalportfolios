import numpy as np

def predeig(R, S):
    """
    R,S       : TxN training data sets
    D         : Nx1 vector of singular values of Pi (ordered from large to small)
    W1        : NxN matrix of eigvecs for Pi'*Pi
    W2        : NxN matrix of eigvecs for Pi*Pi'
    Dsym      : Nx1 vector of eigvals of Pis (ordered from large to small)
    Wsym      : NxN matrix of corresponding eigvecs
    Dasym     : N/2x1 vector of eigvals of Pia (ordered from large to small)
    Wasym     : N/2xN/2 matrix of corresponding eigvecs
    """
    T, N = R.shape

    Pi = np.dot(R.T, S) / T
    Pis = 0.5 * (Pi + Pi.T)
    Pia = 0.5 * (Pi - Pi.T) * 1j

    W1, D1 = np.linalg.eig(np.dot(Pi.T, Pi))
    D1 = np.sqrt(np.abs(D1))
    W2 = np.dot(Pi, np.dot(W1, np.linalg.inv(D1)))
    D = np.diag(D1)
    D = np.flipud(D)
    W1 = np.fliplr(W1)
    W2 = np.fliplr(W2)

    Wsym, Dsym = np.linalg.eig(Pis)
    idx = np.argsort(Dsym)[::-1]
    Dsym = Dsym[idx]
    Wsym = Wsym[:, idx]

    Wasym, Dasym = np.linalg.eig(Pia)
    idx = np.argsort(Dasym)[::-1]
    idx = idx[:N // 2]
    Dasym = Dasym[idx]
    Wasym = Wasym[:, idx]

    return W1, W2, D, Wsym, Dsym, Wasym, Dasym