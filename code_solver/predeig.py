# import packages
import numpy as np

def predeig(R, S) -> tuple:
    """
   Perform eigenvalue decomposition

   :param R: Training data set R with shape (T, N).
   :type R: ndarray
   :param S: Training data set S with shape (T, N).
   :type S: ndarray
   :return: Tuple containing the following arrays:
      - W1: NxN matrix of eigenvectors for Pi'*Pi.
      - W2: NxN matrix of eigenvectors for Pi*Pi'.
      - D: NxN diagonal matrix of singular values of Pi (ordered from large to small).
      - Wsym: NxN matrix of eigenvectors of symmetric component Pis (ordered from large to small).
      - Dsym: Nx1 vector of eigenvalues of symmetric component Pis (ordered from large to small).
      - Wasym: N/2xN/2 matrix of eigenvectors of asymmetric component Pia (ordered from large to small).
      - Dasym: N/2x1 vector of eigenvalues of asymmetric component Pia (ordered from large to small).
   :rtype: tuple

    """

    # SigmaRS and its sym/asym parts
    T, N = R.shape
    Pi = np.dot(R.T, S / T)
    Pis = 0.5 * (Pi + Pi.T)
    Pia = 0.5 * (Pi - Pi.T) * 1j

    # Total Pi
    D1, W1 = np.linalg.eig(np.dot(Pi.T, Pi))
    D1 = np.flipud(np.sqrt(np.abs(D1)))
    W1 = np.fliplr(W1)
    W2 = np.dot(Pi, W1) @ np.linalg.inv(np.diag(D1))
    D = np.flipud(D1)
    W1 = np.fliplr(W1)
    W2 = np.fliplr(W2)

    # Symmetric component
    Dsym, Wsym = np.linalg.eig(Pis)

    # Asymmetric component
    Dasym, Wasym = np.linalg.eig(Pia)
    ix = np.argsort(Dasym)[::-1]
    ix = ix[:N//2]
    Dasym = Dasym[ix]
    Wasym = Wasym[:, ix]

    return W1, W2, D, Wsym, Dsym, Wasym, Dasym