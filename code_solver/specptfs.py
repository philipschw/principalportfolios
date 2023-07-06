import numpy as np

def specptfs(R, S, Roos, Soos):
    """
    R,S           : TxN training data sets
    Roos,Soos     : 1xN testing data
    PP            : Principal portfolios based Pi
    PEP           : Principal exposure portfolios based Pis
    PAP           : Principal alpha portfolios based Pia
    Other outputs passed from predeig
    """
    N = R.shape[1]
    
    Wfull1, Wfull2, Dfull, Wsym, Dsym, Wasym, Dasym = predeig(R, S)
    
    Roos[np.isnan(Roos)] = 0
    
    # Pi: Principal portfolios
    PP = np.zeros_like(Roos)
    for k in range(N):
        w1 = Wfull1[:, k]
        w2 = Wfull2[:, k]
        Sw1 = np.dot(Soos, w1)
        Rw2 = np.dot(Roos, w2)
        PP[:, k] = Sw1 * Rw2
    
    # Pis: Principal exposure portfolios
    PEP = np.zeros_like(Roos)
    for k in range(N):
        ws = Wsym[:, k]
        Sws = np.dot(Soos, ws)
        Rws = np.dot(Roos, ws)
        PEP[:, k] = Sws * Rws
    
    # Pia: Principal alpha portfolios
    PAP = np.zeros((1, int(Roos.shape[1] / 2)))
    for k in range(0, N, 2):
        wa1 = np.real(Wasym[:, k])
        wa2 = np.imag(Wasym[:, k])
        Swa1 = np.dot(Soos, wa1)
        Swa2 = np.dot(Soos, wa2)
        Rwa1 = np.dot(Roos, wa1)
        Rwa2 = np.dot(Roos, wa2)
        PAP[:, k] = Swa1 * Rwa2 - Swa2 * Rwa1
    
    return Wfull1, Wfull2, Dfull, Wsym, Dsym, Wasym, Dasym, PP, PEP, PAP
