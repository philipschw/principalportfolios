import numpy as np
import matplotlib.pyplot as plt

rollwin = 120
momwin = 20
fwdwin = 20
minyr = 1963
maxyr = 2019
writefile = []
cntr = 1
Neig = 3
Stype = 'mom'
savedata = 0
filename = '25_Size_BM.mat'

if filename.startswith('MFret'):
    volstd = 1
else:
    volstd = 0

data = np.load('./Data/' + filename)
retd = data['retd']
datesd = data['datesd']

# Restrict dates
ixtmp = np.where(np.floor(datesd / 10000) == minyr)[0][0]
if ixtmp.size == 0:
    ixtmp = 1
strt = max(ixtmp - rollwin * fwdwin - 2, 1)
loc = (datesd >= np.floor(datesd[strt] / 10000) * 10000) & (datesd < (maxyr + 1) * 10000)
retd = retd[loc]
datesd = datesd[loc]

# Drop columns with missing data. Allow for small fraction of missing data
# to be replaced with zero. This can be improved to allow for calculations
# in the presence of missing data.
loc = np.sum(np.isnan(retd), axis=0) / retd.shape[0] <= 0.02
retd = retd[:, loc]
retd[np.isnan(retd)] = 0

# Vol-standardize for MF data
if volstd == 1:
    retd = np.log(1 + retd)

# Load/merge Fama-French factor data for benchmarking
ffdata = np.load('./Data/ff5daily.npz')
fffac = ffdata['fffac']
ffdates = ffdata['ffdates']
rf = ffdata['rf']
ia, ib = np.where(np.isin(datesd, ffdates))
retd = retd[ia]
datesd = datesd[ia]
rfd = rf[ib]
fffacd = fffac[ib]

# Find non-overlapping forecast observations
T = len(datesd)
Tno = T // fwdwin

# Index unique non-overlapping observations
fwdix = np.repeat(np.arange(1, Tno + 1), fwdwin)
fwdix = np.concatenate((fwdix, np.full(T - len(fwdix), np.nan)))
N = retd.shape[1]
Rfwd = np.zeros((Tno, N))
FFfwd = np.zeros((Tno, fffacd.shape[1]))
S = np.zeros((Tno, N))
datesno = np.zeros(Tno)
skip = 2
for t in range(Tno):
    # Forecast target observation
    loc = np.where(fwdix == (t + 1))[0]
    if volstd == 1:
        tmpstd = np.nanstd(retd[max(loc[0] - 20, 0):max(loc[0], 20)], axis=0)
        Rfwd[t] = np.sum(retd[loc] / tmpstd, axis=0)
    else:
        Rfwd[t] = np.prod(1 + retd[loc], axis=0) - 1
    datesno[t] = datesd[loc[-1]]

    # Forecast FF observation
    mkt = fffacd[loc, 0] + rfd[loc]
    ffoth = fffacd[loc, 1:]
    if volstd == 1:
        FFfwd[t] = np.sum(np.concatenate(([mkt], ffoth), axis=0), axis=0)
        RFfwd = np.sum(rfd[loc], axis=0)
    else:
        FFfwd[t] = np.prod(1 + np.concatenate(([mkt], ffoth), axis=0), axis=0) - 1
        RFfwd = np.prod(1 + rfd[loc], axis=0) - 1
    FFfwd[t, 0] -= RFfwd

    # Signal observation (indexed to align with return it predicts, lagged 'skip' days)
    sigloc = np.arange(loc[0] - skip - (momwin - 1), loc[0] - skip)
    if sigloc[0] < 0:
        continue
    if Stype == 'mom':
        if volstd == 1:
            S[t] = np.sum(retd[sigloc] / tmpstd, axis=0)
        else:
            S[t] = np.sum(retd[sigloc], axis=0)
    elif Stype == 'vol':
        S[t] = np.sqrt(np.sum(retd[sigloc] ** 2, axis=0))
    elif Stype == 'invvol':
        voltmp = np.sqrt(np.sum(retd[sigloc] ** 2, axis=0))
        S[t] = 1 / voltmp
        S[t][np.isinf(S[t])] =1 / np.nanmean(voltmp)
    elif Stype == 'beta':
        X = np.column_stack((np.ones(len(sigloc)), fffacd[sigloc, 0]))
        Y = retd[sigloc]
        B = np.linalg.inv(X.T @ X) @ X.T @ Y
        S[t] = B[1]
    elif Stype == 'corr':
        X = np.column_stack((np.ones(len(sigloc)), fffacd[sigloc, 0]))
        Y = retd[sigloc]
        B = np.linalg.inv(X.T @ X) @ X.T @ Y
        S[t] = B[1] / np.nanstd(retd[sigloc]) * np.nanstd(fffacd[sigloc, 0])
        
# Slicing data
Tno, N = S.shape
Rtrnslc = [None] * (Tno - rollwin)
Strnslc = [None] * (Tno - rollwin)
Rtstslc = [None] * (Tno - rollwin)
Ststslc = [None] * (Tno - rollwin)
for tau in range(rollwin, Tno):
    Rtrnslc[tau - rollwin] = Rfwd[tau - rollwin:tau]
    Strnslc[tau - rollwin] = S[tau - rollwin:tau]
    Rtstslc[tau - rollwin] = Rfwd[tau]
    Ststslc[tau - rollwin] = S[tau]
print('Slicing complete')

# Recursive procedure
Ftil = np.zeros(Tno)
PP = np.zeros((Tno, N))
PEP = np.zeros((Tno, N))
PAP = np.zeros((Tno, N // 2))
Dfull = np.zeros((Tno, N))
Dsym = np.zeros((Tno, N))
Dasym = np.zeros((Tno, N // 2))
Wfull1 = [None] * Tno
Wfull2 = [None] * Tno
Wsym = [None] * Tno
Wasym = [None] * Tno

for tau in range(rollwin, Tno):
    # Carve out training data
    Rtrn = Rtrnslc[tau - rollwin]
    Strn = Strnslc[tau - rollwin]
    if np.isnan(Rtrn).any() or np.isnan(Strn).any():
        continue

    # Carve out test data
    Rtst = Rtstslc[tau - rollwin]
    Stst = Ststslc[tau - rollwin]

    # Rank-standardize signal
    if cntr == 1:
        Strn = (Strn - np.mean(Strn)) / np.std(Strn)
        Stst = (Stst - np.mean(Stst)) / np.std(Stst)

    # Cross-section demean return (training data only)
    if cntr == 1:
        Rtrn = Rtrn - np.mean(Rtrn, axis=1)[:, np.newaxis]
        Rtst = Rtst - np.mean(Rtst, axis=1)[:, np.newaxis]

    # Baseline factor
    Ftil[tau] = np.dot(Stst, Rtst.T)

    # Build portfolios
    W1, W2, Df, Ws, Ds, Wa, Da, PPtmp, PEPtmp, PAPtmp = specptfs(Rtrn, Strn, Rtst, Stst)
    PP[tau] = PPtmp
    PEP[tau] = PEPtmp
    PAP[tau] = PAPtmp
    Dfull[tau] = Df
    Dsym[tau] = Ds
    Dasym[tau] = Da
    Wfull1[tau] = W1
    Wfull2[tau] = W2
    Wsym[tau] = Ws
    Wasym[tau] = Wa
    if tau % 100 == 0:
        print(tau)
print('Recursive procedure complete')

# Regressions of PPs vs factors
filename = '25_Size_BM'
figdir = './Figures/'

# Rescale to market vol
PP3 = np.mean(PP[:, :3], axis=1)
PEP3 = np.mean(PEP[:, :3], axis=1)
PAP3 = np.mean(PAP[:, :3], axis=1)
PEPPAP3 = PEP3 + PAP3 * np.std(PEP3) / np.std(PAP3)

Ftil = Ftil / np.nanstd(Ftil) * np.nanstd(FFfwd[:, 1])
PP3 = PP3 / np.nanstd(PP3) * np.nanstd(FFfwd[:, 1])
PEP3 = PEP3 / np.nanstd(PEP3) * np.nanstd(FFfwd[:, 1])
PAP3 = PAP3 / np.nanstd(PAP3) * np.nanstd(FFfwd[:, 1])
PEPPAP3 = PEPPAP3 / np.nanstd(PEPPAP3) * np.nanstd(FFfwd[:, 1])

Fstats = regstats(Ftil, FFfwd, 'linear', {'rsquare', 'tstat'})
fullstats = regstats(PP3, np.column_stack((Ftil, FFfwd)), 'linear', {'rsquare', 'tstat'})
symstats = regstats(PEP3, np.column_stack((Ftil, FFfwd)), 'linear', {'rsquare', 'tstat'})
asymstats = regstats(PAP3, np.column_stack((Ftil, FFfwd)), 'linear', {'rsquare', 'tstat'})
bothstats = regstats(PEPPAP3, np.column_stack((Ftil, FFfwd)), 'linear', {'rsquare', 'tstat'})

out = np.vstack((
    np.hstack((np.nan * np.ones((2, 1)), np.vstack((Fstats.tstat.beta[1:], Fstats.tstat.t[1:])))),
    np.hstack((fullstats.tstat.beta, fullstats.tstat.t)),
    np.hstack((symstats.tstat.beta, symstats.tstat.t)),
    np.hstack((asymstats.tstat.beta, asymstats.tstat.t)),
    np.hstack((bothstats.tstat.beta, bothstats.tstat.t))
))

out[0, 2:] *= (250 / fwdwin)
out[:, -1] = np.hstack((Fstats.rsquare, fullstats.rsquare, symstats.rsquare, asymstats.rsquare, bothstats.rsquare, np.nan))

labs = ['Factor', '(t)', 'PP 1-3', '(t)', 'PEP 1-3', '(t)', 'PAP 1-3', '(t)', 'PEP and PAP 1-3', '(t)']
tabout = np.column_stack((labs, np.vectorize(lambda x: f'{x:.2f}')(out)))
np.save(figdir + 'table2.npy', tabout)

