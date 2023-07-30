# This script creates Table 2.

# Start notice
print("Start Script")

# import packages
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
from tabulate import tabulate

# add path
sys.path.append('../code_solver')

# import self-written auxiliary functions
from rankstdize import rankstdize
from specptfs import specptfs
from linreg import linreg

# turn off certain warnings
np.warnings.filterwarnings('ignore', category=np.ComplexWarning)
np.warnings.filterwarnings('ignore', category=RuntimeWarning)
np.warnings.filterwarnings('ignore', category=FutureWarning)


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
filename = '25_Size_BM'

if filename.startswith('MFret'):
    volstd = 1
else:
    volstd = 0

# Load data
data = np.load('../data/' + filename + '.npz')
retd = data['retd']
datesd = data['datesd']

# Convert datetime64 to string with format 'YYYYMMDD'
datesd = np.datetime_as_string(datesd, unit='D')

# Extract the year, month, and day from each string and convert them to doubles
datesd = np.array([float(dt[:4] + dt[5:7] + dt[8:10]) for dt in datesd])

# Restrict dates
ixtmp = np.where(np.floor([date / 10000 for date in datesd]) == minyr)[0][0] + 1
ixtmp = 1 if not ixtmp else ixtmp
strt = max(ixtmp - rollwin * fwdwin - 2, 0)
loc = (datesd >= np.floor(datesd[strt] / 10000) * 10000) & (datesd < (maxyr + 1) * 10000)
retd = retd[loc, :]
datesd = datesd[loc]

# Drop columns with missing data
## Calculate the fraction of missing data for each column
missing_frac = np.sum(np.isnan(retd), axis=0) / retd.shape[0]
## Filter columns based on the fraction of missing data being less than or equal to 0.02
loc = missing_frac <= 0.02
retd = retd[:, loc]
## Replace missing values with zero
retd[np.isnan(retd)] = 0

# Vol-standardize for MF data
if volstd == 1:
    retd = np.log(1 + retd)

# Load/merge Fama-French factor data for benchmarking
data_ff = sio.loadmat('../data/ff5daily.mat')
fffac = data_ff['fffac']
ffdates = data_ff['ffdates']
rf = data_ff['rf']
## Find the intersection indices
_, ia, ib = np.intersect1d(datesd, ffdates, return_indices=True)
## Filter variables based on the intersection
retd = retd[ia, :]
datesd = datesd[ia]
rfd = rf[ib]
fffacd = fffac[ib, :]

# Find non-overlapping forecast observations
T = len(datesd)
Tno = T // fwdwin

# Index unique non-overlapping observations
fwdix = np.kron(np.arange(1, Tno + 1), np.ones(fwdwin, dtype=int))
fwdix = np.concatenate((fwdix, np.full(T - len(fwdix), np.nan)))
N = retd.shape[1]
Rfwd = np.full((Tno, N), np.nan)
FFfwd = np.full((Tno, fffacd.shape[1]), np.nan)
S = np.full((Tno, N), np.nan)
datesno = np.full(Tno, np.nan)
skip = 2

for t in range(1, Tno + 1):

    # Forecast target observation
    loc = np.where(fwdix == t)[0]
    if volstd == 1:
        tmpstd = np.nanstd(retd[max(loc[0]-20,0):max(loc[0]-1,20), :], axis=0, ddof=1)
        Rfwd[t-1, :] = np.sum(np.divide(np.array(retd)[loc],tmpstd), axis=0)
    else:
        Rfwd[t-1, :] = np.prod(1 + np.array(retd)[loc], axis=0) - 1
    datesno[t - 1] = datesd[loc[-1]]

    # Forecast FF observation
    mkt = fffacd[loc, 0] + np.squeeze(rfd[loc])
    ffoth = fffacd[loc, 1:]
    if volstd == 1:
        FFfwd[t-1, :] = np.sum(np.concatenate((np.expand_dims(mkt, axis=1), ffoth), axis=1), axis=0)
        RFfwd = np.sum(np.array(rfd)[loc], axis=0)
    else:
        FFfwd[t-1, :] = np.prod(1 + np.concatenate((np.expand_dims(mkt, axis=1), ffoth), axis=1), axis=0) - 1
        RFfwd = np.prod(1 + np.array(rfd)[loc], axis=0) - 1
    FFfwd[t-1, 0] = FFfwd[t-1, 0] - RFfwd


    # Signal observation (indexed to align with return it predicts, lagged 'skip' days)
    sigloc = np.arange(loc[0] - skip - (momwin - 1), loc[0] - skip + 1)
    if sigloc[0] < 0:
        continue
    if Stype == 'mom':
        if volstd == 1:
            S[t - 1, :] = np.sum(np.divide(np.array(retd)[sigloc],tmpstd), axis=0)
        else:
            S[t - 1, :] = np.sum(np.array(retd)[sigloc], axis=0)
    elif Stype == 'vol':
        S[t - 1, :] = np.sqrt(np.nansum(retd[sigloc, :] ** 2, axis=0))
    elif Stype == 'invvol':
        voltmp = np.sqrt(np.nansum(retd[sigloc, :] ** 2, axis=0))
        S[t - 1, :] = 1 / voltmp
        S[t - 1, np.isinf(S[t - 1, :])] = 1 / np.nanmean(voltmp)
    elif Stype == 'beta':
        X = np.column_stack((np.ones(len(sigloc)), fffacd[sigloc, 0]))
        Y = retd[sigloc, :]
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        S[t - 1, :] = B[1, :]
    elif Stype == 'corr':
        X = np.column_stack((np.ones(len(sigloc)), fffacd[sigloc, 0]))
        Y = retd[sigloc, :]
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        S[t - 1, :] = B[1, :] / np.nanstd(retd[sigloc, :], axis=0) * np.nanstd(fffacd[sigloc, 0])

# Slice data
Tno, N = S.shape

Rtrnslc = [[] for _ in range(rollwin-1)]
Strnslc = [[] for _ in range(rollwin-1)]
Rtstslc = [[] for _ in range(rollwin-1)]
Ststslc = [[] for _ in range(rollwin-1)]

for tau in range(rollwin, Tno):
    Rtrnslc.append(Rfwd[tau - rollwin:tau, :])
    Strnslc.append(S[tau - rollwin:tau, :])
    Rtstslc.append(Rfwd[tau, :])
    Ststslc.append(S[tau, :])

# Recursive procedure
Ftil = np.full(Tno, np.nan)
PP = np.full((Tno, N), np.nan)
PEP = np.full((Tno, N), np.nan)
PAP = np.full((Tno, N // 2), np.nan)
Dfull = np.full((Tno, N), np.nan)
Dsym = np.full((Tno, N), np.nan)
Dasym = np.full((Tno, N // 2), np.nan)
Wfull1 = [np.full((N, N), np.nan)] * Tno
Wfull2 = [np.full((N, N), np.nan)] * Tno
Wsym = [np.full((N, N), np.nan)] * Tno
Wasym = [np.full((N, N // 2), np.nan)] * Tno

print("Build Principal Portfolios")
for tau in tqdm(range(rollwin, Tno)):

    # Carve out training data
    Rtrn = Rtrnslc[tau-1]
    Strn = Strnslc[tau-1]
    if np.isnan(np.sum(Rtrn)) or np.isnan(np.sum(Strn)):
        continue

    # Carve out test data
    Rtst = Rtstslc[tau-1]
    Stst = Ststslc[tau-1]

    # Rank-standardize signal
    if cntr == 1:
        Strn = rankstdize(Strn)
        Stst = rankstdize(np.atleast_2d(Stst))[0]

    # Cross-section demean return (training data only)
    if cntr == 1:
        Rtrn = Rtrn - np.nanmean(Rtrn, axis=1)[:, np.newaxis]
        Rtst = Rtst - np.nanmean(Rtst)

    # Baseline factor
    Ftil[tau] = np.sum(Stst * Rtst)

    # Build portfolios
    W1, W2, Df, Ws, Ds, Wa, Da, PPtmp, PEPtmp, PAPtmp = specptfs(Rtrn, Strn, Rtst, Stst)
    PP[tau, :] = PPtmp
    PEP[tau, :] = PEPtmp
    PAP[tau, :] = PAPtmp
    Dfull[tau, :] = Df
    Dsym[tau, :] = Ds
    Dasym[tau, :] = Da # ComplexWarning: Casting complex values to real discards the imaginary part
    Wfull1[tau] = W1
    Wfull2[tau] = W2
    Wsym[tau] = Ws
    Wasym[tau] = Wa

# Regressions of PPs vs factors
filename = '25_Size_BM'
figdir = '../figures/'

# Rescale to market vol
PP3 = np.mean(PP[:, :3], axis=1)
PEP3 = np.mean(PEP[:, :3], axis=1)
PAP3 = np.mean(PAP[:, :3], axis=1)
PEPPAP3 = PEP3 + PAP3 * np.nanstd(PEP3) / np.nanstd(PAP3)

Ftil = Ftil / np.nanstd(Ftil, ddof=1) * np.nanstd(FFfwd[:, 0], ddof=1)
PP3 = PP3 / np.nanstd(PP3, ddof=1) * np.nanstd(FFfwd[:, 0], ddof=1)
PEP3 = PEP3 / np.nanstd(PEP3, ddof=1) * np.nanstd(FFfwd[:, 0], ddof=1)
PAP3 = PAP3 / np.nanstd(PAP3, ddof=1) * np.nanstd(FFfwd[:, 0], ddof=1)
PEPPAP3 = PEPPAP3 / np.nanstd(PEPPAP3, ddof=1) * np.nanstd(FFfwd[:, 0], ddof=1)

print("Run Regressions")
Fstats = linreg(Ftil, FFfwd)
fullstats = linreg(PP3, np.column_stack((Ftil, FFfwd)))
symstats = linreg(PEP3, np.column_stack((Ftil, FFfwd)))
asymstats = linreg(PAP3, np.column_stack((Ftil, FFfwd)))
bothstats = linreg(PEPPAP3, np.column_stack((Ftil, FFfwd)))

print("Create Table")
out = np.vstack((
    np.hstack((np.full((2, 1), np.nan), np.vstack((Fstats[0][1:], Fstats[1][1:])), [[Fstats[0][0]*100*250/fwdwin, Fstats[2]],[Fstats[1][0], np.nan]])),
    np.hstack((np.vstack((fullstats[0][1:], fullstats[1][1:])), [[fullstats[0][0]*100*250/fwdwin, fullstats[2]],[fullstats[1][0], np.nan]])),
    np.hstack((np.vstack((symstats[0][1:], symstats[1][1:])), [[symstats[0][0]*100*250/fwdwin, symstats[2]],[symstats[1][0], np.nan]])),
    np.hstack((np.vstack((asymstats[0][1:], asymstats[1][1:])), [[asymstats[0][0]*100*250/fwdwin, asymstats[2]],[asymstats[1][0], np.nan]])),
    np.hstack((np.vstack((bothstats[0][1:], bothstats[1][1:])), [[bothstats[0][0]*100*250/fwdwin, bothstats[2]],[bothstats[1][0], np.nan]]))
))

labs = ['Factor', '(t)', 'PP 1-3', '(t)', 'PEP 1-3', '(t)', 'PAP 1-3', '(t)', 'PEP and PAP 1-3', '(t)']
tabout = np.vstack((['Portfolio', 'Factor', 'Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA', 'Alpha', '$R^2$'], np.column_stack((labs, np.vectorize(lambda x: f'{x:.2f}')(out)))))
np.save(figdir + 'table2.npy', tabout)

# Print the LaTeX table
print(tabulate(tabout, headers='firstrow', tablefmt='latex'))

# End notice
print("End Script")