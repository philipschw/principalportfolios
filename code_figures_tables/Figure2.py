import numpy as np
import scipy.io as sio
import os

# Set up data
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

data = sio.loadmat('./Data/' + filename)
retd = data['retd']
datesd = data['datesd'].squeeze()

# Restrict dates
ixtmp = np.where(np.floor(datesd / 10000) == minyr)[0][0]
if ixtmp == []:
    ixtmp = 0
strt = max(ixtmp - rollwin * fwdwin - 2, 0)
loc = np.logical_and(datesd >= np.floor(datesd[strt] / 10000) * 10000, datesd < (maxyr + 1) * 10000)
retd = retd[loc]
datesd = datesd[loc]

# Drop columns with missing data
loc = np.sum(np.isnan(retd), axis=0) / retd.shape[0] <= 0.02
retd = retd[:, loc]
retd[np.isnan(retd)] = 0

# Vol-standardize for MF data
if volstd == 1:
    retd = np.log(1 + retd)

# Load/merge Fama-French factor data for benchmarking
data_ff = sio.loadmat('./Data/ff5daily')
fffac = data_ff['fffac']
ffdates = data_ff['ffdates'].squeeze()
rf = data_ff['rf'].squeeze()

_, ia, ib = np.intersect1d(datesd, ffdates, return_indices=True)
retd = retd[ia]
datesd = datesd[ia]
rfd = rf[ib]
fffacd = fffac[ib, :]
ff5daily_vars = ['fffac', 'ffdates', 'rf']
data_ff.close()

# Find non-overlapping forecast observations
T = len(datesd)
Tno = T // fwdwin

# Index unique non-overlapping observations
fwdix = np.kron(np.arange(1, Tno + 1), np.ones(fwdwin, dtype=int))
fwdix = np.concatenate([fwdix, np.full(T - len(fwdix), np.nan)])
N = retd.shape[1]
Rfwd = np.nan * np.ones((Tno, N))
FFfwd = np.nan * np.ones((Tno, fffacd.shape[1]))
S = np.nan * np.ones((Tno, N))
datesno = np.nan * np.ones(Tno)
skip = 2
for t in range(1, Tno + 1):
    # Forecast target observation
    loc = np.where(fwdix == t)[0]
    if volstd == 1:
        tmpstd = np.nanstd(retd[max(loc[0] - 20, 0): max(loc[0], 20)], axis=0)
        Rfwd[t - 1] = np.sum(retd[loc] / tmpstd, axis=0)
    else:
        Rfwd[t - 1] = np.prod(1 + retd[loc], axis=0) - 1
    datesno[t - 1] = datesd[loc[-1]]

    # Forecast FF observation
    mkt = fffacd[loc, 0] + rfd[loc]
    ffoth = fffacd[loc, 1:]
    if volstd == 1:
        FFfwd[t - 1] = np.sum(np.column_stack((mkt, ffoth)), axis=0)
        RFfwd = np.sum(rfd[loc])
    else:
        FFfwd[t - 1] = np.prod(1 + np.column_stack((mkt, ffoth)), axis=0) - 1
        RFfwd = np.prod(1 + rfd[loc]) - 1
    FFfwd[t - 1, 0] -= RFfwd

    # Signal observation (indexed to align with return it predicts, lagged 'skip' days)
    sigloc = np.arange(loc[0] - skip - (momwin - 1), loc[0] - skip)
    if sigloc[0] < 0:
        continue
    if Stype == 'mom':
        if volstd == 1:
            S[t - 1] = np.sum(retd[sigloc] / tmpstd, axis=0)
        else:
            S[t - 1] = np.sum(retd[sigloc], axis=0)
    elif Stype == 'vol':
        S[t - 1] = np.sqrt(np.sum(retd[sigloc] ** 2, axis=0))
    elif Stype == 'invvol':
        voltmp = np.sqrt(np.sum(retd[sigloc] ** 2, axis=0))
        S[t - 1] = 1.0 / voltmp
        S[t - 1][np.isinf(S[t - 1])] = 1.0 / np.nanmean(voltmp)
    elif Stype == 'beta':
        X = np.column_stack((np.ones(len(sigloc)), fffacd[sigloc, 0]))
        Y = retd[sigloc]
        B = np.linalg.inv(X.T @ X) @ X.T @ Y
        S[t - 1] = B[1]
    elif Stype == 'corr':
        X = np.column_stack((np.ones(len(sigloc)), fffacd[sigloc, 0]))
        Y = retd[sigloc]
        B = np.linalg.inv(X.T @ X) @ X.T @ Y
        S[t - 1] = B[1] / np.nanstd(retd[sigloc]) * np.nanstd(fffacd[sigloc, 0])
data_ff_vars = ['fffac', 'ffdates', 'rf', 'rfd', 'fffacd', 'FFfwd', 'S', 'datesno']
del data_ff, fwdix, mkt, ffoth, tmpstd, sigloc, X, Y, B, loc

# Slice data
Tno, N = S.shape
Rtrnslc = [Rfwd[tau - rollwin: tau] for tau in range(rollwin, Tno)]
Strnslc = [S[tau - rollwin: tau] for tau in range(rollwin, Tno)]
Rtstslc = [Rfwd[tau] for tau in range(rollwin, Tno)]
Ststslc = [S[tau] for tau in range(rollwin, Tno)]
del Rfwd, S

print('Slicing complete')

# Recursive procedure
F = np.nan * np.ones(Tno)
Fsym = np.nan * np.ones((Tno, N))
Fasym = np.nan * np.ones((Tno, N // 2))
Dsym = np.nan * np.ones((Tno, N))
Dasym = np.nan * np.ones((Tno, N // 2))
Vsym = [np.nan * np.ones((N, N)) for _ in range(Tno)]
Vasym = [np.nan * np.ones((N, N // 2)) for _ in range(Tno)]
for tau in range(rollwin, Tno):
    # Carve out training data
    Rtrn = Rtrnslc[tau - rollwin]
    Strn = Strnslc[tau - rollwin]
    if np.isnan(Rtrn).sum() + np.isnan(Strn).sum() > 0:
        continue

    # Carve out test data
    Rtst = Rtstslc[tau - rollwin]
    Stst = Ststslc[tau - rollwin]

    # Rank-standardize signal
    if cntr == 1:
        Strn = rankstdize(Strn)
        Stst = rankstdize(Stst)

    # Cross-section demean return (training data only)
    if cntr == 1:
        Rtrn = Rtrn - np.nanmean(Rtrn, axis=1, keepdims=True)
        Rtst = Rtst - np.nanmean(Rtst, axis=1, keepdims=True)

    # Baseline factor
    F[tau] = np.sum(Stst * Rtst, axis=1)

    # Build portfolios
    Wfull1, Wfull2, Dfull, Wsym, Dsym[tau], Wasym, Dasym[tau], _, Fs, Fa = specptfs(Rtrn, Strn, Rtst, Stst)
    Fsym[tau] = Fs
    Fasym[tau] = Fa
    Vsym[tau] = Wfull1
    Vasym[tau] = Wasym
    if tau % 100 == 0:
        print(tau)
datafile = './Data/Results/' + os.path.splitext(filename)[0] + '-rollwin-' + str(rollwin) + '-momwin-' + str(momwin) + '-fwdwin-' + str(fwdwin) + '-center-' + str(cntr) + '-' + Stype + '-nonoverlap'
if savedata == 1:
    data = {
        'F': F,
        'Vsym': Vsym,
        'Vasym': Vasym,
        'Fsym': Fsym,
        'Fasym': Fasym,
        'Dsym': Dsym,
        'Dasym': Dasym,
        'datesno': datesno
    }
    sio.savemat(datafile, data)


#**************************************************************************
# Plots eigenvectors
#**************************************************************************
figdir = './Figures/'

j = 1
plt.close('all')
V = np.concatenate([a[:, j][None, :] for a in Vsym])
Va = np.concatenate([a[:, j][None, :] for a in Vasym])
Vr = np.real(Va)
Vi = np.imag(Va)
del Va

# Labels
labs = [f"S{str(i)}V{str(j)}" for i in range(1, 6) for j in range(1, 6)]

# Align signs of eigenvectors
for t in range(1, V.shape[0]):
    if np.isnan(V[t, :]).sum() > 0:
        continue
    C = np.corrcoef(V[t-1:t, :].T)
    if C[0, 1] < 0:
        V[t, :] = -V[t, :]
    C = np.corrcoef(Vr[t-1:t, :].T)
    if C[0, 1] < 0:
        Vr[t, :] = -Vr[t, :]
    C = np.corrcoef(Vi[t-1:t, :].T)
    if C[0, 1] < 0:
        Vi[t, :] = -Vi[t, :]

# Average symmetric portfolio weights
plt.bar(range(len(labs)), np.nanmean(V, axis=0))
plt.xticks(range(len(labs)), labs, rotation=90)
plt.xlabel('Eigenvectors')
plt.ylabel('Average Portfolio Weight')
plt.gca().set_yticklabels(['{:.1f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.gca().set_ylim(bottom=0)
plt.gca().set_title('Average Symmetric Portfolio Weights')
plt.gca().set_axisbelow(True)
plt.grid(axis='y')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(left=False)
plt.gca().legend().set_visible(False)
plt.gcf().set_size_inches(12, 8)
plt.savefig(figdir + 'Figure2a.eps', format='eps')
plt.close('all')

# Average asymmetric portfolio weights
plt.bar(range(len(labs)), np.nanmean(Vr, axis=0), label='Real ($x$)')
plt.bar(range(len(labs)), np.nanmean(Vi, axis=0), label='Imaginary ($y$)')
plt.xticks(range(len(labs)), labs, rotation=90)
plt.xlabel('Eigenvectors')
plt.ylabel('Average Portfolio Weight')
plt.gca().set_yticklabels(['{:.1f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.gca().set_ylim(bottom=0)
plt.gca().set_title('Average Asymmetric Portfolio Weights')
plt.gca().set_axisbelow(True)
plt.grid(axis='y')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(left=False)
plt.gca().legend(loc='upper left')
plt.gcf().set_size_inches(12, 8)
plt.savefig(figdir + 'Figure2d.eps', format='eps')
plt.close('all')

#**************************************************************************
# Plots Symmetric weights
#**************************************************************************

w = np.nan * np.ones((Vsym.shape[0], 25))
scl = np.nan * np.ones(Vsym.shape[0])
S = np.nan * np.ones((Vsym.shape[0], 25))
for t in range(1, Vsym.shape[0]):
    V1 = Vsym[t][:, 0]
    if np.isnan(V1).sum() > 0:
        continue
    S[t, :] = Ststslc[t-1].flatten()
    S[t, :] = rankstdize(S[t, :])
    scl[t] = V1 @ S[t, :].T
    w[t, :] = V1 @ V1.T @ S[t, :].T

plt.bar(range(len(labs)), np.nanmean(w, axis=0))
plt.xticks(range(len(labs)), labs, rotation=90)
plt.xlabel('Eigenvectors')
plt.ylabel('Average Portfolio Weight')
plt.gca().set_yticklabels(['{:.1f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.gca().set_ylim(bottom=0)
plt.gca().set_title('Average Symmetric Weights')
plt.gca().set_axisbelow(True)
plt.grid(axis='y')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(left=False)
plt.gcf().set_size_inches(12, 8)
plt.savefig(figdir + 'Figure2c.eps', format='eps')
plt.close('all')

plotwithaxis(scl, np.floor(datesno/100))
plt.ylim([-1.5, 1.5])
plt.gca().set_xlabel('Year')
plt.gca().set_ylabel('Scale')
plt.gca().set_title('Scale of Symmetric Weights')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(left=False)
plt.gca().legend().set_visible(False)
plt.gcf().set_size_inches(12, 8)
plt.savefig(figdir + 'Figure2b.eps', format='eps')
plt.close('all')

#**************************************************************************
# Plots Asymmetric weights
#**************************************************************************

w = np.nan * np.ones((Vsym.shape[0], 25))
sclr = np.nan * np.ones(Vsym.shape[0])
scli = np.nan * np.ones(Vsym.shape[0])
S = np.nan * np.ones((Vsym.shape[0], 25))
for t in range(1, Vsym.shape[0]):
    V1 = Vasym[t][:, 0]
    Vr = np.real(V1)
    Vi = np.imag(V1)

    if np.isnan(V1).sum() > 0:
        continue
    S[t, :] = Ststslc[t-1].flatten()
    S[t, :] = rankstdize(S[t, :])
    sclr[t] = S[t, :] @ Vr
    scli[t] = S[t, :] @ Vi
    w[t, :] = (S[t, :] @ Vr) * Vi - (S[t, :] @ Vi) * Vr

plt.bar(range(len(labs)), np.nanmean(w, axis=0))
plt.xticks(range(len(labs)), labs, rotation=90)
plt.xlabel('Eigenvectors')
plt.ylabel('Average Portfolio Weight')
plt.gca().set_yticklabels(['{:.1f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.gca().set_ylim(bottom=0)
plt.gca().set_title('Average Asymmetric Weights')
plt.gca().set_axisbelow(True)
plt.grid(axis='y')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(left=False)
plt.gca().legend(loc='upper left')
plt.gcf().set_size_inches(12, 8)
plt.savefig(figdir + 'Figure2f.eps', format='eps')
plt.close('all')

plotwithaxis(np.column_stack([sclr, scli]), np.floor(datesno/100))
plt.legend(["S'x", "S'y"], loc='upper left')
plt.ylim([-1.5, 1.5])
plt.xlabel('Year')
plt.ylabel('Scale')
plt.gca().set_title('Scale of Asymmetric Weights')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(left=False)
plt.gcf().set_size_inches(12, 8)
plt.savefig(figdir + 'Figure2e.eps', format='eps')
plt.close('all')

