# This script creates Figure 2.

# Start notice
print("Start Script")
# import packages
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Patch

# add path
sys.path.append('../code_solver')

# import self-written auxiliary functions
from rankstdize import rankstdize
from specptfs import specptfs

# turn off certain warnings
np.warnings.filterwarnings('ignore', category=np.ComplexWarning)
np.warnings.filterwarnings('ignore', category=RuntimeWarning)

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
filename = '25_Size_BM'

# Define volstd
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
F = np.full(Tno, np.nan)
Fsym = np.full((Tno, N), np.nan)
Fasym = np.full((Tno, N // 2), np.nan)
Dsym = np.full((Tno, N), np.nan)
Dasym = np.full((Tno, N // 2), np.nan)
Vsym = [np.full((N, N), np.nan)] * Tno
Vasym = [np.full((N, N // 2), np.nan)] * Tno

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
    F[tau] = np.sum(Stst * Rtst)

    # Build portfolios
    _, _, _, Vs, Ds, Va, Da, _, Fs, Fa = specptfs(Rtrn, Strn, Rtst, Stst)
    Fsym[tau, :] = Fs
    Fasym[tau, :] = Fa
    Dsym[tau, :] = Ds
    Dasym[tau, :] = Da
    Vsym[tau] = Vs
    Vasym[tau] = Va

# Save results to file
datafile = f'../data/Results/{filename.replace(".npz", "")}-rollwin-{rollwin}-momwin-{momwin}-fwdwin-{fwdwin}-center-{cntr}-{Stype}-nonoverlap'
if savedata == 1:
    np.savez(datafile, F=F, Vsym=Vsym, Vasym=Vasym, Fsym=Fsym, Fasym=Fasym, Dsym=Dsym, Dasym=Dasym, datesno=datesno)


print("Create Plots")
# Plots eigenvectors

figdir = '../figures/'

j = 0
V = np.vstack([a[:, j].flatten() for a in Vsym])
Va = np.vstack([a[:, j].flatten() for a in Vasym])
Vr = np.real(Va)
Vi = np.imag(Va)

# Labels
labs = [f"S{str(i)}V{str(j)}" for i in range(1, 6) for j in range(1, 6)]

# Align signs of eigenvectors
nanidxV = np.isnan(V)
nanidxVr = np.isnan(Vr)
nanidxVi = np.isnan(Vi)
V[nanidxV] = 0
Vr[nanidxVr] = 0
Vi[nanidxVi] = 0
for t in range(1, V.shape[0]):
    if np.sum(V[t, :]) == 0:
        continue
    C = np.corrcoef(V[t-1:t+1, :])
    if C[0, 1] < 0:
        V[t, :] = -V[t, :]
    C = np.corrcoef(Vr[t-1:t+1, :])
    if C[0, 1] < 0:
        Vr[t, :] = -Vr[t, :]
    C = np.corrcoef(Vi[t-1:t+1, :])
    if C[0, 1] < 0:
        Vi[t, :] = -Vi[t, :]

V[nanidxV] = np.nan
Vr[nanidxVr] = np.nan
Vi[nanidxVi] = np.nan

# Average symmetric portfolio weights
plt.bar(range(len(labs)), np.nanmean(V.tolist(), axis=0), edgecolor='black')
#plt.bar(range(len(labs)), np.nanmedian(V.tolist(), axis=0), edgecolor='black')
plt.xticks(range(len(labs)), labs, rotation=90)
plt.gca().set_title('Average PEP eigenvector $w_1$')
plt.gca().set_axisbelow(True)
plt.gcf().set_size_inches(12.8, 10)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.axhline(y=0, color='black')
plt.savefig(figdir + 'Figure2a.jpg', dpi=300)
plt.close()

# Average asymmetric portfolio weights
fig, ax = plt.subplots(figsize=(12.8,10))
meanVrVi = np.matrix([np.nanmean(Vr, axis=0), np.nanmean(Vi, axis=0)]).T
x_offset = -0.15
for j in range(len(labs)):
    ax.bar(j + x_offset, meanVrVi[j,0], width=0.3, edgecolor='black', color= (0, 0.4470, 0.7410))
for j in range(len(labs)):
    ax.bar(j - x_offset, meanVrVi[j,1], width=0.3, edgecolor='black', color=(0.8500, 0.3250, 0.0980))
plt.xticks(range(len(labs)), labs, rotation=90)
plt.gca().set_title('Average PAP eigenvector $x$, $y$')
plt.gca().set_axisbelow(True)
plt.gcf().set_size_inches(12.8, 10)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.axhline(y=0, color='black')
legend_handles = [Patch(facecolor=color) for color in [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980)]]
legend_labels = ['Real ($x$)', 'Imaginary ($x$)']
fig.gca().legend(legend_handles, legend_labels, prop={'size': 15})
plt.savefig(figdir + 'Figure2d.jpg', dpi=300)
plt.close()


# Plots Symmetric weights

w = np.full((len(Vsym), 25), np.nan)
scl = np.full((len(Vsym), 1), np.nan)
S = np.full((len(Vsym), 25), np.nan)

for t in range(1,len(Vsym)):
    V1 = Vsym[t][:, 0]
    if np.isnan(V1).sum() > 0:
        continue
    S[t, :] = Ststslc[t - 1].T
    S[t, :] = rankstdize(np.atleast_2d(S[t,:]))
    scl[t] = np.dot(np.squeeze(np.array(V1)), S[t, :])
    w[t, :] = np.outer(V1, V1) @ S[t, :]

plt.bar(range(len(labs)), np.nanmean(w.tolist(), axis=0), edgecolor='black')
#plt.bar(range(len(labs)), np.nanmedian(V.tolist(), axis=0), edgecolor='black')
plt.xticks(range(len(labs)), labs, rotation=90)
plt.gca().set_title('Average PEP portfolio weight $S w_1w_1$')
plt.gca().set_axisbelow(True)
plt.gcf().set_size_inches(12.8, 10)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.axhline(y=0, color='black')
plt.savefig(figdir + 'Figure2c.jpg', dpi=300)
plt.close()

datetime_dates = np.array([(str(int(date))) for date in datesno[~np.isnan(scl[:,0])] // 100])
x_ticks = np.arange(0, len(datetime_dates), step=100)
plt.xticks(x_ticks, datetime_dates[x_ticks])
plt.ylim([-1.5, 1.5])
plt.gca().set_title('PEP portfolio scale $S^T w_1$ each period')
plt.gca().set_axisbelow(True)
plt.gcf().set_size_inches(12.8, 10)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.plot(datetime_dates, scl[:,0][~np.isnan(scl[:,0])])
plt.savefig(figdir + 'Figure2b.jpg', dpi=300)
plt.close()


# Plots Asymmetric weights
w = np.full((len(Vsym), 25), np.nan)
sclr = np.full((len(Vsym), 1), np.nan)
scli = np.full((len(Vsym), 1), np.nan)
S = np.full((len(Vsym), 25), np.nan)

for t in range(1,len(Vsym)):
    V1 = Vasym[t][:, 0]
    Vr = np.real(V1)
    Vi = np.imag(V1)
    if np.isnan(V1).sum() > 0:
        continue
    S[t, :] = Ststslc[t - 1].T
    S[t, :] = rankstdize(np.atleast_2d(S[t,:]))
    sclr[t] = np.dot(np.squeeze(np.array(Vr)), S[t, :])
    scli[t] = np.dot(np.squeeze(np.array(Vi)), S[t, :])
    w[t, :] = np.dot(S[t, :], Vr)*Vi - np.dot(S[t, :], Vi)*Vr

plt.bar(range(len(labs)), np.nanmean(w.tolist(), axis=0), edgecolor='black')
#plt.bar(range(len(labs)), np.nanmedian(V.tolist(), axis=0), edgecolor='black')
plt.xticks(range(len(labs)), labs, rotation=90)
plt.gca().set_title('Average PAP portfolio weight $S (yx-xy)$')
plt.gca().set_axisbelow(True)
plt.gcf().set_size_inches(12.8, 10)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.axhline(y=0, color='black')
plt.savefig(figdir + 'Figure2f.jpg', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(12.8,10))
datetime_dates = np.array([(str(int(date))) for date in datesno[~np.isnan(sclr[:,0])] // 100])
x_ticks = np.arange(0, len(datetime_dates), step=100)
plt.xticks(x_ticks, datetime_dates[x_ticks])
plt.ylim([-1.5, 1.5])
plt.gca().set_title('PAP portfolio scale $S^T x$, $S^T y$ each period')
plt.gca().set_axisbelow(True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
legend_handles = [Patch(facecolor=color) for color in [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980)]]
legend_labels = ['Real ($S^T x$)', 'Imaginary ($S^T y$)']
fig.gca().legend(legend_handles, legend_labels, prop={'size': 15})
plt.plot(datetime_dates, sclr[:,0][~np.isnan(sclr[:,0])], scli[:,0][~np.isnan(scli[:,0])])
plt.savefig(figdir + 'Figure2e.jpg', dpi=300)
plt.close()

# End notice
print("End Script")