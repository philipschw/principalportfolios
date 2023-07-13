import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from code_solver.expsmoother import expsmoother
from code_solver.rankstdize import rankstdize
from code_solver.specptfs import specptfs
from code_solver.perform_eval import sharpe, perform_eval

# Choices
minyr = 1963
maxyr = 2019
cntr = 0
tsdm = 0
volstd = 0
mincov = 0
rollwin = 120
Neig = 10

# Load data
dataname = 'HML-Intl-Factors'
figdir = '../figures/'

# Read signals data
data = pd.read_csv('../data/hml_signals_us.csv', delimiter=',')
siglist = data.columns[3:]
ptfid = data.iloc[:, 1].to_numpy()
dates = data.iloc[:, 2]
tmp = np.array([date.split('-') for date in dates], dtype=int)
dates = tmp[:, 0] * 100 + tmp[:, 1]
signals = data.iloc[:, 3:]
idlist = np.unique(ptfid)
datelist = np.unique(dates)

# Read returns data
data = pd.read_csv('../data/hml_us.csv', delimiter=',')
retvwcap = data["ret_vw_cap"].to_numpy()

del data, tmp

# Reshape data
N = len(idlist)
T = len(datelist)
R = np.full((T, N), np.nan)
Scell = [np.full((T, N), np.nan) for _ in range(N)]
for j in range(N):
    loc = np.where(ptfid == idlist[j])[0]
    _, ia, ib = np.intersect1d(datelist, dates[loc], return_indices=True)
    R[ia, j] = retvwcap[loc[ib]]
    for k in range(N):
        Scell[k][ia, j] = signals.iloc[loc, k]

del ptfid, retvwcap, signals, dates, siglist

# Restrict by coverage
if mincov == 1:
    coverage = sio.loadmat('../data/ipcainit_153_coverage.mat')["coverage"].squeeze()
    coveragename = np.genfromtxt("../data/coveragename.csv", delimiter=',', encoding=None, dtype=None)
    loc = np.where(coverage > 0.8)[0]
    covlist = coveragename[loc]
    loc = np.where(np.isin(idlist, covlist))[0]
    Scell = [Scell[i] for i in loc]
    for j in range(len(Scell)):
        Scell[j] = Scell[j][:, loc]
    R = R[:, loc]
    N = R.shape[1]
    idlist = idlist[loc]

    del coverage, coveragename, covlist, loc


# De-volatize
if volstd == 1:
    Rvol = np.full(R.shape, np.nan)
    for j in range(N):
        loc = np.where(~np.isnan(R[:, j]))[0]
        Rvol[loc, j] = np.sqrt(expsmoother(.1, R[loc, j], .1))
    R = np.divide(R, Rvol)

    del Rvol

# Restrict dates
loc = np.where((datelist > minyr * 100) & (datelist < (maxyr + 1) * 100))[0]
datelist = datelist[loc]
Scell = [Scell[i][loc, :] for i in range(len(Scell))]
R = R[loc, :]
T = len(datelist)

# Restrict to factors with non-missing data
loc = np.where(np.sum(np.isnan(R), axis=0) == 0)[0]
R = R[:, loc]
idlist = idlist[loc]
Scell = [Scell[i][:, loc] for i in range(len(Scell))]
N = R.shape[1]

# Labels
labs = idlist

# Load/merge Fama-French factor data for benchmarking
ffdata = sio.loadmat('../data/ff5monthly.mat')
fffac = ffdata["fffac"]
ffdates = ffdata["ffdates"]
fffactmp = np.full((T, 5), np.nan)
_, ia, ib = np.intersect1d(datelist, ffdates, return_indices=True)
fffactmp[ia, :] = fffac[ib, :]
fffac = fffactmp

del ffdata, fffactmp, ffdates

##################
# Recursive PP
SR = np.full((N, 22), np.nan)
PTFS = {}

for sig in range(N):

    # Slice data
    S = Scell[sig]

    Rtrnslc = {}
    Strnslc = {}
    Rtstslc = {}
    Ststslc = {}
    for tau in range(rollwin, T - 1):
        Rtrnslc[tau] = R[tau - rollwin + 1:tau, :]
        Strnslc[tau] = S[tau - rollwin + 1:tau, :]
        Rtstslc[tau] = R[tau + 1, :]
        Ststslc[tau] = S[tau + 1, :]

    Fhis = np.full(T, np.nan)
    Fhks = np.full(T, np.nan)
    Fhksnc = np.full(T, np.nan)
    Ftil = np.full(T, np.nan)
    PP = np.full((T, N), np.nan)
    PEP = np.full((T, N), np.nan)
    PAP = np.full((T, N // 2), np.nan)

    for tau in tqdm(range(rollwin, T - 1)):
        Rtrn = Rtrnslc[tau]
        Strn = Strnslc[tau]
        loc1 = np.where(~np.isnan(np.sum(Rtrn + Strn, axis=1)))[0]
        if len(loc1) < 0.8 * rollwin:
            continue
        Rtrn = Rtrn[loc1, :]
        Strn = Strn[loc1, :]

        Rtst = Rtstslc[tau]
        Stst = Ststslc[tau]

        # Rank-standardize signal
        if cntr == 1:
            Strn = rankstdize(Strn)
            Stst = rankstdize(Stst)

        # Time series de-mean signals (based on training mean)
        if tsdm == 1:
            Strn = Strn - np.nanmean(Strn, axis=0)
            Stst = Stst - np.nanmean(Strn, axis=0)
            Rtrn = Rtrn - np.nanmean(Rtrn, axis=0)
            Rtst = Rtst - np.nanmean(Rtrn, axis=0)

        # Cross-section demean return (training data only)
        if cntr == 1:
            Rtrn = (Rtrn.T - np.nanmean(Rtrn, axis=1)).T
            Rtst = (Rtst.T - np.nanmean(Rtst, axis=1)).T

        Q, D, _ = np.linalg.svd(np.cov(Rtrn, rowvar=False))
        Rhkstrn = Rtrn @ Q
        Rhkstst = Rtst @ Q
        Shkstrn = Strn @ Q
        Shkstst = Stst @ Q
        Phks = np.full(5, np.nan)
        for qq in range(5):
            slope, intercept, r_value, p_value, std_err = stats.linregress(Shkstrn[:, qq], Rhkstrn[:, qq])
            Phks[qq] = np.dot([1, Shkstst[qq]], [intercept, slope])

        Fhkscov = np.cov(Rhkstrn[:, :5], rowvar=False, ddof=0)
        Fhks[tau + 1] = np.dot(Phks, np.linalg.solve(Fhkscov, Rhkstst[0:5]))
        Fhksnc[tau + 1] = np.sum(Rhkstst[:5] * Phks)

        # Baseline factor (sign based on historical mean)
        Ftiltrain = np.sum(Strn * Rtrn, axis=1)
        Ftil[tau + 1] = np.dot(Stst, Rtst) * np.sign(np.nanmean(Ftiltrain, axis=0))

        # Historical mean-weighted factor
        Fhis[tau + 1] = np.nanmean(Rtrn, axis=0) @ Rtst

        # Build portfolios
        W1, W2, Df, Ws, Ds, Wa, Da, PPtmp, PEPtmp, PAPtmp = specptfs(Rtrn * 1000, Strn * 1000, Rtst * 1000, Stst * 1000)
        PP[tau + 1, :] = PPtmp / 1000000
        PEP[tau + 1, :] = PEPtmp / 1000000
        PAP[tau + 1, :] = PAPtmp / 1000000

    PTFS[sig] = {
        "Fhis": Fhis,
        "Ftil": Ftil,
        "Fhks": Fhks,
        "Fhksnc": Fhksnc,
        "PP": PP,
        "PEP": PEP,
        "PAP": PAP
    }
    re = np.hstack([Fhis.reshape(-1,1), Ftil.reshape(-1,1), PP[:, :5], PAP[:, :5], PEP[:, :5], PEP[:, -5:]])
    SR_sig = np.full(22, np.nan)
    for j in range(22):
        SR_sig[j] = sharpe(re[:, j]) * np.sqrt(12)
    SR[sig, :] = SR_sig
    print(f"Iteration {sig+1}/{N}")

# Combined strategies (EW average of 1st PP for each variety)
PP1ALL = np.full((T, N), np.nan)
PEP1ALL = np.full((T, N), np.nan)
PEPNALL = np.full((T, N), np.nan)
PAP1ALL = np.full((T, N), np.nan)
FtilALL = np.full((T, N), np.nan)
FhksALL = np.full((T, N), np.nan)
FhksncALL = np.full((T, N), np.nan)

for i in range(N):
    PP1ALL[:, i] = PTFS[i]["PP"][:, 0]
    PEP1ALL[:, i] = PTFS[i]["PEP"][:, 0]
    PEPNALL[:, i] = PTFS[i]["PEP"][:, -1]
    PAP1ALL[:, i] = PTFS[i]["PAP"][:, 0]
    FtilALL[:, i] = PTFS[i]["Ftil"]
    FhksALL[:, i] = PTFS[i]["Fhks"]
    FhksncALL[:, i] = PTFS[i]["Fhksnc"]

PPew = np.nanmean(PP1ALL, axis=1)
PEP1ew = np.nanmean(PEP1ALL, axis=1)
PEPNew = np.nanmean(PEPNALL, axis=1)
PAPew = np.nanmean(PAP1ALL, axis=1)
Ftilew = np.nanmean(FtilALL, axis=1)
Fhksew = np.nanmean(FhksALL, axis=1)
Fhksncew = np.nanmean(FhksncALL, axis=1)

# Performance summaries
Neig = 10

FtilSR = np.full(N, np.nan)
FtilSRse = np.full(N, np.nan)
FhksSR = np.full(N, np.nan)
FhksSRse = np.full(N, np.nan)
FhksncSR = np.full(N, np.nan)
FhksncSRse = np.full(N, np.nan)
FhksIR = np.full(N, np.nan)
FhksIRse = np.full(N, np.nan)
FhksncIR = np.full(N, np.nan)
FhksncIRse = np.full(N, np.nan)
PPSR = np.full((N, N), np.nan)
PAPSR = np.full((N, N//2), np.nan)
PEPSR = np.full((N, N), np.nan)
PPSRse = np.full((N, N), np.nan)
PAPSRse = np.full((N, N//2), np.nan)
PEPSRse = np.full((N, N), np.nan)
PPIR = np.full((N, N), np.nan)
PAPIR = np.full((N, N//2), np.nan)
PEPIR = np.full((N, N), np.nan)
PPIRse = np.full((N, N), np.nan)
PAPIRse = np.full((N, N//2), np.nan)
PEPIRse = np.full((N, N), np.nan)

for sig in range(N):
    TT = np.sum(~np.isnan(PTFS[0]["PP"][:, 0]))

    SR, SRse, IR, IRse = perform_eval(PTFS[sig]["Ftil"], [], 1 / 12)
    FtilSR[sig] = SR
    FtilSRse[sig] = SRse

    SR, SRse, IR, IRse = perform_eval(PTFS[sig]["Fhks"], np.hstack([PTFS[sig]["Ftil"].reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
    FhksSR[sig] = SR
    FhksSRse[sig] = SRse
    FhksIR[sig] = IR
    FhksIRse[sig] = IRse

    SR, SRse, IR, IRse = perform_eval(PTFS[sig]["Fhksnc"], np.hstack([PTFS[sig]["Ftil"].reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
    FhksncSR[sig] = SR
    FhksncSRse[sig] = SRse
    FhksncIR[sig] = IR
    FhksncIRse[sig] = IRse

    SR, SRse, IR, IRse = perform_eval(PTFS[sig]["PP"], np.hstack([PTFS[sig]["Ftil"].reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
    PPSR[sig, :] = SR.squeeze()
    PPSRse[sig, :] = SRse.squeeze()
    PPIR[sig, :] = IR.squeeze()
    PPIRse[sig, :] = IRse.squeeze()

    SR, SRse, IR, IRse = perform_eval(PTFS[sig]["PEP"], np.hstack([PTFS[sig]["Ftil"].reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
    PEPSR[sig, :] = SR.squeeze()
    PEPSRse[sig, :] = SRse.squeeze()
    PEPIR[sig, :] = IR.squeeze()
    PEPIRse[sig, :] = IRse.squeeze()

    SR, SRse, IR, IRse = perform_eval(PTFS[sig]["PAP"], np.hstack([PTFS[sig]["Ftil"].reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
    PAPSR[sig, :] = SR.squeeze()
    PAPSRse[sig, :] = SRse.squeeze()
    PAPIR[sig, :] = IR.squeeze()
    PAPIRse[sig, :] = IRse.squeeze()

SR, SRse, IR, IRse = perform_eval(PTFS[0]["Fhis"], [], 1 / 12)
FhisSR = SR
FhisSRse = SRse

SR, SRse, IR, IRse = perform_eval(Ftilew, [], 1 / 12)
FtilewSR = SR
FtilewSRse = SRse

SR, SRse, IR, IRse = perform_eval(Fhksew, np.hstack([Ftilew.reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
FhksewSR = SR
FhksewSRse = SRse
FhksewIR = IR
FhksewIRse = IRse

SR, SRse, IR, IRse = perform_eval(Fhksncew, np.hstack([Ftilew.reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
FhksncewSR = SR
FhksncewSRse = SRse
FhksncewIR = IR
FhksncewIRse = IRse

SR, SRse, IR, IRse = perform_eval(PPew, np.hstack([Ftilew.reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
PPewSR = SR
PPewSRse = SRse
PPewIR = IR
PPewIRse = IRse

SR, SRse, IR, IRse = perform_eval(PEP1ew, np.hstack([Ftilew.reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
PEP1ewSR = SR
PEP1ewSRse = SRse
PEP1ewIR = IR
PEP1ewIRse = IRse

SR, SRse, IR, IRse = perform_eval(PEPNew, np.hstack([Ftilew.reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
PEPNewSR = SR
PEPNewSRse = SRse
PEPNewIR = IR
PEPNewIRse = IRse

SR, SRse, IR, IRse = perform_eval(PAPew, np.hstack([Ftilew.reshape(-1,1), PTFS[0]["Fhis"].reshape(-1,1), fffac]), 1 / 12)
PAPewSR = SR
PAPewSRse = SRse
PAPewIR = IR
PAPewIRse = IRse

# Figure 4: Plots Average Sharpe ratios
yrng = [np.nanmean(PEPSR[:, -1], axis=0) - 2 * np.nanmean(PEPSRse[:, 0], axis=0), np.nanmean(PAPSR[:, 0], axis=0) + 2 * np.nanmean(PAPSRse[:, 0], axis=0)]

# Plot PP
bardata = np.hstack([np.nanmean(PPSR[:, 0:Neig], axis=0).reshape(-1,1), np.nanmean(PPIR[:, 0:Neig], axis=0).reshape(-1,1)]).T
barse = np.hstack([np.nanmean(PPSRse[:, 0:Neig], axis=0).reshape(-1,1), np.nanmean(PPIRse[:, 0:Neig], axis=0).reshape(-1,1)]).T
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_ylim(yrng)
for i in range(bardata.shape[0]):
    ax.bar(np.arange(Neig) + i * 0.35, bardata[i], 0.35, yerr=barse[i], capsize=10)
plt.xticks(np.arange(Neig) + 0.35 / 2, np.arange(Neig))
plt.legend(['Sharpe Ratio', 'Information Ratio'])
plt.savefig(figdir + 'Figure4a.jpg')
plt.close()

# Plot PEP
x = list(range(Neig//2)) + list(range(PEPSR.shape[1] - Neig//2, PEPSR.shape[1]))
bardata = np.hstack([np.nanmean(PEPSR[:, x], axis=0).reshape(-1,1), np.nanmean(PEPIR[:, x], axis=0).reshape(-1,1)]).T
barse = np.hstack([np.nanmean(PEPSRse[:, x], axis=0).reshape(-1,1), np.nanmean(PEPIRse[:, x], axis=0).reshape(-1,1)]).T
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_ylim(yrng)
for i in range(bardata.shape[0]):
    ax.bar(np.arange(Neig) + i * 0.35, bardata[i], 0.35, yerr=barse[i], capsize=10)
plt.xticks(np.arange(Neig) + 0.35 / 2, [str(l) for l in x])
plt.legend(['Sharpe Ratio', 'Information Ratio'])
plt.savefig(figdir + 'Figure4b.jpg')
plt.close()

# Plot PAP
bardata = np.hstack([np.nanmean(PAPSR[:, 0:Neig], axis=0).reshape(-1,1), np.nanmean(PAPIR[:, 0:Neig], axis=0).reshape(-1,1)]).T
barse = np.hstack([np.nanmean(PAPSRse[:, 0:Neig], axis=0).reshape(-1,1), np.nanmean(PAPIRse[:, 0:Neig], axis=0).reshape(-1,1)]).T
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_ylim(yrng)
for i in range(bardata.shape[0]):
    ax.bar(np.arange(Neig) + i * 0.35, bardata[i], 0.35, yerr=barse[i], capsize=10)
plt.xticks(np.arange(Neig) + 0.35 / 2, np.arange(Neig))
plt.legend(['Sharpe Ratio', 'Information Ratio'])
plt.savefig(figdir + 'Figure4c.jpg')
plt.close()

# Plot Factors
bardata = [np.nanmean(FtilSR, axis=0), FhisSR[0][0]]
barse = [np.nanmean(FtilSRse, axis=0), FhisSRse[0][0]]
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_ylim(yrng)
ax.bar(np.arange(2), bardata, yerr=barse, capsize=10)
plt.xticks(np.arange(2), ['Factor', 'Hist. Mean Wts.'])
plt.legend(['Sharpe Ratio'])
plt.savefig(figdir + 'Figure4d.jpg')
plt.close()

# Figure 5: Comparison with HKS
bardata = np.hstack([PPewSR, PEP1ewSR, PEPNewSR, PAPewSR, FhksewSR, FhksncewSR])
bardata = np.vstack([bardata, np.hstack([PPewIR, PEP1ewIR, PEPNewIR, PAPewIR, FhksewIR, FhksncewIR])])
barse = np.hstack([PPewSRse, PEP1ewSRse, PEPNewSRse, PAPewSRse, FhksewSRse, FhksncewSRse])
barse = np.vstack([barse, np.hstack([PPewIRse, PEP1ewIRse, PEPNewIRse, PAPewIRse, FhksewIRse, FhksncewIRse])])
fig, ax = plt.subplots(figsize=(12, 8))
for i in range(bardata.shape[0]):
    ax.bar(np.arange(6) + i * 0.35, bardata[i], 0.35, yerr=barse[i], capsize=10)
plt.xticks(np.arange(6) + 0.35 / 2, ['PP', 'PEP 1', 'PEP N', 'PAP', 'HKS', 'HKS (No Cov.)'])
plt.legend(['Sharpe Ratio', 'Information Ratio'])
plt.savefig(figdir + 'Figure5.jpg')
plt.close()
