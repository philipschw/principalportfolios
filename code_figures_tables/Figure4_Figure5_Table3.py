# This script creates Figure 4, Figure 5 and Table 3.

# Start notice
print("Start Script")

# import packages
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from matplotlib.patches import Patch
from tabulate import tabulate

# add path
sys.path.append('../code_solver')

# import self-written auxiliary functions
from expsmoother import expsmoother
from rankstdize import rankstdize
from specptfs import specptfs
from perform_eval import sharpe, perform_eval
from errorbargrouped import errorbargrouped
from linreg import linreg

# turn off certain warnings
np.warnings.filterwarnings('ignore', category=np.ComplexWarning)
np.warnings.filterwarnings('ignore', category=RuntimeWarning)

# Choices
minyr = 1963
maxyr = 2019
cntr = 0
tsdm = 0
volstd = 0
mincov = 0
rollwin = 120
Neig = 10
savedata = 0
loaddata = 0

# Load data
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
Scell = [Scell[j] for j in loc]
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

print("Build Principal Portfolios")
for sig in tqdm(range(N), desc = "Loop over Signals"):

    # Slice data
    S = Scell[sig]

    Rtrnslc = [[] for _ in range(rollwin-1)]
    Strnslc = [[] for _ in range(rollwin-1)]
    Rtstslc = [[] for _ in range(rollwin-1)]
    Ststslc = [[] for _ in range(rollwin-1)]

    for tau in range(rollwin, T):
        Rtrnslc.append(R[tau - rollwin:tau, :])
        Strnslc.append(S[tau - rollwin:tau, :])
        Rtstslc.append(R[tau, :])
        Ststslc.append(S[tau, :])

    Fhis = np.full(T, np.nan)
    Fhks = np.full(T, np.nan)
    Fhksnc = np.full(T, np.nan)
    Ftil = np.full(T, np.nan)
    PP = np.full((T, N), np.nan)
    PEP = np.full((T, N), np.nan)
    PAP = np.full((T, N // 2), np.nan)

    for tau in tqdm(range(rollwin, T-1), desc = "Loop over Time", leave=False):

        # Carve out training data
        Rtrn = Rtrnslc[tau]
        Strn = Strnslc[tau]
        loc1 = np.where(~np.isnan(np.sum(Rtrn + Strn, axis=1)))[0]
        if len(loc1) < 0.8 * rollwin:
            continue
        Rtrn = Rtrn[loc1, :]
        Strn = Strn[loc1, :]

        # Carve out test data
        Rtst = Rtstslc[tau]
        Stst = Ststslc[tau]

        # Rank-standardize signal
        if cntr == 1:
            Strn = rankstdize(Strn)
            Stst = rankstdize(np.atleast_2d(Stst))[0]

        # Time series de-mean signals (based on training mean)
        if tsdm == 1:
            Strn = Strn - np.nanmean(Strn, axis=0)
            Stst = Stst - np.nanmean(Strn, axis=0)
            Rtrn = Rtrn - np.nanmean(Rtrn, axis=0)
            Rtst = Rtst - np.nanmean(Rtrn, axis=0)

        # Cross-section demean return (training data only)
        if cntr == 1:
            Rtrn = Rtrn - np.nanmean(Rtrn, axis=1)[:, np.newaxis]
            Rtst = Rtst - np.nanmean(Rtst)

        Q, D, _ = np.linalg.svd(np.cov(Rtrn, rowvar=False))
        Rhkstrn = Rtrn @ Q
        Rhkstst = Rtst @ Q
        Shkstrn = Strn @ Q
        Shkstst = Stst @ Q
        Phks = np.full(5, np.nan)
        for qq in range(5):
            slope, intercept, r_value, p_value, std_err = stats.linregress(Shkstrn[:, qq], Rhkstrn[:, qq])
            Phks[qq] = np.dot([1, Shkstst[qq]], [intercept, slope])

        Fhkscov = np.cov(Rhkstrn[:, :5], rowvar=False, ddof=1)
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

# Save results to file
if savedata == 1:
    np.savez("../data/Results/Figure4_Figure5_Table3.npz", PTFS=PTFS, SR=SR)

#tmp: load data
if loaddata == 1:
    loaded_data = np.load("../data/Results/Figure4_Figure5_Table3.npz", allow_pickle=True)
    PTFS = loaded_data['PTFS'].item()
    SR = loaded_data['SR']
    loaded_data.close()

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

print("Calculate Performance summaries")
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

for sig in tqdm(range(N), desc='Calculate Sharpe Ratios and Information Ratios'):
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

print("Build Figure 4")
# Figure 4: Plots Average Sharpe ratios

# Plot PP
## Prepare data
bardata = np.hstack([np.nanmean(PPSR[:, 0:Neig], axis=0).reshape(-1,1), np.nanmean(PPIR[:, 0:Neig], axis=0).reshape(-1,1)])
barse = np.hstack([np.nanmean(PPSRse[:, 0:Neig], axis=0).reshape(-1,1), np.nanmean(PPIRse[:, 0:Neig], axis=0).reshape(-1,1)])

## Plot bars
b = errorbargrouped(bardata, barse, 2, True)

## Set ticks
plt.xticks(np.arange(Neig), np.arange(Neig)+1)

## Set font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

## Add legend
bar_colors =[
    (0, 0.4470, 0.7410),   # Blue
    (0.8500, 0.3250, 0.0980),   # Orange
]
legend_handles = [Patch(facecolor=color) for color in bar_colors]
legend_labels = ['Sharpe Ratio', 'Information Ratio']
b.gca().legend(legend_handles, legend_labels, prop={'size': 15})

## Save the figure
b.savefig(figdir + 'Figure4a.jpg', dpi = 300)

## Close the plot
plt.close(b)


# Plot PEP
## Prepare data
x = list(range(Neig//2)) + list(range(PEPSR.shape[1] - Neig//2, PEPSR.shape[1]))
bardata = np.hstack([np.nanmean(PEPSR[:, x], axis=0).reshape(-1,1), np.nanmean(PEPIR[:, x], axis=0).reshape(-1,1)])
barse = np.hstack([np.nanmean(PEPSRse[:, x], axis=0).reshape(-1,1), np.nanmean(PEPIRse[:, x], axis=0).reshape(-1,1)])

## Plot bars
b = errorbargrouped(bardata, barse, 2, True)

## Set ticks
plt.xticks(np.arange(Neig), [str(l) for l in x])

## Set font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

## Add legend
bar_colors =[
    (0, 0.4470, 0.7410),   # Blue
    (0.8500, 0.3250, 0.0980),   # Orange
]
legend_handles = [Patch(facecolor=color) for color in bar_colors]
legend_labels = ['Sharpe Ratio', 'Information Ratio']
b.gca().legend(legend_handles, legend_labels, prop={'size': 15})

## Save the figure
b.savefig(figdir + 'Figure4b.jpg', dpi = 300)

## Close the plot
plt.close(b)


# Plot PAP
## Prepare data
bardata = np.hstack([np.nanmean(PAPSR[:, 0:Neig], axis=0).reshape(-1,1), np.nanmean(PAPIR[:, 0:Neig], axis=0).reshape(-1,1)])
barse = np.hstack([np.nanmean(PAPSRse[:, 0:Neig], axis=0).reshape(-1,1), np.nanmean(PAPIRse[:, 0:Neig], axis=0).reshape(-1,1)])

## Plot bars
b = errorbargrouped(bardata, barse, 2, True)

## Set ticks
plt.xticks(np.arange(Neig), np.arange(Neig)+1)

## Set font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

## Add legend
bar_colors =[
    (0, 0.4470, 0.7410),   # Blue
    (0.8500, 0.3250, 0.0980),   # Orange
]
legend_handles = [Patch(facecolor=color) for color in bar_colors]
legend_labels = ['Sharpe Ratio', 'Information Ratio']
b.gca().legend(legend_handles, legend_labels, prop={'size': 15})

## Save the figure
b.savefig(figdir + 'Figure4c.jpg', dpi = 300)

## Close the plot
plt.close(b)


# Plot Factors
## Prepare data
bardata = np.matrix([np.nanmean(FtilSR, axis=0), FhisSR[0][0]]).T
barse = np.matrix([np.nanmean(FtilSRse, axis=0), FhisSRse[0][0]]).T

## Plot bars
b = errorbargrouped(bardata, barse, 2, True)

## Set ticks
plt.xticks([0,1], ['Factor', 'Hist. Mean Wts.'])

## Set font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

## Add legend
bar_colors =[
    (0, 0.4470, 0.7410),   # Blue
]
legend_handles = [Patch(facecolor=color) for color in bar_colors]
legend_labels = ['Sharpe Ratio']
b.gca().legend(legend_handles, legend_labels, prop={'size': 15})

## Save the figure
b.savefig(figdir + 'Figure4d.jpg', dpi = 300)

## Close the plot
plt.close(b)


print("Build Figure 5")
# Figure 5: Comparison with HKS
## Prepare data
bardata = np.hstack([PPewSR, PEP1ewSR, PEPNewSR, PAPewSR, FhksewSR, FhksncewSR])
bardata = np.vstack([bardata, np.hstack([PPewIR, PEP1ewIR, PEPNewIR, PAPewIR, FhksewIR, FhksncewIR])]).T
barse = np.hstack([PPewSRse, PEP1ewSRse, PEPNewSRse, PAPewSRse, FhksewSRse, FhksncewSRse])
barse = np.vstack([barse, np.hstack([PPewIRse, PEP1ewIRse, PEPNewIRse, PAPewIRse, FhksewIRse, FhksncewIRse])]).T

## Plot bars
b = errorbargrouped(bardata, barse, 2, True)

## Set ticks
plt.xticks(range(6), ['PP', 'PEP 1', 'PEP N', 'PAP', 'HKS', 'HKS (No Cov.)'])

## Set font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

## Add legend
bar_colors =[
    (0, 0.4470, 0.7410),   # Blue
    (0.8500, 0.3250, 0.0980),   # Orange
]
legend_handles = [Patch(facecolor=color) for color in bar_colors]
legend_labels = ['Sharpe Ratio', 'Information Ratio']
b.gca().legend(legend_handles, legend_labels, prop={'size': 15})

## Save the figure
b.savefig(figdir + 'Figure5.jpg', dpi = 300)

## Close the plot
plt.close(b)


print("Build Table 3")
# Table 3: Ex Post Tangency Portfolios
## Prepare data
XXXraw = np.hstack([fffac, PTFS[0]["Fhis"].reshape(-1,1), Ftilew.reshape(-1, 1), PPew.reshape(-1, 1), PEP1ew.reshape(-1, 1), -PEPNew.reshape(-1, 1), PAPew.reshape(-1, 1)])
XXX = (0.1 / np.sqrt(12)) * XXXraw / np.nanstd(XXXraw, axis=0, ddof=1)

## compute tangency portfolios by Britten-Jones (1999) method (see Theorem 1, Corollary 1)
### FF5
bff, twtff, _ = linreg(y=np.ones(XXX.shape[0]), X=XXX[:,:5], intcpt=False)
twff = bff / np.sum(bff) # tangency portfolio weights
tpff = XXX[:,:5] @ twff # tangency portfolio excess return
tpffSR = (np.nanmean(tpff) / np.nanstd(tpff, ddof=1))*np.sqrt(12) # annualized SR

### FF5 + PP
b, twt, _ = linreg(y=np.ones(XXX.shape[0]), X=XXX, intcpt=False)
tw = b / np.sum(b) # tangency portfolio weights
tp = XXX @ tw # tangency portfolio excess return
tpSR = (np.nanmean(tp) / np.nanstd(tp, ddof=1))*np.sqrt(12) # annualized SR

### Nonnegative FF5 + PP
bns, twtns, _ = linreg(y=np.ones(XXX.shape[0]), X=XXX, intcpt=False, nnconstraint=True)
twns = bns / np.sum(bns) # tangency portfolio weights
tpns = XXX @ twns # tangency portfolio excess return
tpnsSR = (np.nanmean(tpns) / np.nanstd(tpns, ddof=1))*np.sqrt(12) # annualized SR

out = np.vstack((
    np.hstack((twff, np.full(6, np.nan), tpffSR)),
    np.hstack((tw, tpSR)),
    np.hstack((twns, tpnsSR))
)).T

out_tstat = np.vstack((
    np.hstack((twtff, np.full(6, np.nan), tpffSR)),
    np.hstack((twt, tpSR)),
    np.hstack((twtns, tpnsSR))
)).T

labs = ['Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA', 'Simple Factor', 'Hist. Mean Wts.', 'PP', 'PEP 1', '-1 x PEP N', 'PAP', 'Sharpe Ratio']
tvalue =  np.vectorize(lambda x: f'{x:.2f}')(out)
for j in range(tvalue.shape[0]-1):
    for k in range(tvalue.shape[1]):
        if np.abs(out_tstat[j,k]) >= 2.58:
            tvalue[j,k] = tvalue[j,k] + '*'

tabout = np.vstack((['Portfolio', 'FF5', 'FF5 + PP', 'Nonnegative FF5 + PP'], np.column_stack((labs, tvalue))))
np.save(figdir + 'table3.npy', tabout)

# Print the LaTeX table
print(tabulate(tabout, headers='firstrow', tablefmt='latex'))

# end notice
print("End Script")
