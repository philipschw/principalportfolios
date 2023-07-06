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
figdir = './Figures/'

# Read signals data
with open('./Data/hml_signals_us.csv') as fid:
    col = fid.readline()
    ncol = 1 + len(col.split(',')) - 3
    siglist = col.split(',')[3:]
    data = np.genfromtxt(fid, delimiter=',', skip_header=1, dtype=None, encoding=None)
ptfid = data[:, 0].astype(str)
dates = data[:, 1].astype(str)
tmp = np.array([date.split('/') for date in dates], dtype=int)
dates = tmp[:, 0] * 100 + tmp[:, 1]
signals = data[:, 3:].astype(float)
idlist = np.unique(ptfid)
datelist = np.unique(dates)

# Read returns data
with open('./Data/hml_us.csv') as fid:
    data = np.genfromtxt(fid, delimiter=',', skip_header=1, dtype=None, encoding=None)
retvwcap = data[:, -1].astype(float)

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
        Scell[k][ia, j] = signals[loc[ib], k]

ptfid = None
retvwcap = None
signals = None
dates = None

# Restrict by coverage
if mincov == 1:
    coverage = np.load('./Data/ipcainit_153_coverage.npy')
    srtcov = np.sort(coverage)[::-1]
    loc = np.where(coverage > 0.8)[0]
    covlist = coveragename[loc]
    loc = np.where(np.isin(idlist, covlist))[0]
    Scell = [Scell[i] for i in loc]
    for j in range(len(Scell)):
        Scell[j] = Scell[j][:, loc]
    R = R[:, loc]
    N = R.shape[1]
    idlist = idlist[loc]
    srtcov = None
    coverage = None
    coveragename = None

# De-volatize
if volstd == 1:
    Rvol = np.full(R.shape, np.nan)
    for j in range(N):
        loc = np.where(~np.isnan(R[:, j]))[0]
        Rvol[loc, j] = np.sqrt(expsmoother(.1, R[loc, j], .1))
    R = R / Rvol
    Rvol = None

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
fffac = np.load('./Data/ff5monthly.npy')
fffac = fffac[np.isin(datelist, fffac[:, 0]), 1:]

# Recursive PP
SR = np.full((N, 22), np.nan)
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
    PAP = np.full((T, int(N / 2)), np.nan)
    for tau in range(rollwin + 1, T - 1):
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
            Strn = Strn - np.nanmean(Strn)
            Stst = Stst - np.nanmean(Strn)
            Rtrn = Rtrn - np.nanmean(Rtrn)
            Rtst = Rtst - np.nanmean(Rtrn)

        # Cross-section demean return (training data only)
        if cntr == 1:
            Rtrn = Rtrn - np.nanmean(Rtrn, axis=1)[:, np.newaxis]
            Rtst = Rtst - np.nanmean(Rtst, axis=1)[:, np.newaxis]

        Q, D, _ = np.linalg.svd(np.cov(Rtrn, rowvar=False))
        Rhkstrn = Rtrn @ Q
        Rhkstst = Rtst @ Q
        Shkstrn = Strn @ Q
        Shkstst = Stst @ Q
        Phks = np.full(5, np.nan)
        for qq in range(5):
            stats = sm.OLS(Rhkstrn[:, qq], Shkstrn[:, qq]).fit()
            Phks[qq] = stats.params[1]

        Fhkscov = np.cov(Rhkstrn[:, :5], rowvar=False, ddof=0)
        Fhks[tau + 1] = Phks @ np.linalg.inv(Fhkscov) @ Rhkstst[:5]
        Fhksnc[tau + 1] = np.sum(Rhkstst[:5] * Phks)

        Ftiltrain = np.sum(Strn * Rtrn, axis=1)
        Ftil[tau + 1] = np.dot(Stst, Rtst) * np.sign(np.nanmean(Ftiltrain))

        Fhis[tau + 1] = np.nanmean(Rtrn) @ Rtst

        W1, W2, _, _, _, _, _, PPtmp, PEPtmp, PAPtmp = specptfs(Rtrn * 1000, Strn * 1000, Rtst * 1000, Stst * 1000)
        PP[tau + 1, :] = PPtmp / 1000000
        PEP[tau + 1, :] = PEPtmp / 1000000
        PAP[tau + 1, :] = PAPtmp / 1000000

    PTFS[sig, 0].Fhis = Fhis
    PTFS[sig, 0].Ftil = Ftil
    PTFS[sig, 0].Fhks = Fhks
    PTFS[sig, 0].Fhksnc = Fhksnc
    PTFS[sig, 0].PP = PP
    PTFS[sig, 0].PEP = PEP
    PTFS[sig, 0].PAP = PAP
    SR[sig, :] = sharpe([Fhis, Ftil, PP[:, 0:5], PAP[:, 0:5], PEP[:, 0:5], PEP[:, -5:]]) * np.sqrt(12)
    print(sig)

# Combined strategies (EW average of 1st PP for each variety)
PP1ALL = np.full((T, N), np.nan)
PEP1ALL = np.full((T, N), np.nan)
PEPNALL = np.full((T, N), np.nan)
PAP1ALL = np.full((T, N), np.nan)
FtilALL = np.full((T, N), np.nan)
FhksALL = np.full((T, N), np.nan)
FhksncALL = np.full((T, N), np.nan)
for i in range(N):
    PP1ALL[:, i] = PTFS[i].PP[:, 0]
    PEP1ALL[:, i] = PTFS[i].PEP[:, 0]
    PEPNALL[:, i] = PTFS[i].PEP[:, -1]
    PAP1ALL[:, i] = PTFS[i].PAP[:, 0]
    FtilALL[:, i] = PTFS[i].Ftil[:, 0]
    FhksALL[:, i] = PTFS[i].Fhks[:, 0]
    FhksncALL[:, i] = PTFS[i].Fhksnc[:, 0]

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
PPSR = np.full((N, N), np.nan)
PAPSR = np.full((N, int(N / 2)), np.nan)
PEPSR = np.full((N, N), np.nan)
PPSRse = np.full((N, N), np.nan)
PAPSRse = np.full((N, int(N / 2)), np.nan)
PEPSRse = np.full((N, N), np.nan)
PPIR = np.full((N, N), np.nan)
PAPIR = np.full((N, int(N / 2)), np.nan)
PEPIR = np.full((N, N), np.nan)
PPIRse = np.full((N, N), np.nan)
PAPIRse = np.full((N, int(N / 2)), np.nan)
PEPIRse = np.full((N, N), np.nan)
for sig in range(N):
    TT = np.sum(~np.isnan(PTFS[0, 0].PP[:, 0]))

    SR, SRse, IR, IRse = perform_eval(PTFS[sig, 0].Ftil, None, 1 / 12)
    FtilSR[sig, :] = SR
    FtilSRse[sig, :] = SRse

    SR, SRse, IR, IRse = perform_eval(PTFS[sig, 0].Fhks, [PTFS[sig, 0].Ftil, PTFS[0, 0].Fhis, fffac], 1 / 12)
    FhksSR[sig, :] = SR
    FhksSRse[sig, :] = SRse
    FhksIR[sig, :] = IR
    FhksIRse[sig, :] = IRse

    SR, SRse, IR, IRse = perform_eval(PTFS[sig, 0].Fhksnc, [PTFS[sig, 0].Ftil, PTFS[0, 0].Fhis, fffac], 1 / 12)
    FhksncSR[sig, :] = SR
    FhksncSRse[sig, :] = SRse
    FhksncIR[sig, :] = IR
    FhksncIRse[sig, :] = IRse

    SR, SRse, IR, IRse = perform_eval(PTFS[sig, 0].PP, [PTFS[sig, 0].Ftil, PTFS[0, 0].Fhis, fffac], 1 / 12)
    PPSR[sig, :] = SR
    PPSRse[sig, :] = SRse
    PPIR[sig, :] = IR
    PPIRse[sig, :] = IRse

    SR, SRse, IR, IRse = perform_eval(PTFS[sig, 0].PEP, [PTFS[sig, 0].Ftil, PTFS[0, 0].Fhis, fffac], 1 / 12)
    PEPSR[sig, :] = SR
    PEPSRse[sig, :] = SRse
    PEPIR[sig, :] = IR
    PEPIRse[sig, :] = IRse

    SR, SRse, IR, IRse = perform_eval(PTFS[sig, 0].PAP, [PTFS[sig, 0].Ftil, PTFS[0, 0].Fhis, fffac], 1 / 12)
    PAPSR[sig, :] = SR
    PAPSRse[sig, :] = SRse
    PAPIR[sig, :] = IR
    PAPIRse[sig, :] = IRse

SR, SRse, IR, IRse = perform_eval(PTFS[0, 0].Fhis, None, 1 / 12)
FhisSR = SR
FhisSRse = SRse

SR, SRse, IR, IRse = perform_eval(Ftilew, None, 1 / 12)
FtilewSR = SR
FtilewSRse = SRse

SR, SRse, IR, IRse = perform_eval(Fhksew, [Ftilew, PTFS[0, 0].Fhis, fffac], 1 / 12)
FhksewSR = SR
FhksewSRse = SRse
FhksewIR = IR
FhksewIRse = IRse

SR, SRse, IR, IRse = perform_eval(Fhksncew, [Ftilew, PTFS[0, 0].Fhis, fffac], 1 / 12)
FhksncewSR = SR
FhksncewSRse = SRse
FhksncewIR = IR
FhksncewIRse = IRse

SR, SRse, IR, IRse = perform_eval(PPew, [Ftilew, PTFS[0, 0].Fhis, fffac], 1 / 12)
PPewSR = SR
PPewSRse = SRse
PPewIR = IR
PPewIRse = IRse

SR, SRse, IR, IRse = perform_eval(PEP1ew, [Ftilew, PTFS[0, 0].Fhis, fffac], 1 / 12)
PEP1ewSR = SR
PEP1ewSRse = SRse
PEP1ewIR = IR
PEP1ewIRse = IRse

SR, SRse, IR, IRse = perform_eval(PEPNew, [Ftilew, PTFS[0, 0].Fhis, fffac], 1 / 12)
PEPNewSR = SR
PEPNewSRse = SRse
PEPNewIR = IR
PEPNewIRse = IRse

SR, SRse, IR, IRse = perform_eval(PAPew, [Ftilew, PTFS[0, 0].Fhis, fffac], 1 / 12)
PAPewSR = SR
PAPewSRse = SRse
PAPewIR = IR
PAPewIRse = IRse

# Figure 4: Plots Average Sharpe ratios
yrng = [np.nanmean(PEPSR[:, -1]) - 2 * np.nanmean(PEPSRse[:, 0]), np.nanmean(PAPSR[:, 0]) + 2 * np.nanmean(PAPSRse[:, 0])]

# Plot PP
bardata = [np.nanmean(PPSR[:, 0:Neig]), np.nanmean(PPIR[:, 0:Neig])].T
barse = [np.nanmean(PPSRse[:, 0:Neig]), np.nanmean(PPIRse[:, 0:Neig])].T
b = errorbargrouped(bardata, barse, 2, 10)
plt.legend(['Sharpe Ratio', 'Information Ratio'])
plt.gca().set(xticklabels=[1, 2, 3, 4, 5, 96, 97, 98, 99, 100], fontname='Times New Roman', fontsize=20, ylim=yrng)
plt.gcf().set_size_inches(1280, 1000)
plt.savefig(figdir + 'Figure4a.eps', format='eps2c')
plt.close()

# Plot PEP
bardata = [np.nanmean(PEPSR[:, [0:Neig//2, -Neig//2+1:]]), np.nanmean(PEPIR[:, [0:Neig//2, -Neig//2+1:]])].T
barse = [np.nanmean(PEPSRse[:, [0:Neig//2, -Neig//2+1:]]), np.nanmean(PEPIRse[:, [0:Neig//2, -Neig//2+1:]])].T
b = errorbargrouped(bardata, barse, 2, 10)
plt.legend(['Sharpe Ratio', 'Information Ratio'])
plt.gca().set(xticklabels=list(range(1, 6)) + list(range(96, 101)), fontname='Times New Roman', fontsize=20, ylim=yrng)
plt.gcf().set_size_inches(1280, 1000)
plt.savefig(figdir + 'Figure4b.eps', format='eps2c')
plt.close()

# Plot PAP
bardata = [np.nanmean(PAPSR[:, 0:Neig]), np.nanmean(PAPIR[:, 0:Neig])].T
barse = [np.nanmean(PAPSRse[:, 0:Neig]), np.nanmean(PAPIRse[:, 0:Neig])].T
b = errorbargrouped(bardata, barse, 2, 10)
plt.legend(['Sharpe Ratio', 'Information Ratio'])
plt.gca().set(fontname='Times New Roman', fontsize=20, ylim=yrng)
plt.gcf().set_size_inches(1280, 1000)
plt.savefig(figdir + 'Figure4c.eps', format='eps2c')
plt.close()

# Plot Factors
bardata = [np.nanmean(FtilSR), FhisSR[0]]
barse = [np.nanmean(FtilSRse), FhisSRse[0]]
b = errorbargrouped(bardata, barse, 2, 15)
plt.gca().set(xticklabels=['Factor', 'Hist. Mean Wts.'])
plt.legend(['Sharpe Ratio'])
plt.gca().set(fontname='Times New Roman', fontsize=20, ylim=yrng)
plt.gcf().set_size_inches(1280, 1000)
plt.savefig(figdir + 'Figure4d.eps', format='eps2c')
plt.close()

# Figure 5: Comparison with HKS
bardata = [PPewSR, PEP1ewSR, PEPNewSR, PAPewSR, FhksewSR, FhksncewSR]
barse = [PPewSRse, PEP1ewSRse, PEPNewSRse, PAPewSRse, FhksewSRse, FhksncewSRse]
b = errorbargrouped(bardata, barse, 2, 20)
plt.legend(['Sharpe Ratio', 'Information Ratio'])
plt.gca().set(xticklabels=['PP', 'PEP 1', 'PEP N', 'PAP', 'HKS', 'HKS (No Cov.)'])
plt.gca().set(fontname='Times New Roman', fontsize=20, ylim=[-1.5, 1.75])
plt.gcf().set_size_inches(1280, 1000)
plt.savefig(figdir + 'Figure5.eps', format='eps2c')
plt.close()

# Table 3: Ex post tangency portfolio
XXXraw = np.concatenate([fffac, PTFS[0].Fhis, Ftilew, PPew, PEP1ew, -PEPNew, PAPew], axis=1)
XXX = (.1 / np.sqrt(12)) * XXXraw / np.nanstd(XXXraw, axis=0)
tpff, twff, _, twtff = tanptf(XXX[:, 0:5])
tp, tw, twc, twt = tanptf(XXX)
tpns, twns = tanptfnoshort(XXX)
out = np.concatenate([twff.T, np.full((1, 6), np.nan), sharpe(tpff) * np.sqrt(12)], axis=1)
out = np.concatenate([out, np.concatenate([tw.T, sharpe(tp) * np.sqrt(12)], axis=1)], axis=0)
out = np.concatenate([out, np.concatenate([twns.T, sharpe(tpns) * np.sqrt(12)], axis=1)], axis=0)
np.save(figdir + 'table3.npy', out)
