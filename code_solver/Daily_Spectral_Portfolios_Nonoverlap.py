
# import packages
import numpy as np
import scipy.io as sio
from typing import List
from tqdm import tqdm

# import self-written auxiliary functions
from rankstdize import rankstdize
from specptfs import specptfs
from perform_eval import perform_eval

# turn off certain warnings
np.warnings.filterwarnings('ignore', category=np.ComplexWarning)
np.warnings.filterwarnings('ignore', category=RuntimeWarning)

def Daily_Spectral_Portfolios_Nonoverlap(
    filename: str,
    rollwin: int,
    momwin: int,
    fwdwin: int,
    minyr: int,
    maxyr: int,
    cntr: int,
    Stype: str,
    savedata: int,
    writefile: List[str],
    Neig: int
) -> None:
    """
    Perform daily spectral portfolios with non-overlapping data.

    :param filename: The name of the file to load the data from.
    :type filename: str
    :param rollwin: The rolling window size.
    :type rollwin: int
    :param momwin: The momentum window size.
    :type momwin: int
    :param fwdwin: The forward window size.
    :type fwdwin: int
    :param minyr: The minimum year.
    :type minyr: int
    :param maxyr: The maximum year.
    :type maxyr: int
    :param cntr: The centering flag.
    :type cntr: int
    :param Stype: The signal type.
    :type Stype: str
    :param savedata: The flag indicating whether to save the data.
    :type savedata: int
    :param writefile: The name of the file to write the results to.
    :type writefile: List[str]
    :param Neig: The number of top eigenportfolios.
    :type Neig: int

    :return: None
    """

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

    # Save results to file
    datafile = f'../data/Results/{filename.replace(".npz", "")}-rollwin-{rollwin}-momwin-{momwin}-fwdwin-{fwdwin}-center-{cntr}-{Stype}-nonoverlap'
    if savedata == 1:
        np.savez(datafile, Ftil=Ftil, Wfull1=Wfull1, Wfull2=Wfull2, Wsym=Wsym, Wasym=Wasym, PP=PP, PEP=PEP, PAP=PAP, Dfull=Dfull, Dsym=Dsym, Dasym=Dasym)

    # Evaluate performance and write to file
    if len(writefile) != 0:
        filetmp = '../data/Results/' + writefile[0] + '.txt'
        with open(filetmp, 'a') as fid:
            fid.write(filename.replace('.npz', '') + ',' + Stype + ',')
            fid.write('%5.0f,%5.0f,%5.0f,%5.0f,%5.0f,%5.0f,' % (fwdwin, rollwin, momwin, minyr, maxyr, cntr))

    # Restrict dates
    loc = (datesno >= minyr * 10000) & (datesno < (maxyr + 1) * 10000)
    full = np.nanmean(PP[loc, 0:Neig], axis=1)
    pos = np.nanmean(PEP[loc, 0:Neig], axis=1)
    neg = np.nanmean(PEP[loc, -Neig:], axis=1)
    pnm = pos - neg * np.nanstd(pos) / np.nanstd(neg)
    asym = np.nanmean(PAP[loc, 0:Neig], axis=1)
    posasym = pos + asym * np.nanstd(pos) / np.nanstd(asym)
    pnmasym = pnm + asym * np.nanstd(pnm) / np.nanstd(asym)
    Ftil = np.atleast_2d(Ftil[loc]).T
    FFfwd = FFfwd[loc, :]
    bench = np.concatenate((Ftil, FFfwd), axis=1)

    # Factor
    FtilSR, FtilSRse, _, _ = perform_eval(Ftil, [], fwdwin / 250)
    # Top Neig full
    PPSR, PPSRse, PPIR, PPIRse = perform_eval(np.atleast_2d(full).T, bench, fwdwin / 250)
    _, _, PPIRfac, PPIRsefac = perform_eval(np.atleast_2d(full).T, Ftil, fwdwin / 250)
    # Top Neig positive
    PEPposSR, PEPposSRse, PEPposIR, PEPposIRse = perform_eval(np.atleast_2d(pos).T, bench, fwdwin / 250)
    _, _, PEPposIRfac, PEPposIRsefac = perform_eval(np.atleast_2d(pos).T, Ftil, fwdwin / 250)
    # Top Neig negative
    PEPnegSR, PEPnegSRse, PEPnegIR, PEPnegIRse = perform_eval(np.atleast_2d(neg).T, bench, fwdwin / 250)
    _, _, PEPnegIRfac, PEPnegIRsefac = perform_eval(np.atleast_2d(neg).T, Ftil, fwdwin / 250)
    # Top Neig symmetric pnm
    pnmSR, pnmSRse, pnmIR, pnmIRse = perform_eval(np.atleast_2d(pnm).T, bench, fwdwin / 250)
    _, _, pnmIRfac, pnmIRsefac = perform_eval(np.atleast_2d(pnm).T, Ftil, fwdwin / 250)
    # Top Neig asymmetric
    PAPSR, PAPSRse, PAPIR, PAPIRse = perform_eval(np.atleast_2d(asym).T, bench, fwdwin / 250)
    _, _, PAPIRfac, PAPIRsefac = perform_eval(np.atleast_2d(asym).T, Ftil, fwdwin / 250)
    # Top Neig positive and top Neig asymmetric
    PEPPAPSR, PEPPAPSRse, PEPPAPIR, PEPPAPIRse = perform_eval(np.atleast_2d(posasym).T, bench, fwdwin / 250)
    _, _, PEPPAPIRfac, PEPPAPIRsefac = perform_eval(np.atleast_2d(posasym).T, Ftil, fwdwin / 250)
    # Top Neig pnm and top Neig asymmetric
    pnmPAPSR, pnmPAPSRse, pnmPAPIR, pnmPAPIRse = perform_eval(np.atleast_2d(pnmasym).T, bench, fwdwin / 250)
    _, _, pnmPAPIRfac, pnmPAPIRsefac = perform_eval(np.atleast_2d(pnmasym).T, Ftil, fwdwin / 250)

    # Write file
    output = np.concatenate([FtilSR, FtilSRse, PPSR, PPSRse, PPIR, PPIRse, PPIRfac, PPIRsefac,
                        PEPposSR, PEPposSRse, PEPposIR, PEPposIRse, PEPposIRfac, PEPposIRsefac,
                        PEPnegSR, PEPnegSRse, PEPnegIR, PEPnegIRse, PEPnegIRfac, PEPnegIRsefac,
                        pnmSR, pnmSRse, pnmIR, pnmIRse, pnmIRfac, pnmIRsefac,
                        PAPSR, PAPSRse, PAPIR, PAPIRse, PAPIRfac, PAPIRsefac,
                        PEPPAPSR, PEPPAPSRse, PEPPAPIR, PEPPAPIRse, PEPPAPIRfac, PEPPAPIRsefac,
                        pnmPAPSR, pnmPAPSRse, pnmPAPIR, pnmPAPIRse, pnmPAPIRfac, pnmPAPIRsefac, [[Neig]]], axis=0).T[0]

    if len(writefile) != 0:
        with open(filetmp, 'a') as fid:
            fid.write('%5.4f,' * output.size % tuple(output))


