import numpy as np

def Daily_Spectral_Portfolios_Nonoverlap(filename, rollwin, momwin, fwdwin, minyr, maxyr, cntr, Stype, savedata, writefile, Neig):

    def rankstdize(data):
        rank = np.argsort(np.argsort(data, axis=0), axis=0)
        rank = rank.astype(float)
        rank[np.isnan(data)] = np.nan
        rank -= np.nanmean(rank, axis=1)[:, None]
        rank /= np.nanstd(rank, axis=1, ddof=1)[:, None]
        rank[np.isnan(rank)] = 0
        return rank

    if filename.startswith('MFret'):
        volstd = 1
    else:
        volstd = 0

    data = np.load('./Data/' + filename + '.npz')
    retd = data['retd']
    datesd = data['datesd']

    # Restrict dates
    ixtmp = np.where(np.floor(datesd / 10000) == minyr)[0][0]
    strt = max(ixtmp - rollwin * fwdwin - 2, 0)
    loc = (datesd >= np.floor(datesd[strt] / 10000) * 10000) & (datesd < (maxyr + 1) * 10000)
    retd = retd[loc, :]
    datesd = datesd[loc]

    # Drop columns with missing data
    loc = np.sum(np.isnan(retd), axis=0) / retd.shape[0] <= 0.02
    retd = retd[:, loc]
    retd[np.isnan(retd)] = 0

    # Vol-standardize for MF data
    if volstd == 1:
        retd = np.log(1 + retd)

    # Load/merge Fama-French factor data for benchmarking
    fffac = data['fffac']
    ffdates = data['ffdates']
    rf = data['rf']
    ia = np.intersect1d(np.where(np.isin(datesd, ffdates)), np.arange(len(datesd)))
    retd = retd[ia, :]
    datesd = datesd[ia]
    rfd = rf[ia]
    fffacd = fffac[ia, :]

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
        loc = np.where(fwdix == t)[0]
        if volstd == 1:
            tmpstd = np.nanstd(retd[max(loc[0] - 20, 0):max(loc[0], 20), :], axis=0)
            Rfwd[t - 1, :] = np.nansum(retd[loc, :] / tmpstd, axis=1)
        datesno[t - 1] = datesd[loc[-1]]

        # Forecast FF observation
        mkt = fffacd[loc, 0] + rfd[loc]
        ffoth = fffacd[loc, 1:]
        if volstd == 1:
            FFfwd[t - 1, :] = np.nansum(np.concatenate(([mkt], ffoth), axis=0), axis=0)
            RFfwd = np.nansum(rfd[loc], axis=0)
        else:
            FFfwd[t - 1, :] = np.prod(np.concatenate(([mkt], ffoth), axis=0), axis=0) - 1
            RFfwd = np.prod(1 + rfd[loc], axis=0) - 1
        FFfwd[t - 1, 0] = FFfwd[t - 1, 0] - RFfwd

        # Signal observation (indexed to align with return it predicts, lagged 'skip' days)
        sigloc = np.arange(loc[0] - skip - (momwin - 1), loc[0] - skip)
        if sigloc[0] < 0:
            continue
        if Stype == 'mom':
            if volstd == 1:
                S[t - 1, :] = np.nansum(retd[sigloc, :] / tmpstd, axis=0)
            else:
                S[t - 1, :] = np.nansum(retd[sigloc, :], axis=0)
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

    Rtrnslc = []
    Strnslc = []
    Rtstslc = []
    Ststslc = []

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

    for tau in range(rollwin, Tno):
        Rtrn = Rtrnslc[tau - rollwin]
        Strn = Strnslc[tau - rollwin]

        if np.isnan(np.sum(Rtrn)) or np.isnan(np.sum(Strn)):
            continue

        Rtst = Rtstslc[tau - rollwin]
        Stst = Ststslc[tau - rollwin]

        if cntr == 1:
            Strn = rankstdize(Strn)
            Stst = rankstdize(Stst)

        if cntr == 1:
            Rtrn -= np.nanmean(Rtrn, axis=1)[:, None]
            Rtst -= np.nanmean(Rtst, axis=1)[:, None]

        Ftil[tau] = np.sum(Stst * Rtst)

        W1, W2, Df, Ws, Ds, Wa, Da, PPtmp, PEPtmp, PAPtmp = specptfs(Rtrn, Strn, Rtst, Stst)
        PP[tau, :] = PPtmp
        PEP[tau, :] = PEPtmp
        PAP[tau, :] = PAPtmp
        Dfull[tau, :] = Df
        Dsym[tau, :] = Ds
        Dasym[tau, :] = Da
        Wfull1[tau] = W1
        Wfull2[tau] = W2
        Wsym[tau] = Ws
        Wasym[tau] = Wa

        if tau % 100 == 0:
            print(tau)

    datafile = f'./Data/Results/{filename.replace(".mat", "")}-rollwin-{rollwin}-momwin-{momwin}-fwdwin-{fwdwin}-center-{cntr}-{Stype}-nonoverlap'
    if savedata == 1:
        np.savez(datafile, Ftil=Ftil, Wfull1=Wfull1, Wfull2=Wfull2, Wsym=Wsym, Wasym=Wasym, PP=PP, PEP=PEP, PAP=PAP, Dfull=Dfull, Dsym=Dsym, Dasym=Dasym)

    # Evaluate performance and write to file
    if writefile:
        filetmp = './Data/Results/' + writefile + '.txt'
        with open(filetmp, 'a') as fid:
            fid.write(erase(filename, '.mat') + ',' + Stype + ',')
            fid.write('%5.0f,%5.0f,%5.0f,%5.0f,%5.0f,%5.0f,%5.0f,' % (fwdwin, rollwin, momwin, minyr, maxyr, cntr))

    # Restrict dates
    loc = np.where((datesno >= minyr * 10000) & (datesno < (maxyr + 1) * 10000))[0]
    full = np.nanmean(PP[loc, 0:Neig], axis=1)
    pos = np.nanmean(PEP[loc, 0:Neig], axis=1)
    neg = np.nanmean(PEP[loc, -Neig:], axis=1)
    pnm = pos - neg * np.nanstd(pos) / np.nanstd(neg)
    asym = np.nanmean(PAP[loc, 0:Neig], axis=1)
    posasym = pos + asym * np.nanstd(pos) / np.nanstd(asym)
    pnmasym = pnm + asym * np.nanstd(pnm) / np.nanstd(asym)
    Ftil = Ftil[loc, :]
    FFfwd = FFfwd[loc, :]

    # Factor
    FtilSR, FtilSRse, _, _ = perform_eval(Ftil, [], fwdwin / 250)
    # Top Neig full
    PPSR, PPSRse, PPIR, PPIRse = perform_eval(full, [Ftil, FFfwd], fwdwin / 250)
    _, _, PPIRfac, PPIRsefac = perform_eval(full, Ftil, fwdwin / 250)
    # Top Neig positive
    PEPposSR, PEPposSRse, PEPposIR, PEPposIRse = perform_eval(pos, [Ftil, FFfwd], fwdwin / 250)
    _, _, PEPposIRfac, PEPposIRsefac = perform_eval(pos, Ftil, fwdwin / 250)
    # Top Neig negative
    PEPnegSR, PEPnegSRse, PEPnegIR, PEPnegIRse = perform_eval(neg, [Ftil, FFfwd], fwdwin / 250)
    _, _, PEPnegIRfac, PEPnegIRsefac = perform_eval(neg, Ftil, fwdwin / 250)
    # Top Neig symmetric pnm
    pnmSR, pnmSRse, pnmIR, pnmIRse = perform_eval(pnm, [Ftil, FFfwd], fwdwin / 250)
    _, _, pnmIRfac, pnmIRsefac = perform_eval(pnm, Ftil, fwdwin / 250)
    # Top Neig asymmetric
    PAPSR, PAPSRse, PAPIR, PAPIRse = perform_eval(asym, [Ftil, FFfwd], fwdwin / 250)
    _, _, PAPIRfac, PAPIRsefac = perform_eval(asym, Ftil, fwdwin / 250)
    # Top Neig positive and top Neig asymmetric
    PEPPAPSR, PEPPAPSRse, PEPPAPIR, PEPPAPIRse = perform_eval(posasym, [Ftil, FFfwd], fwdwin / 250)
    _, _, PEPPAPIRfac, PEPPAPIRsefac = perform_eval(posasym, Ftil, fwdwin / 250)
    # Top Neig pnm and top Neig asymmetric
    pnmPAPSR, pnmPAPSRse, pnmPAPIR, pnmPAPIRse = perform_eval(pnmasym, [Ftil, FFfwd], fwdwin / 250)
    _, _, pnmPAPIRfac, pnmPAPIRsefac = perform_eval(pnmasym, Ftil, fwdwin / 250)

    # Write file
    output = np.hstack((FtilSR, FtilSRse, PPSR, PPSRse, PPIR, PPIRse, PPIRfac, PPIRsefac,
                        PEPposSR, PEPposSRse, PEPposIR, PEPposIRse, PEPposIRfac, PEPposIRsefac,
                        PEPnegSR, PEPnegSRse, PEPnegIR, PEPnegIRse, PEPnegIRfac, PEPnegIRsefac,
                        pnmSR, pnmSRse, pnmIR, pnmIRse, pnmIRfac, pnmIRsefac,
                        PAPSR, PAPSRse, PAPIR, PAPIRse, PAPIRfac, PAPIRsefac,
                        PEPPAPSR, PEPPAPSRse, PEPPAPIR, PEPPAPIRse, PEPPAPIRfac, PEPPAPIRsefac,
                        pnmPAPSR, pnmPAPSRse, pnmPAPIR, pnmPAPIRse, pnmPAPIRfac, pnmPAPIRsefac, Neig))

    if writefile:
        with open(filetmp, 'a') as fid:
            fid.write('%5.4f,' * (output.size - 1) % tuple(output))
            fid.write('%5.4f\n' % output[-1])


