# Run cases
rerun = 1
filesuffix = str(int(time.time() * 10000)).zfill(9)

# Choices
savedata = 0
rollwinlist = [120]
momwinlist = [20]
fwdwinlist = [20]
Neiglist = [3]
Stypelist = ['mom']
samplist = [(1963, 2019)]
cntrlist = [1]
filelist = ['25_Size_BM']

if rerun == 1:
    for rr in range(len(rollwinlist)):
        for mm in range(len(momwinlist)):
            for m in range(len(samplist)):
                for f in range(len(fwdwinlist)):
                    for c in range(len(cntrlist)):
                        for s in range(len(Stypelist)):
                            for nn in range(len(Neiglist)):
                                for ff in range(len(filelist)):
                                    filename = filelist[ff]
                                    writefile = f'Results_{filename}_{filesuffix}'
                                    Daily_Spectral_Portfolios_Nonoverlap(
                                        filename, rollwinlist[rr], momwinlist[mm], fwdwinlist[f],
                                        samplist[m][0], samplist[m][1], cntrlist[c], Stypelist[s],
                                        savedata, writefile, Neiglist[nn]
                                    )

# Collect cases
for rr in range(len(rollwinlist)):
    for mm in range(len(momwinlist)):
        for m in range(len(samplist)):
            for c in range(len(cntrlist)):
                for s in range(len(Stypelist)):
                    for nn in range(len(Neiglist)):
                        for ff in range(len(filelist)):
                            filename = filelist[ff]
                            writefile = f'Results_{filename}_{filesuffix}'

                            filetmp = f'./Data/Results/{writefile}.txt'
                            with open(filetmp, 'r') as f:
                                data = f.readlines()

                            dataset = []
                            Stype = []
                            fwdwin = []
                            rollwin = []
                            momwin = []
                            minyr = []
                            maxyr = []
                            cntr = []
                            FtilSR = []
                            FtilSRse = []
                            PPSR = []
                            PPSRse = []
                            PPIR = []
                            PPIRse = []
                            PPIRfac = []
                            PPIRsefac = []
                            PEPposSR = []
                            PEPposSRse = []
                            PEPposIR = []
                            PEPposIRse = []
                            PEPposIRfac = []
                            PEPposIRsefac = []
                            PEPnegSR = []
                            PEPnegSRse = []
                            PEPnegIR = []
                            PEPnegIRse = []
                            PEPnegIRfac = []
                            PEPnegIRsefac = []
                            pmnSR = []
                            pmnSRse = []
                            pmnIR = []
                            pmnIRse = []
                            pmnIRfac = []
                            pmnIRsefac = []
                            PAPSR = []
                            PAPSRse = []
                            PAPIR = []
                            PAPIRse = []
                            PAPIRfac = []
                            PAPIRsefac = []
                            PEPPAPSR = []
                            PEPPAPSRse = []
                            PEPPAPIR = []
                            PEPPAPIRse = []
                            PEPPAPIRfac = []
                            PEPPAPIRsefac = []
                            pmnPAPSR = []
                            pmnPAPSRse = []
                            pmnPAPIR = []
                            pmnPAPIRse = []
                            pmnPAPIRfac = []
                            pmnPAPIRsefac = []
                            Neig = []

                            for line in data:
                                values = line.strip().split(',')
                                dataset.append(values[0])
                                Stype.append(values[1])
                                fwdwin.append(float(values[2]))
                                rollwin.append(float(values[3]))
                                momwin.append(float(values[4]))
                                minyr.append(float(values[5]))
                                maxyr.append(float(values[6]))
                                cntr.append(float(values[7]))
                                FtilSR.append(float(values[8]))
                                FtilSRse.append(float(values[9]))
                                PPSR.append(float(values[10]))
                                PPSRse.append(float(values[11]))
                                PPIR.append(float(values[12]))
                                PPIRse.append(float(values[13]))
                                PPIRfac.append(float(values[14]))
                                PPIRsefac.append(float(values[15]))
                                PEPposSR.append(float(values[16]))
                                PEPposSRse.append(float(values[17]))
                                PEPposIR.append(float(values[18]))
                                PEPposIRse.append(float(values[19]))
                                PEPposIRfac.append(float(values[20]))
                                PEPposIRsefac.append(float(values[21]))
                                PEPnegSR.append(float(values[22]))
                                PEPnegSRse.append(float(values[23]))
                                PEPnegIR.append(float(values[24]))
                                PEPnegIRse.append(float(values[25]))
                                PEPnegIRfac.append(float(values[26]))
                                PEPnegIRsefac.append(float(values[27]))
                                pmnSR.append(float(values[28]))
                                pmnSRse.append(float(values[29]))
                                pmnIR.append(float(values[30]))
                                pmnIRse.append(float(values[31]))
                                pmnIRfac.append(float(values[32]))
                                pmnIRsefac.append(float(values[33]))
                                PAPSR.append(float(values[34]))
                                PAPSRse.append(float(values[35]))
                                PAPIR.append(float(values[36]))
                                PAPIRse.append(float(values[37]))
                                PAPIRfac.append(float(values[38]))
                                PAPIRsefac.append(float(values[39]))
                                PEPPAPSR.append(float(values[40]))
                                PEPPAPSRse.append(float(values[41]))
                                PEPPAPIR.append(float(values[42]))
                                PEPPAPIRse.append(float(values[43]))
                                PEPPAPIRfac.append(float(values[44]))
                                PEPPAPIRsefac.append(float(values[45]))
                                pmnPAPSR.append(float(values[46]))
                                pmnPAPSRse.append(float(values[47]))
                                pmnPAPIR.append(float(values[48]))
                                pmnPAPIRse.append(float(values[49]))
                                pmnPAPIRfac.append(float(values[50]))
                                pmnPAPIRsefac.append(float(values[51]))
                                Neig.append(float(values[52]))

                            # Plots
                            caseloc = np.where(
                                (np.array(Neig) == Neiglist[nn]) &
                                (np.array(rollwin) == rollwinlist[rr]) &
                                (np.array(momwin) == momwinlist[mm]) &
                                (np.array(minyr) == samplist[m][0]) &
                                (np.array(maxyr) == samplist[m][1]) &
                                (np.array(cntr) == cntrlist[c]) &
                                (np.array(Stype) == Stypelist[s])
                            )[0]

                            figdir = './Figures/'

                            # Sharpe and info ratios
                            bardata = np.vstack(
                                [PPSR[caseloc, 0], PEPposSR[caseloc, 0], PAPSR[caseloc, 0], PEPPAPSR[caseloc, 0], FtilSR[caseloc, 0]],
                                [PPIR[caseloc, 0], PEPposIR[caseloc, 0], PAPIR[caseloc, 0], PEPPAPIR[caseloc, 0], 0]
                            ).T
                            barse = np.vstack(
                                [PPSRse[caseloc, 0], PEPposSRse[caseloc, 0], PAPSRse[caseloc, 0], PEPPAPSRse[caseloc, 0], FtilSRse[caseloc, 0]],
                                [PPIRse[caseloc, 0], PEPposIRse[caseloc, 0], PAPIRse[caseloc, 0], PEPPAPIRse[caseloc, 0], 0]
                            ).T

                            b = errorbargrouped(bardata, barse, 2)
                            b[4].set_facecolor('none')
                            b[4].set_edgecolor('black')
                            b[4].set_hatch('//')
                            b[4].set_linewidth(0)
                            b[4].set_label('Factor')

                            plt.xticks([1, 2], ['Sharpe Ratio', 'Information Ratio'])
                            plt.xlabel('Forecast Horizon (Days)')
                            plt.gca().set_ylabel('Ratio')
                            plt.gca().set_title('Sharpe and Information Ratios')
                            plt.gca().set_xticklabels(['Sharpe Ratio', 'Information Ratio'])
                            plt.gca().spines['right'].set_visible(False)
                            plt.gca().spines['top'].set_visible(False)
                            plt.gca().tick_params(axis='both', which='both', length=0)
                            plt.gca().legend()

                            plt.gcf().set_size_inches(12.8, 10)
                            plt.savefig(figdir + 'Figure3.eps', format='eps')
                            plt.close()
