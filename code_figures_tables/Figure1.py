
# import packages
import numpy as np
import matplotlib.pyplot as plt

# import self-written auxiliary functions
from Daily_Spectral_Portfolios_Nonoverlap import Daily_Spectral_Portfolios_Nonoverlap

# Run cases
rerun = 1
minyr = 1963
maxyr = 2019
savedata = 1
rollwin = 120
momwin = 20
fwdwin = 20
cntr = 1
Neig = 3
Stype = ['mom']
writefile = []
filelist = ['25_Size_BM']

if rerun == 1:
    for f in range(len(filelist)):
        filename = filelist[f]
        Daily_Spectral_Portfolios_Nonoverlap(
            filename,
            rollwin,
            momwin,
            fwdwin,
            minyr,
            maxyr,
            cntr,
            Stype[0],
            savedata,
            writefile,
            Neig
        )

# Plot returns and eigenvalues for FF 25

# Load estimates
filename = '25_Size_BM'
figdir = '../figures/'
datafile = '../data/Results/' + filename.replace('.npz', '') + '-rollwin-' + str(rollwin) + '-momwin-' + str(momwin) + '-fwdwin-' + str(fwdwin) + '-center-' + str(cntr) + '-' + Stype[0] + '-nonoverlap'
data = np.load(datafile + '.npz')

PP = data['PP']
PEP = data['PEP']
PAP = data['PAP']
Dfull = data['Dfull']
Dsym = data['Dsym']
Dasym = data['Dasym']

# Mean ER
PPbar = np.nanmean(PP, axis=0)
PPbarse = np.nanstd(PP, axis=0) / np.sqrt(np.sum(~np.isnan(PP[:, 0])))
PEPbar = np.nanmean(PEP, axis=0)
PEPbarse = np.nanstd(PEP, axis=0) / np.sqrt(np.sum(~np.isnan(PEP[:, 0])))
PAPbar = np.nanmean(PAP, axis=0)
PAPbarse = np.nanstd(PAP, axis=0) / np.sqrt(np.sum(~np.isnan(PAP[:, 0])))

# Mean eigs
Dfullbar = np.nanmean(Dfull, axis=0)
Dfullbarse = np.nanstd(Dfull[::rollwin, :], axis=0) / np.sqrt(np.sum(~np.isnan(Dfull[::rollwin, 0])))
Dsymbar = np.nanmean(Dsym, axis=0)
Dsymbarse = np.nanstd(Dsym[::rollwin, :], axis=0) / np.sqrt(np.sum(~np.isnan(Dsym[::rollwin, 0])))
Dasymbar = np.nanmean(Dasym, axis=0)
Dasymbarse = np.nanstd(Dasym[::6, :], axis=0) / np.sqrt(np.sum(~np.isnan(Dasym[::6, 0])))

# PP Rets
plt.figure()
plt.plot(list(range(1,len(PPbar)+1)), 100 * PPbar, '-ok', linewidth=2)
plt.plot(list(range(1,len(PPbar)+1)), 100 * (PPbar + 2 * PPbarse), '--r', linewidth=1)
plt.plot(list(range(1,len(PPbar)+1)), 100 * (PPbar - 2 * PPbarse), '--r', linewidth=1)
plt.ylabel('PP Return (%)')
plt.grid(True)
plt.xlabel('Eigenvalue Number')
plt.xticks([1] + list(range(5, Dsym.shape[1], 5)))
plt.xlim(1, 25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.gcf().set_size_inches(9, 10)
plt.savefig(figdir + 'Figure1d.jpg')
plt.close()

# PP Eigs
plt.figure()
plt.plot(list(range(1,len(Dfullbar)+1)), 100 * Dfullbar, '-ok', linewidth=2)
plt.ylabel('Eigenvalue Return Estimate (%)')
plt.grid(True)
plt.xlabel('Eigenvalue Number')
plt.xticks([1] + list(range(5, Dsym.shape[1], 5)))
plt.xlim(1, 25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.gcf().set_size_inches(9, 10)
plt.savefig(figdir + 'Figure1a.jpg')
plt.close()

# PEP Rets
plt.figure()
plt.plot(list(range(1,len(PEPbar)+1)), 100 * PEPbar, '-ok', linewidth=2)
plt.plot(list(range(1,len(PEPbar)+1)), 100 * (PEPbar + 2 * PEPbarse), '--r', linewidth=1)
plt.plot(list(range(1,len(PEPbar)+1)), 100 * (PEPbar - 2 * PEPbarse), '--r', linewidth=1)
plt.ylabel('PEP Return (%)')
plt.grid(True)
plt.xlabel('Eigenvalue Number')
plt.xticks([1] + list(range(5, Dsym.shape[1], 5)))
plt.xlim(1, 25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.gcf().set_size_inches(9, 10)
plt.savefig(figdir + 'Figure1e.jpg')
plt.close()

# PEP Eigs
plt.figure()
plt.plot(list(range(1,len(Dsymbar)+1)), 100 * Dsymbar, '-ok', linewidth=2)
plt.ylabel('Eigenvalue Return Estimate (%)')
plt.grid(True)
plt.xlabel('Eigenvalue Number')
plt.xticks([1] + list(range(5, Dsym.shape[1], 5)))
plt.xlim(1, 25)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.gcf().set_size_inches(9, 10)
plt.savefig(figdir + 'Figure1b.jpg')
plt.close()

# PAP Rets
plt.figure()
plt.plot(list(range(1,len(PAPbar)+1)), 100 * PAPbar, '-ok', linewidth=2)
plt.plot(list(range(1,len(PAPbar)+1)), 100 * (PAPbar + 2 * PAPbarse), '--r', linewidth=1)
plt.plot(list(range(1,len(PAPbar)+1)), 100 * (PAPbar - 2 * PAPbarse), '--r', linewidth=1)
plt.ylabel('PAP Return (%)')
plt.grid(True)
plt.xlabel('Eigenvalue Number')
plt.xticks(list(range(1, 13)))
plt.xlim(1, 12)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.gcf().set_size_inches(9, 10)
plt.savefig(figdir + 'Figure1f.jpg')
plt.close()

# PAP Eigs
plt.figure()
plt.plot(list(range(1,len(Dasymbar)+1)), 100 * Dasymbar, '-ok', linewidth=2)
plt.ylabel('Eigenvalue Return Estimate (%)')
plt.grid(True)
plt.xlabel('Eigenvalue Number')
plt.xticks(list(range(1, 13)))
plt.xlim(1, 12)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.gcf().set_size_inches(9, 10)
plt.savefig(figdir + 'Figure1c.jpg')
plt.close()