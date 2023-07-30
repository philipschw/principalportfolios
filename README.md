# Principal Portfolios

Python Code to compute the Figures and Tables in Kelly, Malamud and Pedersen (2022) [Principal Portfolios](https://doi.org/10.1111/jofi.13199).

## Data
To run this code you have to create a `data` and `data\Results` directory. The data provided by the authors can be found [here](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fjofi.13199&file=jofi13199-sup-0002-ReplicationCode.zip). After downloading the files, load the data by running `LoadFamaFrenchData.ipynb`.

## Code:

* `Figure1.py`: Creates Figure 1.
* `Figure2.py`: Creates Figure 2.
* `Figure3.py`: Creates Figure 3.
* `Table2.py`: Creates Table 2.
* `Figure4_Figure5_Table3.py`: Creates Figure4, Figure 5, Table 3.
* `FigureIA1.py`: Creates Internet Appendix Figure 1 (not implemented).
* `FigureIA2_PanelA_PanelB.py`: Creates Internet Appendix Figure 2 Panel A and Panel B (not implemented).
* `FigureIA3.py`: Creates Internet Appendix Figure 3 (not implemented).
* `FigureIA4.py`: Creates Internet Appendix Figure 4 (not implemented).
* `FigureIA5.py`: Creates Internet Appendix Figure 5 (not implemented).

* `LoadFamaFrenchData.ipynb`: Loads the 14 different FF datasets used in main paper and internet appendix.
* `Daily_Spectral_Portfolios_Nonoverlap.py`: Perform daily spectral portfolios with non-overlapping data.
* `rankstdize.py`: Perform rank standardization on the input array (the signals).
* `specptfs.py`: Calculate principal portfolios, exposure portfolios, and alpha portfolios based on training and testing data.
* `predeig.py`: Perform eigenvalue decomposition
* `perform_eval.py`: Calculates various performance metrics for a set of test assets and Sharpe Ratios.
* `linreg.py`: Perform linear regression and calculate statistics (t-statistics, Rsquared).
* `errorbargrouped.py`: Create a grouped bar plot with error bars.
* `expsmoother.py`: Perform double exponential smoothing (Holt Linear).

* For more information and advanced options see the documentation for each function.


## ToDo's:

* `Figure4_Figure5_Table3.py`: Improve runtime of main computational loop, e.g. parallelization.
* `FigureIA1.py`, `FigureIA2_PanelA_PanelB.py`, `FigureIA3.py`, `FigureIA4.py`, `FigureIA5.py`: Finish Internet Appendix Code.


## Reference
Kelly, B., Malamud, S., and Pedersen, L. H. Principal Portfolios,
<em>Journal of Finance</em>, 78(1), 347-387 (2022). [[journal]](https://doi.org/10.1111/jofi.13199)<br />


## LICENCSE
This replication work originates from the [replication code](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fjofi.13199&file=jofi13199-sup-0002-ReplicationCode.zip) of Kelly, Malamud and Pedersen (2022) [Principal Portfolios](https://doi.org/10.1111/jofi.13199). Therefore, [AFA JoF Code Sharing Policy](https://afajof.org/wp-content/uploads/files/policies-and-guidelines/CodePolicy.pdf) is still in place. In particular:

Any	person downloading any of the file(s) will need	to certify that the programs will be used only for academic
research. Any other	use, including for commercial purposes, is strictly prohibited except with explicit permission from 
all	authors	of the published article, or, if the code comes from a third-party source, with explicit permission	of the
cited originators of the code. Academic	researchers	using the code,	or fragments of	it, in their own work are required
to acknowledge the origin of the code. Such	researchers	agree that they	will not seek out assistance or further support
of the authors of the article or cited originators of the code.
