# import packages
import pandas as pd
import numpy as np

"""
Here we load 14 different FF datasets:
1. FF3 Daily
2. FF3 Monthly
3. 25 Size/BM Daily and Monthly
4. 25 Size/OP Daily
5. 25 Size/Inv Daily
6. FF3 Developed ex US Daily
7. FF5 Developed ex US Monthly
8. 25 Size/BM Developed ex US Daily and Monthly
9. 25 Size/OP Developed ex US Daily and Monthly
10. 25 Size/Inv Developed ex US Daily and Monthly
"""

# set default directory
maindir = "C:/Users/phils/OneDrive/Dokumente/Studium/4_PhD/3_Courses/CrossSectional_AssetPricing/Presentations/code/python/"

### here write a function(s) from .ipynb since the structure of loading daily factors, monthly factors, daily 25 portfolios, monthly 25 portfolios is extremely seimiliar, except the paths, skiprows and nrows.