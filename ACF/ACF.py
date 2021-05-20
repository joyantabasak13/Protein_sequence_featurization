import numpy as np
import pandas as pd

df = pd.read_csv(r'../Data/miniDataset_window_15 .csv')

N_exmp = len(df.index)
L = len(df['Sequence'][0])

# Amino acid side chain interaction parameters found from 'krigbaum1979'
aminoAcidSCIP = {'A': 4.32, 'C': 1.73, 'D': 6.04, 'E': 6.17, 'F': 2.59, 'G': 6.09, 'H': 5.66, 'I': 2.31,
                 'K': 7.92, 'L': 3.93, 'M': 2.44, 'N': 6.24, 'P': 7.19,  'Q': 6.13, 'R': 6.55, 'S': 5.37,
                 'T': 5.16, 'V': 3.31, 'W': 2.78, 'Y': 3.58}

# Feature matrix
ftMat = np.empty((N_exmp, L-1))

for row in range(N_exmp):
    seq = df['Sequence'][row]

    # Peptide is transformed into a numerical vector of SCIP
    S = np.empty(L)
    for i in range(L):
        S[i] = aminoAcidSCIP[seq[i]]

    # ACF calculation
    for m in range(1, L):
        # r_m calculation
        r_m = 0
        for i in range(L-m):
            r_m = r_m + S[i]*S[i+m]
        r_m = r_m/(L-m)
        ftMat[row][m-1] = r_m

headerList = list()
for i in range(1, L):
    s = 'R_' + str(i)
    headerList.append(s)

featureMat = pd.DataFrame(data=ftMat, columns=headerList)
featureMat = pd.concat([df['Class'], featureMat], axis=1)
pd.DataFrame(featureMat).to_csv("ACF_featureMatrix.csv", index=False)
