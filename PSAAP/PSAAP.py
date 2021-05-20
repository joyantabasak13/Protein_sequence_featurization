import numpy as np
import pandas as pd

df = pd.read_csv(r'../Data/miniDataset_window_15 .csv')

N_exmp = len(df.index)
L = len(df['Sequence'][0])

# Each amino acid is indexed with a number
aminoAcidMap = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
                'P': 12,  'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'X': 19, 'Y': 20}

Zrow = len(aminoAcidMap)
Zcol = L

Z = np.zeros((Zrow, Zcol))

# Z matrix calculation
for row in range(N_exmp):
    seq = df['Sequence'][row]
    if(df['Class'][row] == 'suc'):
        for j in range(L):
            i = aminoAcidMap[seq[j]]
            # Occurrence frequency of i_th amino acid in j_th position incremented for positive examples
            Z[i][j] = Z[i][j] + 1
    else:
        for j in range(L):
            i = aminoAcidMap[seq[j]]
            # Occurrence frequency of i_th amino acid in j_th position decremented for negative examples
            Z[i][j] = Z[i][j] - 1

ftMat = np.empty((N_exmp, L))
headerList = list()

# Feature generation
for row in range(N_exmp):
    seq = df['Sequence'][row]
    for u in range(L):
        i = aminoAcidMap[seq[u]]
        # Occurrence frequency of i_th amino acid in u_th position
        ftMat[row][u] = Z[i][u]

headerList = list()
for i in range(L):
    s = 'F_R_' + str(i+1)
    headerList.append(s)

featureMat = pd.DataFrame(data=ftMat, columns=headerList)
featureMat = pd.concat([df['Class'], featureMat], axis=1)
pd.DataFrame(featureMat).to_csv("PSAAP_featureMatrix.csv", index=False)
