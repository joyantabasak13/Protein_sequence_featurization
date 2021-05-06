import pandas as pd
import numpy as np
df = pd.read_csv (r'../Data/miniDataset_window_15 .csv')
sequence = df['Sequence'].astype(str).values.tolist()

windowSize = (len(sequence[1]) -1)/2
aminoAcids = ['G','A','V','L','I','M','P','F','W','Q','N','S','T','Y','C', 'D','E', 'H','K','R']
headerList = list()
for i in range(len(aminoAcids)):
    fName = "WAAC_" + str(aminoAcids[i])
    headerList.append(fName)

fMat = np.zeros((len(sequence),len(headerList)))
print(fMat.shape)
const = 1.0/float(windowSize*(windowSize+1))
print(const)
seqIndex = 0
for seq in sequence:
    i = 0
    for amino in aminoAcids:
        val = 0.0
        j = -windowSize
        for ch in str(seq):
            if (ch == amino):
               val = val + (float(j) + float(float(abs(j)) / float(windowSize)))
            j = j + 1
        val = val * const
        fMat[seqIndex][i] = val
        i = i + 1
    seqIndex = seqIndex + 1
print(sequence[0])
print(fMat[0])

featureMat = pd.DataFrame(data=fMat, columns=headerList)
featureMat = pd.concat([df['Class'],featureMat], axis=1)
print(featureMat)
featureMat.to_csv('WAAC_featureMatrix.csv', index=False)
#END


