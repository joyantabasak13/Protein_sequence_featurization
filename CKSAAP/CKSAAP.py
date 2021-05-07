import pandas as pd
import numpy as np
df = pd.read_csv (r'../Data/miniDataset_window_15 .csv')
sequence = df['Sequence'].astype(str).values.tolist()

windowSize = len(sequence[1])
k = 5
if (k>(windowSize-2)):
    print("Invalid K value. No such feature Possible")
    exit()

aminoAcids = ['G','A','V','L','I','M','P','F','W','Q','N','S','T','Y','C', 'D','E', 'H','K','R']
aaDict = {}

headerList = list()
val = 0
for dist in range(k):
    for aa1 in aminoAcids:
        for aa2 in aminoAcids:
            fName = aa1 + "_" + str(dist) + "_" + aa2
            headerList.append(fName)
            aaDict[fName] = val
            val = val + 1

fMat = np.zeros((len(sequence),len(headerList)))
print(fMat.shape)
print(len(headerList))
#print(aaDict)

perDistFeature = len(aminoAcids)*len(aminoAcids)
print(perDistFeature)
seqCount = 0
for seq in sequence:
    for dist in range(k):
        for i in range(windowSize - dist - 1): #Exclude last ch
            fName = seq[i] + "_" + str(dist) + "_" + seq[i+dist+1]
            pos = aaDict[fName]
            fMat[seqCount][pos] = fMat[seqCount][pos] + 1.0
        #Normalize
        startIndex = dist * perDistFeature
        possibleFeatureCount = float(windowSize - dist - 1)

        for i in range(startIndex, startIndex + perDistFeature):
            fMat[seqCount][i] = fMat[seqCount][i] / possibleFeatureCount

    seqCount = seqCount + 1

featureMat = pd.DataFrame(data=fMat, columns=headerList)
featureMat = pd.concat([df['Class'],featureMat], axis=1)
featureMat.to_csv('CKSAAP_featureMatrix.csv', index=False)
#END


