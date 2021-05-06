import pandas as pd
import numpy as np
df = pd.read_csv (r'../Data/miniDataset_window_15 .csv')
sequence = df['Sequence'].astype(str).values.tolist()

#Number of Partition
l = 5
subSeqLen = int(len(sequence[1]) / l)
if (subSeqLen < 1):
    print("Invalid Data: Number of partition is bigger than sequence length. Existing...")
    exit()
#Sets
c1 = {'G','A','V','L','I','M','P','F','W'}
c2 = {'Q','N','S','T','Y','C'}
c3 = {'D','E'}
c4 = {'H','K','R'}
#Charecteristics Sequences
hSeqList1 = list()
hSeqList2 = list()
hSeqList3 = list()

#FeatureHeaders
H1 = list()
H2 = list()
H3 = list()
for i in range(l):
    str1 = "EBGW_H1_" + str(i+1)
    str2 = "EBGW_H2_" + str(i+1)
    str3 = "EBGW_H3_" + str(i+1)
    H1.append(str1)
    H2.append(str2)
    H3.append(str3)

h1Mat = np.zeros((len(sequence),l))
h2Mat = np.zeros((len(sequence),l))
h3Mat = np.zeros((len(sequence),l))

seqCount = 0
for seq in sequence:
    hSeq1 = list()
    hSeq2 = list()
    hSeq3 = list()
    curLen = 0
    subSeqCount = 0
    weight = [0, 0, 0]
    freq = [0.0, 0.0, 0.0]
    for ch in str(seq):
        curLen = curLen + 1
        if ch in c1 or ch in c2:
            hSeq1.append(1)
            weight[0] = weight[0] + 1
        else:
            hSeq1.append(0)
        if ch in c1 or ch in c3:
            hSeq2.append(1)
            weight[1] = weight[1] + 1
        else:
            hSeq2.append(0)
        if ch in c1 or ch in c3:
            hSeq3.append(1)
            weight[2] = weight[2] + 1
        else:
            hSeq3.append(0)
        #Normalize
        for i in range(3):
            freq[i] = float(weight[i])/float(curLen)
        #grouped Normalized weight
        if ((curLen % subSeqLen) == 0):
            h1Mat[seqCount][subSeqCount] = freq[0]
            h2Mat[seqCount][subSeqCount] = freq[1]
            h3Mat[seqCount][subSeqCount] = freq[2]
            subSeqCount = subSeqCount + 1

    seqCount = seqCount + 1
    hSeqList1.append(hSeq1)
    hSeqList2.append(hSeq2)
    hSeqList3.append(hSeq3)
df1 = pd.DataFrame(data=h1Mat, columns=H1)
df2 = pd.DataFrame(data=h2Mat, columns=H2)
df3 = pd.DataFrame(data=h3Mat, columns=H3)

featureMat = pd.concat([df['Class'], df1, df2, df3], axis=1)
#featureMat.reset_index(drop=True, inplace=True)
#print(featureMat)
featureMat.to_csv('EBGW_featureMatrix.csv', index=False)
#END


