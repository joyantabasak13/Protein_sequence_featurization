import pandas as pd

df = pd.read_csv (r'../Data/miniDataset_window_15 .csv')
sequence = pd.DataFrame(df, columns= ['Sequence']).values.tolist()

#Sets
c1 = {'G','A','V','L','I','M','P','F','W'}
c2 = {'Q','N','S','T','Y','C'}
c3 = {'D','E'}
c4 = {'H','K','R'}
#Charecteristics Sequences
hSeqList1 = list()
hSeqList2 = list()
hSeqList3 = list()
count =0
for seq in sequence:
    hSeq1 = list()
    hSeq2 = list()
    hSeq3 = list()
    #extra "[" and "'" are added in str(seq)
    for ch in str(seq)[2:-2]:
        if ch in c1 or ch in c2:
            hSeq1.append(1)
        else:
            hSeq1.append(0)
        if ch in c1 or ch in c3:
            hSeq2.append(1)
        else:
            hSeq2.append(0)
        if ch in c1 or ch in c3:
            hSeq3.append(1)
        else:
            hSeq3.append(0)
    hSeqList1.append(hSeq1)
    hSeqList2.append(hSeq2)
    hSeqList3.append(hSeq3)
#


