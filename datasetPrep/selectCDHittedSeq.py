import csv
import pandas as pd

csv_file = r"Succinylation.csv"
df = pd.read_csv(csv_file)
plmdList = []
with open("../Data/cdHit_20_out.1", "r", encoding="utf-8") as fin:
    for line in fin:
        if line.startswith('>'):
            plmdList.append(line[1:-1])
print(plmdList)
print(len(plmdList))
print(df.shape)
df = df.loc[df['PLMD ID'].isin(plmdList)]
print(df.shape)
print(df['PLMD ID'].nunique())
df.to_csv("selectedCDHittedSeq.csv", index = False)