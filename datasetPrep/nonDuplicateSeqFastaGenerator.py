import csv
import pandas as pd

csv_file = r"Succinylation.csv"
df = pd.read_csv(csv_file)
print(df.head())
result_df = df.drop_duplicates(subset=['Sequence'], keep='first')
print(result_df.shape[0])

write_file = "nonDupSeq.fasta"
with open(write_file, "w", encoding="utf-8") as output:
    for i in range(result_df.shape[0]):
        line1 = ">" + result_df.iat[i,0] + "\t" + result_df.iat[i,5]
        line2 = result_df.iat[i,4]
        output.write(line1 + '\n')
        output.write(line2 + '\n')
