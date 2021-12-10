import csv

elm_file = r"../Data/Succinylation.elm"
csv_file = r"Succinylation.csv"
in_txt = csv.reader(open(elm_file, "r"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'w'))

out_csv.writerows(in_txt)
