import csv
with open('WA_Fn-UseC_-HR-Employee-Attrition.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

print your_list