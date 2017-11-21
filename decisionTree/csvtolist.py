import csv
with open('Attrition.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

print len(your_list)

header = your_list[0]
#print(header,"\n\n")

label = header[1]
#print(label,"\n\n")

del header [1]
#print(header,"\n\n")

header.append(label)
#print(header,"\n\n")

training_data = your_list
del training_data[0]

print(training_data[0])

