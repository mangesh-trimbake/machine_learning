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

print(len(training_data))

for current_row in training_data :
	current_label = current_row[1]
	del current_row[1]
	current_row.append(current_label)

print(training_data[0])
