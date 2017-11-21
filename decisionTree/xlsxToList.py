from xlrd import open_workbook

file_var = open_workbook("WA_Fn-UseC_-HR-Employee-Attrition.csv.xlsx") 

sheet = file_var.sheet_by_index(0)

list_data = []

for k in range(1,sheet.nrows):
    list_data.append(str(sheet.row_values(k)[j-1]))

print(len(list_data))
