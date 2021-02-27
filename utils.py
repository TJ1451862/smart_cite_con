import csv


def read_data(data_path):
    f = csv.reader(open(data_path, "r", encoding='utf-8'))
    data_list = []
    is_first_line = True
    for i in f:
        if is_first_line:
            is_first_line = False
            continue
        else:
            i[2] = int(i[2])
            data_list.append(i)
    return data_list
