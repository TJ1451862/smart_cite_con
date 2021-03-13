import csv


def read_data(read_path):
    data_list = []
    with open(read_path, "r", encoding='utf-8') as file:
        reader = csv.reader(file)
        is_first_line = True
        for i in reader:
            if is_first_line:
                is_first_line = False
                continue
            else:
                i[2] = int(i[2])
                data_list.append(i)
    return data_list





def write_data(data_list, write_path):
    headers = ['text_a', 'text_b', 'label']
    with open(write_path, "w", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data_list)

# for testing

list = [
    ["a", "b", "c"],
    ["a", "b", "c"],
    ["a", "b", "c"],
]

if __name__ == '__main__':
    write_data(list, "outputs/test.csv")
