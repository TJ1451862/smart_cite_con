# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv

data_path = 'E:\\code\\smart_cite_con\\data\\bertData1_20201203.csv'


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    f = csv.reader(open(data_path, "r", encoding='utf-8'))
    for i in f:
        print(i)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
