# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import utils

data_path = 'E:\\code\\smart_cite_con\\data\\bertData1_20201203.csv'


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_list = utils.read_data(data_path)
    for i in data_list:
        print(len(i[0]), " ", len(i[1]))
    # print(data_list)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
