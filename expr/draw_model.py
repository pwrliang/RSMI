import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import re

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


def read_points(path):
    x_list = []
    y_list = []
    mbr = None
    level = None
    index = None

    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()
            if len(line) == 0:
                continue
            r = re.match("^Mbr: (.*?) (.*?) (.*?) (.*?)$", line)
            if r is not None:
                x1 = float(r.groups()[0])
                x2 = float(r.groups()[1])
                y1 = float(r.groups()[2])
                y2 = float(r.groups()[3])
                mbr = (x1, y1, x2 - x1, y2 - y1)
            r = re.match("^Level: (\d+) Index: (\d+)$", line)
            if r is not None:
                level = int(r.groups()[0])
                index = int(r.groups()[1])

            arr = line.split(' ')
            if len(arr) == 2:
                x = float(arr[0])
                y = float(arr[1])
                x_list.append(x)
                y_list.append(y)
    return mbr, level, index, x_list, y_list


dir = "/Users/liang/CLionProjects/RSMI/expr/_users_PAS0350_geng161_Datasets_normal_normal_10000_1_2_.csv"
for f in os.listdir(dir):
    f = f.strip()
    if len(f) == 0:
        continue
    mbr, level, index, x_list, y_list = read_points(dir + "/" + f)
    x1, y1, w, h = mbr
    # print(mbr, level, index)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    ax.plot(x_list, y_list, '.', markersize=1)
    rect = Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect)
    fig.savefig(f + ".png", format='png', bbox_inches='tight', pad_inches=0)
