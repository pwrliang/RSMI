import matplotlib.pyplot as plt
import os

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


def read_points(path):
    x_list = []
    y_list = []
    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()
            arr = line.split(' ')
            x = float(arr[0])
            y = float(arr[1])
            x_list.append(x)
            y_list.append(y)
    return x_list, y_list




for f in os.listdir("/Users/liang/_users_PAS0350_geng161_Datasets_normal_normal_5000000_1_2_.csv"):
    f = f.strip()
    if len(f) == 0:
        continue
    x_list, y_list = read_points("/Users/liang/_users_PAS0350_geng161_Datasets_normal_normal_5000000_1_2_.csv/" + f)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    ax.plot(x_list, y_list, '.', markersize=1)
    fig.savefig(f + ".png", format='png', bbox_inches='tight', pad_inches=0)
