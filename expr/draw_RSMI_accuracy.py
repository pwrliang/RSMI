import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
import os
import numpy as np
import sys
import re
import pandas as pd


def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def get_points(prefix, dist, n_points_list):
    accuracy_all = []
    training_time = []
    acc_accesses = []
    model_accesses = []

    for n_points in n_points_list:
        path = prefix + "/Accuracy_{dist}_{n_points}.log".format(dist=dist, n_points=n_points)
        print(path)
        with open(path, 'r') as fi:
            for line in fi:
                m = re.search(r"window_query accuracy: (.*?)$", line)
                if m is not None:
                    accuracy_all.append(float(m.groups()[0]))
                m = re.search(r"build time: (\d+) s$", line)
                if m is not None:
                    training_time.append(int(m.groups()[0]))
                m = re.search(r"^RSMI::acc_window_query page_access: (.*?)$", line)
                if m is not None:
                    acc_accesses.append(float(m.groups()[0]))
                m = re.search(r"^window_query page_access: (.*?)$", line)
                if m is not None:
                    model_accesses.append(float(m.groups()[0]))

    return np.asarray(accuracy_all), np.asarray(training_time), \
        np.asarray(acc_accesses), np.asarray(model_accesses)


def scale_size(size_list, k_scale=1024):
    return tuple(str(int(kb)) + "K" if kb < k_scale else str(int(kb / k_scale)) + "M" for kb in
                 np.asarray(size_list) / k_scale)


def draw_query_size(prefix, ):
    titles = ("(a) Uniform Dist. Recall", "(b) Norm Dist. Recall",
              "(c) Uniform Dist. Training Time", "(d) Norm Dist. Training Time")
    query_size_list = (1000000, 3000000, 5000000, 7000000, 9000000,)
    loc = [x for x in range(len(query_size_list))]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.5, 4.7))

    accuracy_uniform, time_uniform, acc_accesses_uniform, model_accesses_uniform = get_points(prefix, "uniform",
                                                                                              query_size_list)
    accuracy_normal, time_normal, acc_accesses_normal, model_accesses_normal = get_points(prefix, "normal",
                                                                                          query_size_list)

    ax_uniform_accuracy, ax_normal_accuracy = axes[0]
    ax_uniform_accesses, ax_normal_accesses = axes[1]

    ax_uniform_accuracy.plot(loc, accuracy_uniform, marker='o', label='Uniform distribution', color='red')
    ax_normal_accuracy.plot(loc, accuracy_normal, marker='o', label='Normal distribution', color='red')

    ax_uniform_accesses.plot(loc, acc_accesses_uniform, marker='x', label='Tree-based Method', color='blue')
    ax_uniform_accesses.plot(loc, model_accesses_uniform, marker='o', label='ML-based Method', color='red')

    ax_normal_accesses.plot(loc, acc_accesses_normal, marker='x', label='Tree-based Method', color='blue')
    ax_normal_accesses.plot(loc, model_accesses_normal, marker='o', label='ML-based Method', color='red')

    for i in range(2):
        for j in range(2):
            ax = axes[i][j]
            ax.set_xticks(loc, scale_size(query_size_list, 1000), rotation=0)
            ax.legend(loc='upper right', ncol=1, handletextpad=0.2, columnspacing=0.8,
                      fontsize='medium', borderaxespad=1, borderpad=0, frameon=False)
            ax.set_title(titles[i * 2 + j], verticalalignment="top")
            ax.autoscale(tight=True)
            ax.margins(x=0.1, y=0.5)
            ax.set_xlabel(xlabel='Number of Points')
            if i == 0:
                ax.set_ylabel(ylabel='Recall', labelpad=1)
            else:
                ax.set_ylabel(ylabel='Page Accesses', labelpad=1)
            ylim = list(ax.get_ylim())
            ylim[0] = 0
            ylim[1] = ylim[1] * 1.0
            ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(prefix, '../', 'RSMI_accuracy.png'), format='png',
                bbox_inches='tight')
    plt.show()


def draw_training_time():
    titles = ("(a) Uniform Dist. Training Time", "(b) Norm Dist. Training Time")
    query_size_list = (1000000, 3000000, 5000000, 7000000, 9000000,)
    loc = [x for x in range(len(query_size_list))]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 3.5))

    _, train_time_uniform_complex, _, _ = get_points("complex_net/", "uniform",
                                                     query_size_list)
    _, train_time_normal_complex, _, _ = get_points("complex_net/", "normal",
                                                    query_size_list)

    _, train_time_uniform_origin, _, _ = get_points("original_accuracy/", "uniform",
                                                    query_size_list)
    _, train_time_normal_origin, _, _ = get_points("original_accuracy/", "normal",
                                                   query_size_list)

    train_uniform, train_normal = axes

    train_uniform.plot(loc, train_time_uniform_complex, marker='o', label='HRSMI Uniform Dist.', color='blue')
    train_uniform.plot(loc, train_time_uniform_origin, marker='o', label='RSMI Uniform Dist.', color='red')

    train_normal.plot(loc, train_time_normal_complex, marker='o', label='HRSMI Normal Dist.', color='blue')
    train_normal.plot(loc, train_time_normal_origin, marker='o', label='RSMI Normal Dist.', color='red')

    for i in range(2):
        ax = axes[i]
        ax.set_xticks(loc, scale_size(query_size_list, 1000), rotation=0)
        ax.legend(loc='upper right', ncol=1, handletextpad=0.2, columnspacing=0.8,
                  fontsize='medium', borderaxespad=1, borderpad=0, frameon=False)
        ax.set_title(titles[i], verticalalignment="top")
        ax.autoscale(tight=True)
        ax.margins(x=0.1, y=0.5)
        ax.set_xlabel(xlabel='Number of Points')
        ax.set_ylabel(ylabel='Training Time (s)', labelpad=1)
        ylim = list(ax.get_ylim())
        ylim[0] = 0
        ylim[1] = ylim[1] * 1.0
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(prefix, '../', 'RSMI_train.png'), format='png',
                bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dir = os.path.dirname(sys.argv[0])
    prefix = os.path.join(dir, "new_accuracy/")
    draw_query_size(prefix, )
    # draw_training_time()
