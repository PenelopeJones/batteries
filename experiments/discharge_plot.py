import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

import matplotlib.colors as colors
from collections import OrderedDict

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import pdb

color1 = '#2A00FB'
color2 = '#F18400'
color3 = 'C2'
color4 = 'C3'
fontsize = 30
alpha = 0.7
ms = 4
capsize = 3
marker = 'o'
linewidth = 3.0
figsize = (9, 7)
#loc = 'upper left'
#bbox_to_anchor = (0.0, 0.5, 0.3, 0.5)
loc = 'upper left'
bbox_to_anchor = (0.0, 0.5, 0.3, 0.5)
labelspacing = 0.35
ncol = 2
handlelength = 1.0
handletextpad = 0.5

def plot_setup(figsize, linewidth, xmin, xmax, ymin, ymax, xticks, xticklabels, yticks, yticklabels, fontsize):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(linewidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_visible(False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.set_yticklabels(yticklabels, fontsize=fontsize)
    return fig, ax

def main():
    # set up figures

    #xmin = 0.1
    #xmax = 2.3
    #ymin = -0.1
    #ymax = 0.6
    """
    xmin = 0
    xmax = 1
    ymin = -0.05
    ymax = 1.1

    xticks = [xmin, 0.5, xmax]
    yticks = [-0.0346055, 0.982312]
    xticklabels = [100, 50, 0]
    yticklabels = [r'$V_{min}$', r'$V_{max}$']
    fig, ax = plot_setup(figsize, linewidth, xmin, xmax, ymin, ymax, xticks, xticklabels, yticks,
                           yticklabels, fontsize)
    ax.set_xlabel('Capacity retained / %', fontsize=fontsize + 1)
    ax.set_ylabel('Voltage / V', fontsize=fontsize + 1)

    directory = 'data/examples/'

    file = 'discharge_curve_example.csv'

    df = pd.read_csv(directory + file, header=None)

    pdb.set_trace()

    c = df[0].to_numpy()
    voltage = df[1].to_numpy()

    ax.plot(c, voltage, color=color1, alpha=alpha, linewidth=linewidth)

    fig.tight_layout()
    fig.savefig('discharge_curve_example.png', dpi=400)
    """
    xmin = 0
    xmax = 200
    ymin = 60
    ymax = 105

    xticks = [xmin, 100, xmax]
    yticks = [60, 80, 100]
    xticklabels = [0, 100, 200]
    yticklabels = [60, 80, 100]
    colors = [color4, color3]
    fnames = ['Fastcharge.csv',  'gbatteries.csv']
    labels = ['Conventional', 'G-batteries']
    fig, ax = plot_setup(figsize, linewidth, xmin, xmax, ymin, ymax, xticks, xticklabels, yticks,
                           yticklabels, fontsize)
    ax.set_xlabel('Cycle number', fontsize=fontsize + 1)
    ax.set_ylabel('Capacity / %', fontsize=fontsize + 1)

    directory = 'data/examples/'

    for file, color, label in zip(fnames, colors, labels):

        df = pd.read_csv(directory + file, header=None)
        c = df[0].to_numpy()
        idx = c.argsort()
        voltage = df[1].to_numpy()

        ax.plot(c[idx], voltage[idx], color=color, alpha=alpha, linewidth=linewidth, label=label)
    ax.legend(fontsize=fontsize - 4, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, frameon=False,
              handlelength=handlelength, handletextpad=handletextpad, labelspacing=labelspacing)
    fig.tight_layout()
    fig.savefig('gbatteries.png', dpi=400)

if __name__ == '__main__':
    main()