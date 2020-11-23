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
fontsize = 24
alpha = 0.7
ms = 4
capsize = 3
marker = 'o'
linewidth = 2.0
figsize = (7, 7)
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
    xmin = 0
    xmax = 3
    ymin = -0.1
    ymax = 0.6

    xticklabels = [xmin, xmax]
    yticklabels = [ymin, ymax]
    fig, ax = plot_setup(figsize, linewidth, xmin, xmax, ymin, ymax, xticklabels, xticklabels, yticklabels,
                           yticklabels, fontsize)
    ax.set_xlabel('Re (Z) / Ohm', fontsize=fontsize + 1)
    ax.set_ylabel('Im (Z) / Ohm', fontsize=fontsize + 1)

    directory = 'data/eis/'
    Ts = ['25']
    cells = ['01', '02', '03', '04']
    states = ['I']
    n = 60
    data = []


    for cell in cells:
        for T in Ts:
            for state in states:

                file = 'EIS_state_{}_{}C{}.txt'.format(state, T, cell)
                label = 'Cell {}'.format(cell, state)

                df = pd.read_csv(directory + file, delimiter='\t')

                for i in range(1):
                    re_z = df['           Re(Z)/Ohm'].loc[i*n:int((i+1)*n - 1)].to_numpy()
                    im_z = df['  -Im(Z)/Ohm'].loc[i*n:int((i+1)*n - 1)].to_numpy()

                    ax.plot(re_z, im_z, marker='o', ms=ms, alpha=alpha, label=label)

    ax.legend(fontsize=fontsize - 4, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=1, frameon=True,
               handlelength=handlelength, handletextpad=handletextpad, labelspacing=labelspacing)
    fig.tight_layout()
    fig.savefig('eis_spectra_cell.png', dpi=400)
    """
    for cell in cells:
        for T in Ts:
            for state in states:

                file = 'EIS_state_{}_{}C{}.txt'.format(state, T, cell)

                df = pd.read_csv(directory + file, delimiter='\t')

                for i in range(1, 301, 50):
                    label = 'Cycle {}'.format(i)
                    re_z = df['           Re(Z)/Ohm'].loc[i * n:int((i + 1) * n - 1)].to_numpy()
                    im_z = df['  -Im(Z)/Ohm'].loc[i * n:int((i + 1) * n - 1)].to_numpy()

                    ax.plot(re_z, im_z, marker='o', ms=ms, alpha=alpha, linewidth=linewidth, label=label)

    ax.legend(fontsize=fontsize - 4, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, frameon=True,
              handlelength=handlelength, handletextpad=handletextpad, labelspacing=labelspacing)
    fig.tight_layout()
    fig.savefig('eis_spectra_cycle.png', dpi=400)
    """




if __name__ == '__main__':
    main()