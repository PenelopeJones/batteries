import sys
sys.path.append('../../')
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

from utils.eis_utils import extract_features, plot_setup

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

# set up figures
xmin = 0
xmax = 3
ymin = -0.1
ymax = 0.6
xticklabels = [xmin, xmax]
yticklabels = [ymin, ymax]
xlabel = 'Re (Z) / Ohm'
ylabel = 'Im (Z) / Ohm'

column_map = {
            '01': ['time', 'cycle number', 'ox/red', 'capacity'],
            '02': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1', 'nan2', 'nan3'],
            '03': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1'],
            '04': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1'],
            '05': ['time', 'cycle number', 'ox/red', 'capacity', 'nan1'],
            '06': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1', 'nan2'],
            '07': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1'],
            '08': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity']
            }


def main():

    # Create figure?
    #fig, ax = plot_setup(figsize, linewidth, xmin, xmax, ymin, ymax, xticklabels, xticklabels, yticklabels,
                           #yticklabels, fontsize)
    #ax.set_xlabel(xlabel, fontsize=fontsize + 1)
    #ax.set_ylabel(ylabel, fontsize=fontsize + 1)

    dir_eis = '../data/eis/'
    dir_cap = '../data/capacity/'

    # Whether to predict capacity after charge (True) or after discharge (False)
    charge_cap = False
    T = '25'
    state = 'IX'
    cells = ['01', '02', '03', '04', '05', '06', '07', '08']
    to_train = [True, True, True, True, False, False, False, False]
    n = 60

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for cell, train in zip(cells, to_train):
        file_eis = 'EIS_state_{}_{}C{}.txt'.format(state, T, cell)
        file_cap = 'Data_Capacity_{}C{}.txt'.format(T, cell)
        label = 'Cell {}'.format(cell, state)
        df_eis = pd.read_csv(dir_eis + file_eis, delimiter='\t')
        df_cap = pd.read_csv(dir_cap + file_cap, delimiter='\t')
        df_eis.columns = ['time/s', 'cycle number', 'freq', 're_z', '-im_z', 'mod_z', 'phase_z']
        df_cap.columns = column_map[cell]

        X_cell = []
        y_cell = []

        print('\n\nCell {}'.format(cell))
        for cycle in range(int(df_eis.shape[0] / n)):
            if cycle % 10 == 0:
                print('Cycle {}'.format(cycle))
            # Extract 'x' - i.e. relevant features of the EIS spectrum
            re_z = df_eis['re_z'].loc[cycle*n:int((cycle+1)*n - 1)].to_numpy()
            im_z = df_eis['-im_z'].loc[cycle*n:int((cycle+1)*n - 1)].to_numpy()
            log_omega = np.log10(df_eis['freq'].loc[cycle*n:int((cycle+1)*n - 1)].to_numpy())
            features = extract_features(re_z, im_z, log_omega)
            if features is None:
                continue

            # Extract 'y' - i.e. the capacity after discharge or charge
            if charge_cap:
                cap = df_cap.loc[(df_cap['ox/red'] == 1) & (df_cap['ox/red'].shift(-1) == 0) &
                                     (df_cap['cycle number'] == cycle)]['capacity'].to_numpy()
            else:
                cap = df_cap.loc[(df_cap['ox/red'] == 0) & (df_cap['ox/red'].shift(-1) == 1) &
                                     (df_cap['cycle number'] == cycle)]['capacity'].to_numpy()

            if cap.shape[0] != 1:
                continue
            else:
                X_cell.append(features)
                y_cell.append(cap[0])

        if train:
            X_train.append(np.array(X_cell))
            y_train.append(np.array(y_cell))

        else:
            X_test.append(np.array(X_cell))
            y_test.append(np.array(y_cell))

    # Training set created
    y_train = np.hstack(y_train)
    X_train = np.vstack(X_train)

    y_test = np.hstack(y_test)
    X_test = np.vstack(X_test)

    pdb.set_trace()
    

if __name__ == '__main__':
    main()