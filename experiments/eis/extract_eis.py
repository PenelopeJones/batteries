import os
import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
import gpytorch
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

from utils.eis_utils import extract_features, plot_setup
from utils.rl_utils import to_tensor
from models.gp import ExactGPModel

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

# Hyperparameters for the GP
lr = 0.1
n_iterations = 500

# We Exact GP Model.
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main():

    dir_eis = '../data/eis/'
    dir_cap = '../data/capacity/'

    # Whether to predict capacity after charge (True) or after discharge (False)
    charge_cap = False
    T = '25'
    state = 'IX'
    cells = ['01', '02', '03', '04', '05', '06', '07', '08']
    to_train = [True, True, True, True, False, False, False, False]

    # Number of EIS frequencies
    nf = 60

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for cell, train in zip(cells, to_train):
        print('\n\nCell {}'.format(cell))
        file_eis = 'EIS_state_{}_{}C{}.txt'.format(state, T, cell)
        file_cap = 'Data_Capacity_{}C{}.txt'.format(T, cell)

        # Load relevant EIS spectrum and capacities
        df_eis = pd.read_csv(dir_eis + file_eis, delimiter='\t')
        df_cap = pd.read_csv(dir_cap + file_cap, delimiter='\t')
        df_eis.columns = ['time/s', 'cycle number', 'freq', 're_z', '-im_z', 'mod_z', 'phase_z']
        df_cap.columns = column_map[cell]

        X_cell = []
        y_cell = []

        for cycle in range(int(df_eis.shape[0] / nf)):
            # Extract 'x' - i.e. relevant features of the EIS spectrum
            re_z = df_eis['re_z'].loc[cycle*nf:int((cycle+1)*nf - 1)].to_numpy()
            im_z = df_eis['-im_z'].loc[cycle*nf:int((cycle+1)*nf - 1)].to_numpy()
            log_omega = np.log10(df_eis['freq'].loc[cycle*n:int((cycle+1)*nf - 1)].to_numpy())
            features = np.hstack([re_z, im_z])
            #features = extract_features(re_z, im_z, log_omega)
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

    # Reformat training data (i.e. stack and covert to torch.tensor)
    y_train = to_tensor(np.hstack(y_train))
    X_train = np.vstack(X_train)

    # Standardise the data (zero mean, unit variance)
    scaler = StandardScaler().fit(X_train)
    X_train = to_tensor(scaler.transform(X_train))

    # Set up the GP model - use Exact GP and Gaussian Likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)

    # Reload partially trained model if it exists
    if os.path.isfile('model_state.pth'):
        state_dict = torch.load('model_state.pth')
        model.load_state_dict(state_dict)

    # Use Adam optimiser for optimising hyperparameters
    optimiser = Adam(model.parameters(), lr=lr)

    # Loss is the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train the GP
    model.train()
    likelihood.train()

    for i in range(n_iterations):
        optimiser.zero_grad()

        # Output from model
        output = model(X_train)

        # Calculate the loss
        loss = -mll(output, y_train)
        loss.backward()

        if i % 50 == 0:
            print('Iteration {}/{} - Loss: {:.3f}'.format(i + 1, n_iterations, loss.item()))

        optimiser.step()

    # Save trained model
    torch.save(model.state_dict(), 'model_state.pth')

    # Make predictions
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # First check the predictions on training data
        predictions = likelihood(model(X_train))
        mn = predictions.mean
        var = predictions.variance
        lower = mn.numpy() - np.sqrt(var.numpy())
        upper = mn.numpy() + np.sqrt(var.numpy())
        cycles = np.arange(X_train.shape[0])

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # Plot training data as black stars
        ax.scatter(cycles, y_train.numpy(), c='red', label='observed')
        # Plot predictive means as blue line
        ax.scatter(cycles, mn.numpy(), c='blue', label='mean')
        ax.fill_between(cycles, lower, upper, alpha=0.5)
        plt.savefig('train.png', dpi=400)


        for j in range(4):
            # Make predictions for the 4 test cells - first need to transform the input.
            y = to_tensor(y_test[j])
            x = to_tensor(scaler.transform(X_test[j]))

            cycles = np.arange(y.shape[0])

            # Make predictions for y_test
            predictions = likelihood(model(x))
            mn = predictions.mean
            var = predictions.variance
            lower = mn.numpy() - np.sqrt(var.numpy())
            upper = mn.numpy() + np.sqrt(var.numpy())

            # Initialize plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            # Plot training data as black stars
            ax.scatter(cycles, y.numpy(), c='red', label='observed')
            # Plot predictive means as blue line
            ax.scatter(cycles, mn.numpy(), c='blue', label='mean')
            ax.fill_between(cycles, lower, upper, alpha=0.5)
            plt.savefig('test{}.png'.format(j), dpi=400)























if __name__ == '__main__':
    main()