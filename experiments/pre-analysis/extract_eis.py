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
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

from utils.eis_utils import extract_features, plot_setup
from utils.rl_utils import to_tensor

import pdb

def general_sigmoid(x, a, b, c):
    return a / (1.0 + np.exp(b*(x-c)))


# We Exact GP Model.
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

column_map = {
            '0125': ['time', 'cycle number', 'ox/red', 'capacity'],
            '0225': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1', 'nan2', 'nan3'],
            '0325': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1'],
            '0425': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1'],
            '0525': ['time', 'cycle number', 'ox/red', 'capacity', 'nan1'],
            '0625': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1', 'nan2'],
            '0725': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity', 'nan1'],
            '0825': ['time', 'cycle number', 'ox/red', 'ewe', 'i', 'capacity'],
            '0135': ['time', 'cycle number', 'ox/red', 'capacity'],
            '0235': ['time', 'cycle number', 'ox/red', 'capacity'],
            '0145': ['time', 'cycle number', 'ox/red', 'capacity'],
            '0245': ['time', 'cycle number', 'ox/red', 'capacity']
            }

# Hyperparameters for the GP
lr = 0.1
n_iterations = 2500
fixed_noise = 25.0

def main():

    dir_eis = '../data/eis/'
    dir_cap = '../data/capacity/'

    # Whether to predict capacity after charge (True) or after discharge (False)
    charge_cap = False
    Ts = ['25', '25', '25', '25', '25', '25']
    state = 'V'
    cells = ['02', '03', '04', '06', '07', '08']
    to_train = [True, True, True, True, False, False]

    # Number of EIS frequencies
    nf = 60
    # Capacity reduction at which we declare battery dead - typically around 80%
    boundary = 0.8


    # Form the training and test set
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for T, cell, train in zip(Ts, cells, to_train):
        print('\n\nCell {}'.format(cell))
        file_eis = 'EIS_state_{}_{}C{}.txt'.format(state, T, cell)
        file_cap = 'Data_Capacity_{}C{}.txt'.format(T, cell)

        # Load relevant EIS spectrum and capacities
        df_eis = pd.read_csv(dir_eis + file_eis, delimiter='\t')
        df_cap = pd.read_csv(dir_cap + file_cap, delimiter='\t')

        df_eis.columns = ['time/s', 'cycle number', 'freq', 're_z', '-im_z', 'mod_z', 'phase_z']
        df_cap.columns = column_map[cell + T]
        X_cell = []
        y_cell = []

        capacities = []
        cycles = []

        life = 2.0*int(df_eis.shape[0] / nf)

        for cycle in range(1, int(df_eis.shape[0] / nf)):
            # Extract 'x' - i.e. relevant features of the EIS spectrum
            re_z = df_eis['re_z'].loc[cycle*nf:int((cycle+1)*nf - 1)].to_numpy()
            im_z = df_eis['-im_z'].loc[cycle*nf:int((cycle+1)*nf - 1)].to_numpy()
            log_omega = np.log10(df_eis['freq'].loc[cycle*nf:int((cycle+1)*nf - 1)].to_numpy())

            # Either use the concatenation of Re(z) and Im(z) or extract features
            #features = np.hstack([re_z, im_z])
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

            if (len(capacities) > 0) and (cap.shape[0] == 1):
                if cap[0] < boundary * capacities[0]:
                    life = 2.0 * cycle
                    break

            if cap.shape[0] == 1:
                # Extract features of the discharge curve
                t_min = df_cap.loc[(df_cap['ox/red'] == 1) & (df_cap['ox/red'].shift(-1) == 0) &
                                   (df_cap['cycle number'] == cycle)]['time'].to_numpy()
                t_min = t_min[0]
                t_max = df_cap.loc[(df_cap['ox/red'] == 0) & (df_cap['ox/red'].shift(-1) == 1) &
                                   (df_cap['cycle number'] == cycle)]['time'].to_numpy()
                t_max = t_max[0]
                df_cycle = df_cap.loc[(df_cap['time'] >= t_min) & (df_cap['time'] <= t_max)]
                time = df_cycle['time'].to_numpy() - t_min
                capacity = df_cycle['capacity'].to_numpy()
                voltage = df_cycle['ewe'].to_numpy()
                popt, pcov = curve_fit(general_sigmoid, voltage[1:], capacity[1:])

                X_cell.append(np.hstack([popt, features]))
                cycles.append(2.0 * cycle)
                capacities.append(cap[0])
                #y_cell.append(cap[0])

        cycles = np.array(cycles)

        # Remaining useful life (RUL) is the value to be predicted
        y_cell = life - cycles

        if train:
            X_train.append(np.array(X_cell))
            y_train.append(np.array(y_cell))

        else:
            X_test.append(np.array(X_cell))
            y_test.append(np.array(y_cell))

    y_train = np.hstack(y_train)
    X_train = np.vstack(X_train)

    # Reformat training data (i.e. stack and covert to torch.tensor)
    y_train = to_tensor(y_train)

    # Standardise the data (zero mean, unit variance)
    scaler = StandardScaler().fit(X_train)
    X_train = to_tensor(scaler.transform(X_train))

    # Set up the GP model - use Exact GP and Gaussian Likelihood
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed_noise*torch.ones(X_train.shape[0]))
    model = ExactGPModel(X_train, y_train, likelihood)

    # Reload partially trained model if it exists
    #if os.path.isfile('model_state.pth'):
    #    state_dict = torch.load('model_state.pth')
     #   model.load_state_dict(state_dict)

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
        predictions = likelihood(model(X_train), noise=fixed_noise*torch.ones(X_train.shape[0]))
        y = y_train.numpy()
        mn = predictions.mean
        var = predictions.variance

        lower = mn.numpy() - np.sqrt(var.numpy())
        upper = mn.numpy() + np.sqrt(var.numpy())

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(y, mn.numpy(), c='blue', label='mean')
        ax.fill_between(y, lower, upper, alpha=0.5)
        ax.set_xlabel('Actual RUL')
        ax.set_ylabel('Predicted RUL')
        plt.savefig('train_rul_discharge_x.png', dpi=400)

        for j in range(2):
            # Make predictions for the 4 test cells - first need to transform the input.
            y = y_test[j]

            x = to_tensor(scaler.transform(X_test[j]))

            # Make predictions for y_test
            predictions = likelihood(model(x), noise=fixed_noise*torch.ones(x.shape[0]))
            mn = predictions.mean
            var = predictions.variance

            lower = mn.numpy() - np.sqrt(var.numpy())
            upper = mn.numpy() + np.sqrt(var.numpy())

            # Initialize plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.scatter(y, mn.numpy(), c='blue', label='mean')
            ax.fill_between(y, lower, upper, alpha=0.5)
            ax.set_xlabel('Actual RUL')
            ax.set_ylabel('Predicted RUL')
            plt.savefig('test{}_rul_discharge_x.png'.format(j), dpi=400)


if __name__ == '__main__':
    main()