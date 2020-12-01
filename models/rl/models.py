import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    """
    weight initializer known from https://arxiv.org/abs/1502.01852
    :param size:
    :param fanin:
    :return:
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, w0=0.01):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        for dim in range(len(hidden_dims) + 1):
            if dim == 0:
                self.layers.append(nn.Linear(self.in_dim, hidden_dims[dim]))
            elif dim == len(hidden_dims):
                self.layers.append(nn.Linear(hidden_dims[-1], self.out_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[dim - 1], hidden_dims[dim]))

        self.initialise_weights(w0)

    def initialise_weights(self, w0):
        for i in range(len(self.layers) - 1):
            self.layers[i].weight.data = fanin_init(self.layers[i].weight.data.size())
        self.layers[-1].weight.data.uniform_(-w0, w0)

    def forward(self, state, action):
        """
        return critic Q(s,a)
        :param state: state [batch_size, state_dim]
        :param action: action [batch_size, action_dim]
        :return: [batch_size, 1]
        """
        assert len(state.shape) == len(action.shape) == 2

        # Input to the critic function is the concatenation of state and action
        x = torch.cat((state, action), dim=1)

        for i in range(len(self.layers) - 1):
            x = self.non_linearity(self.layers[i](x))

        return self.layers[-1](x)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, non_linearity=F.relu, w0=0.01):
        """
        :param in_dim: (int) Dimensionality of the input.
        :param out_dim: (int) Dimensionality of the output.
        :param hidden_dims: (list of ints) Architecture of the network.
        :param non_linearity: Non-linear activation function to apply after each linear transformation,
                                e.g. relu or tanh.
        """
        super().__init__()

        self.in_dim = state_dim
        self.out_dim = action_dim
        self.non_linearity = non_linearity
        self.layers = nn.ModuleList()

        for dim in range(len(hidden_dims) + 1):
            if dim == 0:
                self.layers.append(nn.Linear(self.in_dim, hidden_dims[dim]))
            elif dim == len(hidden_dims):
                self.layers.append(nn.Linear(hidden_dims[-1], self.out_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[dim - 1], hidden_dims[dim]))

        self.initialise_weights(w0)

    def initialise_weights(self, w0):
        for i in range(len(self.layers) - 1):
            self.layers[i].weight.data = fanin_init(self.layers[i].weight.data.size())
        self.layers[-1].weight.data.uniform_(-w0, w0)

    def forward(self, state):
        """
        Actor function pi(s)
        :param state: [batch_size, state_dim]
        :return: action [batch_size, action_dim]
        """
        assert len(state.shape) == 2, 'Input must be of shape [batch_size, state_dim].'

        for i in range(len(self.layers) - 1):
            state = self.non_linearity(self.layers[i](state))

        return self.layers[-1](state)


