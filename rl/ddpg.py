import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from rl.models import Actor, Critic
from rl.rl_utils import hard_update

class DDPG(object):
    def __init__(self, state_dim, action_dim, actor_dims, critic_dims, alr=0.001, clr=0.001):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor network
        self.actor = Actor(state_dim, action_dim, actor_dims)
        self.actor_target = Actor(state_dim, action_dim, actor_dims)
        hard_update(self.actor_target, self.actor)
        self.actor_optimiser = Adam(self.actor.parameters(), lr=alr)

        # Critic network
        self.critic = Critic(state_dim, action_dim, critic_dims)
        self.critic_target = Critic(state_dim, action_dim, critic_dims)
        hard_update(self.critic_target, self.critic)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=clr)

        # Replay buffer
        self.memory = 



