import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from rl.models import Actor, Critic
from rl.memory import SequentialMemory
from rl.random_process import OrnsteinUhlenbeckProcess
from rl.rl_utils import hard_update, soft_update, to_numpy, to_tensor

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, state_dim, action_dim, actor_dims, critic_dims, alr=0.001, clr=0.001,
                 rm_size=10, window_length=10, ou_theta, ou_mu, ou_sigma, batch_size, tau, discount,
                 depsilon):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyperparameters
        self.batch_size = batch_size
        self.tau = tau
        self.discount = discount
        self.depsilon = depsilon
        self.epsilon = 1.0


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
        self.memory = SequentialMemory(limit=rm_size, window_length=window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=action_dim, theta=ou_theta,
                                                       mu=ou_mu, sigma=ou_sigma)

        # Most recent state and action
        self.s_t = None
        self.a_t = None
        self.is_training = True

        #
        if USE_CUDA:
            self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target.forward([to_tensor(next_state_batch, volatile=True),
                                            self.actor_target.forward(to_tensor(next_state_batch, volatile=True)),
                                            ])
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic.forward([to_tensor(state_batch), to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optimiser.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic.forward([
            to_tensor(state_batch),
            self.actor.forward(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimiser.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)




