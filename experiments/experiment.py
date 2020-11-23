
from PyExpLabSys.battery_utils.potentiostats import MPG2
from PyExpLabSys.battery_utils.experiment_utils import cycle_cell, reward_function, state_function, actor_function, critic_function, loss_function

# TODO: decide on parameters - initial charging protocol, restrictions on charging protocol, threshold current,
#  min and max voltage, parameters in the reward function,

def run_experiment():

    # Experiment parameters
    n_cycles = 10
    stop = False
    channel = 0

    # Specify the initial protocol
    protocol = [1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0]

    # Start with a single channel
    ip_address = '192.168.0.257'

    # Identify the potentiostat
    mpg2 = MPG2(ip_address)

    initial_capacity = 0
    previous_capacity = 0

    for cycle in range(n_cycles):
        t_charge, current_capacity, observations = cycle_cell(mpg2, channel, protocol)

        if cycle == 0:
            initial_capacity = current_capacity
            previous_capacity = current_capacity

        # Stage 5: Compute the reward received in this cycle and determine if end of episode for cell
        reward, stop = reward_function(t_charge, current_capacity, previous_capacity, initial_capacity, cycle)
        previous_capacity = current_capacity

        # Stage 6: Calculate the new state from the discharge curve features and the EIS spectrum after discharge
        # TODO: Write the state function!
        state = state_function(observations)

        # Stage 6: Compute the next charging protocol
        # TODO: Write the actor function! Interweave with DDPG now.
        new_protocol = actor_function(state)

        # Stage 7: Compute the action-value function and evaluate the loss.
        value = critic_function(state, new_protocol)
        loss = loss_function(value, reward)
















