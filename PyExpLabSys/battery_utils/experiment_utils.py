from __future__ import print_function
from pprint import pprint
import time

import numpy
from scipy import integrate

from PyExpLabSys.PyExpLabSys.drivers.bio_logic import OCV, CV, CA, GEIS, CPLimit
from PyExpLabSys.battery_utils.potentiostats import MPG2

import pdb

# Build charging protocol
def sequence_builder(protocol):
    v_max = 4.2
    v_min = 1.0

    # Set up the charging protocol
    cp1_ = CPLimit(current_step=(protocol[0],), test1_value=(protocol[1],),
                   test2_value=(v_min - 0.1,))
    ca1_ = CA(voltage_step=(protocol[1],), duration_step=(protocol[2],))
    cp2_ = CPLimit(current_step=(protocol[3],), test1_value=(protocol[4],),
                   test2_value=(v_min - 0.1,))
    ca2_ = CA(voltage_step=(protocol[4],), duration_step=(protocol[5],))
    cp3_ = CPLimit(current_step=(protocol[6],), test1_value=(v_max,),
                   test2_value=(v_min - 0.1,))
    ca3_ = CA(voltage_step=(v_max,), duration_step=(1000.0,))

    return [cp1_, ca1_, cp2_, ca2_, cp3_, ca3_]

def cycle_cell(potentiostat, channel, protocol):
    """

    :param potentiostat: the potentiostat to connect to
    :param channel: (int) channel to load techniques onto
    :param protocol: (array) [i0, v0, t0, i1, v1, t1, i2]
    :return:
    """

    # Stage 1: Charge the cell and measure the time taken to charge and the capacity of the cell.
    # TODO: Add ability to save the data from this process - write to file. Do for all stages
    t_charge, current_capacity = charge_cell(potentiostat, channel, protocol)

    # Stage 2: Record the EIS spectrum (galvanostatic) and extract the relevant parameters.
    re_z_charge, im_z_charge = measure_eis(potentiostat, channel)

    # Stage 3: Discharge the cell at a constant pre-specified C rate and extract relevant parameters
    discharge_features = discharge_cell(potentiostat, channel)

    # Stage 4: Record the EIS spectrum and extract relevant parameters. Write to file as well.
    # TODO: PCA for extracting parameters.
    re_z_discharge, im_z_discharge = measure_eis(potentiostat, channel)

    # Stage 5: Compute the reward received in this cycle
    reward = reward_function(t_charge, current_capacity, previous_capacity, initial_capacity, cycle_number)

    # Stage 6: Calculate the new state from the discharge curve features and the EIS spectrum after discharge
    # TODO: Write the state function!
    state = state_function(discharge_features, re_z_discharge, im_z_discharge)

    # Stage 6: Compute the next charging protocol
    # TODO: Write the actor function! Interweave with DDPG now.
    new_protocol = actor_function(state)

    # Stage 7: Compute the action-value function and evaluate the loss.


def reward_function(t_charge, current_capacity, previous_capacity, initial_capacity, cycle_number):
    A = 1
    B = 1
    extra_penalty = -100.0
    threshold = 0.8
    min_n_cycles = 10

    fast_charge_reward = 1 / t_charge
    degradation_penalty = (current_capacity - previous_capacity) / initial_capacity

    reward = fast_charge_reward + degradation_penalty

    if current_capacity / initial_capacity < threshold:
        if cycle_number > min_n_cycles:
            reward += 0
        if cycle_number < min_n_cycles:
            reward += extra_penalty

    return reward


# TODO: extract time to charge (time at which the current drops to less than a specified threshold value)
#       this will be an input to the reward function (need to write this function - i.e. given the time and capacity reduction what is the threshold)
#       also need to extract the capacity reduction (both cycle to cycle and reduction from the start -
#       so need to store both the last capacity and the initial capacity)
#       also need to write function to compute the new state from a) the discharge curve - second derivative / change from cycle to cycle ?!
#       and b) the EIS curve after discharge - look at Yunwei data - see if 5 PCA components captures enough information about spectrum and use that??
#       i.e. after dimensionality reduction can use the EIS features.
#       Then incorporate the RL algorithm here - lots of github code online...



def measure_eis(potentiostat, channel):
    """
    Measure the EIS Spectrum on a specific channel
    :param potentiostat:
    :param channel:
    :return:
    """

    # TODO: Choose these parameters
    delta_t = 2
    initial_frequency = 2.0e-2
    final_frequency = 20.0e3
    frequency_number = 60

    # Connect to potentiostat
    potentiostat.connect()

    # Set up measurement of EIS spectrum when fully charged
    # TODO: Choose these parameters
    technique = GEIS(vs_initial=False, vs_final=False, initial_current_step=0.1,
                     final_current_step=0.1,
                     duration_step=1.0,
                     step_number=3,
                     final_frequency=final_frequency,
                     initial_frequency=initial_frequency,
                     frequency_number=frequency_number,
                     sweep=False,
                     average_n_times=2,
                     I_range='KBIO_IRANGE_1mA')

    # Load techniques onto channel
    potentiostat.load_technique(channel, technique, True, True)

    # Start channel! Measure the EIS.
    potentiostat.start_channel(channel)

    try:
        while True:
            # Every delta_t seconds we get data from the channel
            time.sleep(delta_t)

            data_out = potentiostat.get_data(channel)

            # Stop when complete spectrum measured
            if data_out is None:
                break

            # TODO: Need to ascertain what this data looks like... how to process it to extract relevant features

            print('Technique', data_out.technique)
            print('Process index', data_out.process)
            if data_out.process == 0:
                print('time', data_out.time)
                print('Ewe', data_out.Ewe)
                print('I', data_out.I)
                print('step', data_out.step)
            else:
                print('freq', data_out.freq)
                print('abs_Ewe', data_out.abs_Ewe)
                print('abs_I', data_out.abs_I)
                print('Phase_Zwe', data_out.Phase_Zwe)
                print('Ewe', data_out.Ewe)
                print('I', data_out.I)
                print('abs_Ece', data_out.abs_Ece)
                print('abs_Ice', data_out.abs_Ice)
                print('Phase_Zce', data_out.Phase_Zce)
                print('Ece', data_out.Ece)
                print('step', data_out.step)

                # TODO: It is something along the lines of....
                Modulo_Z = data_out.abs_Ewe_numpy / data_out.abs_I_numpy
                Phase_Z = (2*numpy.pi()/180)*data_out.Phase_Zwe_numpy
                Re_Z = Modulo_Z * numpy.cos(Phase_Z)
                Im_Z = Modulo_Z * numpy.sin(Phase_Z)

                return Re_Z, Im_Z

    except KeyboardInterrupt:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    else:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()


def discharge_cell(potentiostat, channel):
    i_discharge = -0.1
    v_max = 4.2
    v_min = 1.0
    delta_t = 0.1

    # Connect to potentiostat
    potentiostat.connect()

    # Set up measurement of EIS spectrum when fully charged
    # TODO: Choose these parameters
    technique = CPLimit(current_step=(i_discharge,), test1_value=(v_max + 0.1,),
                        test2_value=(v_min,))

    # Load techniques onto channel
    potentiostat.load_technique(channel, technique, True, True)

    # Start channel! Discharge the cell
    potentiostat.start_channel(channel)

    ew_ = []
    ii_ = []
    ts_ = []

    try:
        while True:
            # Every delta_t seconds we get data from the channel
            time.sleep(delta_t)

            data_out = potentiostat.get_data(channel)

            # Stop when the cell is fully discharged (to the minimum potential)
            if data_out is None:
                break

            ew_ += data_out.Ewe
            ii_ += data_out.I
            ts_ += data_out.time

    except KeyboardInterrupt:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    else:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()

    observation = extract_features(ew_, ii_, ts_)

    return observation


def charge_cell(potentiostat, channel, protocol):
    """
    Option A action space
    :param potentiostat: the potentiostat to connect to
    :param channel: (int) channel to load techniques onto
    :param protocol: (array) [i0, v0, t0, i1, v1, t1, i2]
    :return:
    """
    # TODO: Decide these parameters
    i_threshold = 1.0e-4
    delta_t = 0.1

    potentiostat.connect()

    # Build sequence of techniques based on protocol
    techniques = sequence_builder(protocol)

    # Load techniques onto channel
    for i, technique in enumerate(techniques):
        if i == 0:
            potentiostat.load_technique(channel, technique, True, False)
        elif i == (len(techniques) - 1):
            potentiostat.load_technique(channel, technique, False, True)
        else:
            potentiostat.load_technique(channel, technique, False, False)

    # Start channel! i.e. charge potentiostat, measure EIS, discharge, remeasure EIS.
    potentiostat.start_channel(channel)

    try:
        while True:
            # Every delta_t seconds we get data from the channel
            time.sleep(delta_t)
            data_out = potentiostat.get_data(channel)
            data_current = potentiostat.get_current_values(channel)

            technique_index = data_out.technique_index
            # FIXME: Could also be data_current.I - probably need to debug
            i_current = data_current['I']
            t_elapsed = data_current['ElapsedTime']

            # The battery is charged if at the final stage of charging (technique 5 - 0-based indexing) the current dips below the prespecified
            # current threshold i_threshold
            if (technique_index == 5) and (i_current < i_threshold):
                # Pause channel
                potentiostat.stop_channel(channel)

                t_charge = t_elapsed

                # Now compute the charge stored in that time, and use to compute the amount of capacity reduction
                # FIXME: Might also be the case that this is not an array but a single value - need to check this
                #       In that case would need to set up a loop for adding current values to array at each time step
                times = data_out.time_numpy
                currents = data_out.I_numpy
                charges = integrate.cumtrapz(currents, times, initial=0)

                capacity = charges[-1]

                potentiostat.disconnect()

                return t_charge, capacity

    except KeyboardInterrupt:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()


if __name__ == '__main__':

    ip_address = '192.168.0.257'
    channel = 0

    # Connect to potentiostat
    mpg2 = MPG2(ip_address)

    # Get basic info
    basic(mpg2)

    # Test CV on specified channel
    run_charge(mpg2, channel, protocol)


    #current_values()
    #test_ocv_technique()
    #test_cp_technique()
    #test_ca_technique()
    #test_cva_technique()
    #test_speis_technique()
    #test_message()




