from __future__ import print_function
import time

from PyExpLabSys.PyExpLabSys.drivers.bio_logic import OCV, CV, CA
from PyExpLabSys.battery_utils.potentiostats import MPG2
from PyExpLabSys.battery_utils.techniques import ModularPulse

def run_ocv(channel):
    """
    Test technique - find out the IP address of Forse group
    :return:
    """
    ip_address = '192.168.0.257'

    # Connect to potentiostat
    mpg2 = MPG2(ip_address)
    mpg2.connect()

    # Instantiate the technique. In this case, run OCV as a test
    technique = OCV(rest_time_T=0.2, record_every_dE=10.0, record_every_dT=0.01)

    # Load the technique onto desired channel of the potentiostat, and then start it
    mpg2.load_technique(channel, technique)
    mpg2.start_channel(channel)

    time.sleep(0.1)

    while True:
        # Get currently available data on specified channel
        data_out = mpg2.get_data(channel)

        if data_out is None:
            break

    mpg2.stop_channel(channel)


def run_charge(channel, value_step, mode_step):
    ip_address = '192.168.0.257'

    # Connect to potentiostat
    mpg2 = MPG2(ip_address)
    mpg2.connect()

    # Instantiate the technique. In this case, run OCV as a test
    technique = OCV(rest_time_T=0.2, record_every_dE=10.0, record_every_dT=0.01)

    # Load the technique onto desired channel of the potentiostat, and then start it
    mpg2.load_technique(channel, technique)
    mpg2.start_channel(channel)

    time.sleep(0.1)

    while True:
        # Get currently available data on specified channel
        data_out = mpg2.get_data(channel)

        if data_out is None:
            break

    mpg2.stop_channel(channel)



