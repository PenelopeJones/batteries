"""Integration tests for the biologic SP-150 driver"""
from __future__ import print_function
from pprint import pprint
import time
import sys

sys.path.append('../')
from PyExpLabSys.PyExpLabSys.drivers.bio_logic import OCV, CV, MPG2


def basic(potentiostat, channel):
    """ Main method for tests """
    print('## Device info before connect:', potentiostat.device_info)

    print('\n## Lib version:', potentiostat.get_lib_version())
    dev_info = potentiostat.connect()
    print('\n## Connect returned device info:')
    pprint(dev_info)

    # Information about whether the channels are plugged
    channels = potentiostat.get_channels_plugged()
    print('\n## Channels plugged:', channels)
    for index in range(10):
        print('Channel {} plugged:'.format(index),
              potentiostat.is_channel_plugged(index))

    print('\n## Device info:')
    pprint(potentiostat.device_info)

    channel_info = potentiostat.get_channel_infos(channel)
    print('\n## Channel {} info'.format(channel))
    pprint(channel_info)

    potentiostat.disconnect()
    print('\n## Disconnect and test done')

def test_ocv_technique(potentiostat, channel):
    """Test the OCV technique"""
    potentiostat.connect()
    ocv = OCV(rest_time_T=0.2,
              record_every_dE=10.0,
              record_every_dT=0.01)
    potentiostat.load_technique(channel, ocv)
    potentiostat.start_channel(channel)
    try:
        time.sleep(0.1)
        while True:
            data_out = potentiostat.get_data(channel)
            if data_out is None:
                break
            print(data_out.Ewe)
            print(data_out.Ewe_numpy)
            time.sleep(0.1)
    except KeyboardInterrupt:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    else:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()



def test_cv_technique(potentiostat, channel):
    """Test the CV technique"""
    import matplotlib.pyplot as plt
    potentiostat.connect()
    cv_ = CV(vs_initial=(True,) * 5,
             voltage_step=(0.0, 0.5, -0.7, 0.0, 0.0),
             scan_rate=(10.0,) * 5,
             record_every_dE=0.01,
             N_cycles=3)
    potentiostat.load_technique(channel, cv_)

    potentiostat.start_channel(channel)
    ew_ = []
    ii_ = []
    try:
        while True:
            time.sleep(0.1)
            data_out = potentiostat.get_data(channel)
            if data_out is None:
                break
            print(data_out.technique)
            print('Ewe:', data_out.Ewe)
            print('I:', data_out.I)
            ew_ += data_out.Ewe
            ii_ += data_out.I
            print('cycle:', data_out.cycle)
    except KeyboardInterrupt:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    else:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    plt.plot(ew_, ii_)
    plt.show()
    print('end')



if __name__ == '__main__':

    ip_address = b"10.64.2.254"
    channel = 7

    print('Connecting...')
    # Connect to potentiostat
    mpg2 = MPG2(ip_address)

    # Get basic info
    basic(mpg2)

    # Test OCV technique
    test_ocv_technique(mpg2, channel)

    # Test CV on specified channel
    #test_cv_technique(mpg2, channel)
