"""Integration tests for the biologic SP-150 driver"""
from __future__ import print_function
from pprint import pprint
import time
import sys

sys.path.append('../')
from PyExpLabSys.PyExpLabSys.drivers.bio_logic import OCV, CP, CA, CV, CVA, SPEIS, MPG2


def basic(potentiostat):
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

    channel_info = potentiostat.get_channel_infos(7)
    print('\n## Channel 7 info')
    pprint(channel_info)

    #print('\n## Load_firmware:', potentiostat.load_firmware(channels))

    print('\n## Message left in the queue:')
    while True:
        msg = potentiostat.get_message(7)
        if msg == '':
            break
        print(msg)

    potentiostat.disconnect()
    print('\n## Disconnect and test done')


def current_values(potentiostat, channel):
    """Test the current values method"""
    potentiostat.connect()
    current_values_ = potentiostat.get_current_values(channel)
    pprint(current_values_)
    potentiostat.disconnect()


def mess_with_techniques(potentiostat, channel=0):
    """Test adding techniques"""
    potentiostat.connect()
    ocv = OCV(rest_time_T=0.3,
              record_every_dE=10.0,
              record_every_dT=0.01)
    potentiostat.load_technique(channel, ocv, False, False)
    potentiostat.load_technique(channel, ocv, True, True)
    #potentiostat.load_technique(0, ocv, False, True)
    print(potentiostat.get_channel_infos(channel)['NbOfTechniques'])
    potentiostat.disconnect()


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


def test_cp_technique(potentiostat, channel):
    """Test the CP technique"""
    potentiostat.connect()
    cp_ = CP(current_step=(-1E-6, -10E-6, -100E-6),
             vs_initial=(False, False, False),
             duration_step=(2.0, 2.0, 2.0),
             record_every_dE=1.0,
             record_every_dT=1.0)
    potentiostat.load_technique(channel, cp_)
    #potentiostat.disconnect()
    #return

    potentiostat.start_channel(channel)
    try:
        while True:
            time.sleep(2)
            data_out = potentiostat.get_data(channel)
            if data_out is None:
                break
            #print(data_out.Ewe)
            print(data_out.I_numpy)
            print('NP:', data_out.cycle_numpy)
    except KeyboardInterrupt:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    else:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()


def test_ca_technique(potentiostat):
    """Test the CA technique"""
    potentiostat.connect()
    ca_ = CA(voltage_step=(0.01, 0.02, 0.03),
             vs_initial=(False, False, False),
             duration_step=(5.0, 5.0, 5.0),
             record_every_dI=1.0,
             record_every_dT=0.1)
    potentiostat.load_technique(0, ca_)
    #potentiostat.disconnect()
    #return

    potentiostat.start_channel(0)
    try:
        while True:
            time.sleep(5)
            data_out = potentiostat.get_data(0)
            if data_out is None:
                break
            print(data_out.technique)
            print('Ewe:', data_out.Ewe)
            print('I:', data_out.I)
            print('cycle:', data_out.cycle)
    except KeyboardInterrupt:
        potentiostat.stop_channel(0)
        potentiostat.disconnect()
    else:
        potentiostat.stop_channel(0)
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


def test_cva_technique(potentiostat, channel):
    """Test the CVA technique"""
    import matplotlib.pyplot as plt
    potentiostat.connect()
    print('kk')
    cva = CVA(
        vs_initial_scan=(False,) * 4,
        voltage_scan=(0.0, 0.2, -0.2, 0.0),
        scan_rate=(50.0,) * 4,
        vs_initial_step=(False,) * 2,
        voltage_step=(0.1,) * 2,
        duration_step=(1.0,) * 2,
    )
    potentiostat.load_technique(channel, cva)

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
            print('time:', data_out.time,
                  'numpy', data_out.time_numpy, data_out.time_numpy.dtype)
            print('I:', data_out.I)
            print('Ec:', data_out.Ec)
            print('Ewe:', data_out.Ewe)
            print('Cycle:', data_out.cycle,
                  'numpy', data_out.cycle_numpy, data_out.cycle_numpy.dtype)
            ew_ += data_out.Ewe
            ii_ += data_out.I
    except KeyboardInterrupt:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    else:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    plt.plot(ew_, ii_)
    plt.show()
    print('end')


def test_speis_technique(potentiostat, channel):
    """Test the SPEIS technique"""
    potentiostat.connect()
    print('kk')
    speis = SPEIS(
        vs_initial=False, vs_final=False,
        initial_voltage_step=0.1,
        final_voltage_step=0.2,
        duration_step=1.0,
        step_number=3,
        final_frequency=100.0E3, initial_frequency=10.0E3,
        I_range='KBIO_IRANGE_1mA'
    )
    potentiostat.load_technique(channel, speis)
    potentiostat.start_channel(channel)

    try:
        while True:
            time.sleep(0.1)
            data_out = potentiostat.get_data(channel)
            if data_out is None:
                break
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
                print('t', data_out.t)
                print('Irange', data_out.Irange)
                print('step', data_out.step)
    except KeyboardInterrupt:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    else:
        potentiostat.stop_channel(channel)
        potentiostat.disconnect()
    print('end')

def run_ocv(potentiostat, channel):
    """
    Test technique - find out the IP address of Forse group
    :return:
    """

    # Connect to potentiostat
    potentiostat.connect()

    # Instantiate the technique. In this case, run OCV as a test
    technique = OCV(rest_time_T=0.2, record_every_dE=10.0, record_every_dT=0.01)

    # Load the technique onto desired channel of the potentiostat, and then start it
    potentiostat.load_technique(channel, technique)
    potentiostat.start_channel(channel)

    time.sleep(0.1)

    while True:
        # Get currently available data on specified channel
        data_out = potentiostat.get_data(channel)
        print(data_out.technique)
        print('time:', data_out.time,
              'numpy', data_out.time_numpy, data_out.time_numpy.dtype)
        print('I:', data_out.I)
        print('Ewe:', data_out.Ewe)

        if data_out is None:
            break

    potentiostat.stop_channel(channel)



if __name__ == '__main__':

    ip_address = 171967230
    channel = 7

    print('Connecting...')
    # Connect to potentiostat
    mpg2 = MPG2(ip_address)

    print('## Device info before connect:', mpg2.device_info)
    print('\n## Lib version:', mpg2.get_lib_version())
    dev_info = mpg2.connect()
    print('\n## Connect returned device info:')
    pprint(dev_info)
    mpg2.disconnect()

    # Get basic info
    basic(mpg2)
    test_ocv_technique(mpg2, channel)

    # Test CV on specified channel
    #test_cv_technique(mpg2, channel)
    #current_values()
    #test_ocv_technique()
    #test_cp_technique()
    #test_ca_technique()
    #test_cva_technique()
    #test_speis_technique()
    #test_message()
