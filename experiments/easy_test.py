import sys
sys.path.append('../')

# Import the easy biologic modules
import easy_biologic as ebl
import easy_biologic.base_programs as blp

# Set up the device to connect to

ip_address = "10.64.2.254"
channels = [7,]
params = {'time_interval': 1.0,
          'time': 30.0,
          'voltage_interval': 0.01}

potentiostat = ebl.BiologicDevice(ip_address)

print('Connecting...')

potentiostat.connect()

print('Connected.')

program = blp.OCV(potentiostat, params, channels, autoconnect=True)

program.run()

print('Program successfully executed.')



