from __future__ import print_function
import os
import sys
import inspect
from collections import namedtuple
from ctypes import c_uint8, c_uint32, c_int32
from ctypes import c_float, c_double, c_char
from ctypes import Structure
from ctypes import create_string_buffer, byref, POINTER, cast
try:
    from ctypes import WinDLL
except ImportError:
    RUNNING_SPHINX = False
    for module in sys.modules:
        if 'sphinx' in module:
            RUNNING_SPHINX = True
    # Let the module continue after this fatal import error, if we are running
    # on read the docs or we can detect that sphinx is imported
    if not (os.environ.get('READTHEDOCS', None) == 'True' or RUNNING_SPHINX):
        raise

# Numpy is optional and is only required if it is resired to get the data as
# numpy arrays
try:
    import numpy
    GOT_NUMPY = True
except ImportError:
    GOT_NUMPY = False

from PyExpLabSys.PyExpLabSys.drivers.bio_logic import Technique, TechniqueArgument, DataField

# Section 7.3 in the specification
class ModularPulse(Technique):
    """Modular Pulse (MP) technique class.

    The MP technique returns data on fields (in order):

    * time (float)
    * Ec (float)
    * I (float)
    * Ewe (float)
    * cycle (int)
    """

    #:Data fields definition
    data_fields = {
        'common': [
            DataField('Ec', c_float),
            DataField('I', c_float),
            DataField('Ewe', c_float),
            DataField('cycle', c_uint32),
        ]
    }

    def __init__(self, vs_initial, value_step, duration_step,
                 record_every_dT, record_every_dM, mode_step,
                 step_number=0, record_every_rc=0, N_cycles=0):
        """
        Initialize the MP technique:

        Args:
            value_step (list): List (or tuple) of 20 singles, indicating the voltage step (V) if potentiostatic mode,
                or current step (A) in galvanostatic mode
            vs_initial(list of 20 bools): voltage/current step vs initial one
            duration_step (list of 20 singles): List (or tuple) of 20 floats indicating the duration step (s)
            record_every_dT (list of 20 singles): Record every dt (s)
            record_every_dM (list of 20 singles): Record every dI (A) or dE(V)
            mode_step (list of 20 ints): If 0, potentiostatic mode, if 1, galvanostatic mode
            step_number (int): number of steps minus 1 (0 <= step_number <= 19)
            record_every_rc (int): Record every cycle (record_every_rc >=0)
            N_cycles (int): The number of cycles - 1 (n_cycle >= 0)

        Raises:
            ValueError: If vs_initial, value_step, duration_step, record_every_dT, record_every_dM,
            mode_step and scan_rate are not all of length 20
        """
        for input_name in ('vs_initial', 'Value_step', 'duration_step',
                           'record_every_dM', 'record_every_dT', 'Mode_step'
                           ):
            if len(locals()[input_name]) != 20:
                message = 'Input \'{}\' must be of length 5, not {}'.format(
                    input_name, len(locals()[input_name]))
                raise ValueError(message)
        args = (
            TechniqueArgument('vs_initial', '[bool]', vs_initial,
                              'in', [True]*20),
            TechniqueArgument('Value_step', '[single]', value_step,
                              [0.0]*20),
            TechniqueArgument('Duration_step', '[single]', duration_step,
                              [1.0] * 20),
            TechniqueArgument('Record_every_dT', '[single]', record_every_dT,
                              [0.01] * 20),
            TechniqueArgument('Record_every_dM', '[single]', record_every_dM,
                              [0.01] * 20),
            TechniqueArgument('Mode_step', '[integer]', mode_step,
                              [1] * 20),
            TechniqueArgument('Step_number', '[integer]', mode_step, 20),
            TechniqueArgument('Record_every_rc', 'integer', record_every_rc,
                              '>=', 0),
            TechniqueArgument('N_Cycles', 'integer', N_cycles, '>=', 0),
        )
        super(ModularPulse, self).__init__(args, 'mp.ecc')

