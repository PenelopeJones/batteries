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

# Section 7.36 in the specification
class CPLimit(Technique):
    """Chrono-Potentiometry (CP) technique class, with limits.

    The CP technique returns data on fields (in order):

    * time (float)
    * Ewe (float)
    * I (float)
    * cycle (int)
    """

    #: Data fields definition
    data_fields = {
        'common': [
            DataField('Ec', c_float),
            DataField('Ewe', c_float),
            DataField('I', c_float),
            DataField('cycle', c_uint32),
        ]
    }

    def __init__(self, current_step=(50E-6,), vs_initial=(False,),
                 duration_step=(10000,), record_every_dT=0.1, record_every_dE=0.001,
                 N_cycles=0, test1_config=(1,), test1_value=(3.0,),
                 test2_config=(5,), test2_value=(1.0,)):
        """Initialize the CP technique

        The first test is E < test1_value
        The second test is E > test2_value

        NOTE: The current_step, vs_initial and duration_step must be a list or
        tuple with the same length.

        Args:
            current_step (list): List (or tuple) of floats indicating the
                current steps (A). See NOTE above.
            vs_initial (list): List (or tuple) of booleans indicating whether
                the current steps is vs. the initial one. See NOTE above.
            duration_step (list): List (or tuple) of floats indicating the
                duration of each step (s). See NOTE above.
            record_every_dT (float): Record every dT (s)
            record_every_dE (float): Record every dE (V)
            N_cycles (int): The number of times the technique is REPEATED.
                NOTE: This means that the default value is 0 which means that
                the technique will be run once.

        Raises:
            ValueError: On bad lengths for the list arguments
        """
        if not len(current_step) == len(vs_initial) == len(duration_step) == len(test1_value):
            message = 'The length of current_step, vs_initial and '\
                      'duration_step must be the same'
            raise ValueError(message)

        args = (
            TechniqueArgument('Current_step', '[single]', current_step,
                              None, None),
            TechniqueArgument('vs_initial', '[bool]', vs_initial,
                              'in', [True, False]),
            TechniqueArgument('Duration_step', '[single]', duration_step,
                              '>=', 0),
            TechniqueArgument('Step_number', 'integer', len(current_step),
                              'in', range(99)),
            TechniqueArgument('Record_every_dT', 'single', record_every_dT,
                              '>=', 0),
            TechniqueArgument('Record_every_dE', 'single', record_every_dE,
                              '>=', 0),
            TechniqueArgument('N_Cycles', 'integer', N_cycles, '>=', 0),
            TechniqueArgument('Test1_Config', '[integer]', test1_config, '>=', 0),
            TechniqueArgument('Test1_Value', '[single]', test1_value, '>=', 0),
            TechniqueArgument('Test2_Config', '[integer]', test2_config, '>=', 0),
            TechniqueArgument('Test2_Value', '[single]', test2_value, '>=', 0),
        )
        super(CPLimits, self).__init__(args, 'cplimit.ecc')

# Section 7.36 in the specification
class CALimit(Technique):
    """Chrono-Amperometry (CA) technique class, with limits.

    The CP technique returns data on fields (in order):

    * time (float)
    * Ewe (float)
    * I (float)
    * cycle (int)
    """

    #: Data fields definition
    data_fields = {
        'common': [
            DataField('Ewe', c_float),
            DataField('I', c_float),
            DataField('cycle', c_uint32),
        ]
    }

    def __init__(self, current_step=(50E-6,), vs_initial=(False,),
                 duration_step=(10000,), record_every_dT=0.1, record_every_dE=0.001,
                 N_cycles=0, test1_config=(5,), test1_value=(1.0e-5,)):
        """Initialize the CA technique

        NOTE: The current_step, vs_initial and duration_step must be a list or
        tuple with the same length.

        Limit: stop discharging when the current is lower than a certain value.

        Args:
            current_step (list): List (or tuple) of floats indicating the
                current steps (A). See NOTE above.
            vs_initial (list): List (or tuple) of booleans indicating whether
                the current steps is vs. the initial one. See NOTE above.
            duration_step (list): List (or tuple) of floats indicating the
                duration of each step (s). See NOTE above.
            record_every_dT (float): Record every dT (s)
            record_every_dE (float): Record every dE (V)
            N_cycles (int): The number of times the technique is REPEATED.
                NOTE: This means that the default value is 0 which means that
                the technique will be run once.

        Raises:
            ValueError: On bad lengths for the list arguments
        """
        if not len(current_step) == len(vs_initial) == len(duration_step) == len(test1_value):
            message = 'The length of current_step, vs_initial and '\
                      'duration_step must be the same'
            raise ValueError(message)

        args = (
            TechniqueArgument('Current_step', '[single]', current_step,
                              None, None),
            TechniqueArgument('vs_initial', '[bool]', vs_initial,
                              'in', [True, False]),
            TechniqueArgument('Duration_step', '[single]', duration_step,
                              '>=', 0),
            TechniqueArgument('Step_number', 'integer', len(current_step),
                              'in', range(99)),
            TechniqueArgument('Record_every_dT', 'single', record_every_dT,
                              '>=', 0),
            TechniqueArgument('Record_every_dE', 'single', record_every_dE,
                              '>=', 0),
            TechniqueArgument('N_Cycles', 'integer', N_cycles, '>=', 0),
            TechniqueArgument('Test1_Config', '[integer]', test1_config, '>=', 0),
            TechniqueArgument('Test1_Value', '[single]', test1_value, '>=', 0),
        )
        super(CALimit, self).__init__(args, 'calimit.ecc')


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

