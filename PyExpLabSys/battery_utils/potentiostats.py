from PyExpLabSys.PyExpLabSys.drivers.bio_logic import GeneralPotentiostat


# Implement a new potentiostat
class MPG2(GeneralPotentiostat):
    """Specific driver for the MPG2 potentiostat"""

    def __init__(self, address, EClib_dll_path=None):
        """Initialize the MPG2 potentiostat driver

        See the __init__ method for the GeneralPotentiostat class for an
        explanation of the arguments.
        """
        super(MPG2, self).__init__(
            type_='KBIO_DEV_MPG2',
            address=address,
            EClib_dll_path=EClib_dll_path
        )

