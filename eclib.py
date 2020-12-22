"""
EClib dll interface.
"""

# imports
import ctypes;


# ECLIB class
class eclib(object):

    # Constants
    UNITS_NB = 16;

    # EC-Lab DLL (private)
    __dll = ctypes.WinDLL("..\\..\\EC-Lab Development Package\\EClib64.dll");

    # Device Info Structure
    class DeviceInfoType(ctypes.Structure):
        _fields_ = [("DeviceCode", ctypes.c_int32),
                    ("RAMSize", ctypes.c_int32),
                    ("CPU", ctypes.c_int32),
                    ("NumberOfChannels", ctypes.c_int32),
                    ("NumberOfSlots", ctypes.c_int32),
                    ("FirmwareVersion", ctypes.c_int32),
                    ("FirmwareDate_yyyy", ctypes.c_int32),
                    ("FirmwareDate_mm", ctypes.c_int32),
                    ("FirmwareDate_dd", ctypes.c_int32),
                    ("HTdisplayOn", ctypes.c_int32),
                    ("NbOfConnectedPC", ctypes.c_int32)];

    # Current Values Type
    class CurrentValuesType(ctypes.Structure):
        _fields_ = [("State", ctypes.c_int32),
                    ("MemFilled", ctypes.c_int32),
                    ("TimeBase", ctypes.c_float),
                    ("Ewe", ctypes.c_float),
                    ("EweRangeMin", ctypes.c_float),
                    ("EweRangeMax", ctypes.c_float),
                    ("Ece", ctypes.c_float),
                    ("EceRangeMin", ctypes.c_float),
                    ("EceRangeMax", ctypes.c_float),
                    ("Eoverflow", ctypes.c_int32),
                    ("I", ctypes.c_float),
                    ("IRange", ctypes.c_int32),
                    ("Ioverflow", ctypes.c_int32),
                    ("ElapsedTime", ctypes.c_float),
                    ("Freq", ctypes.c_float),
                    ("Rcomp", ctypes.c_float),
                    ("Saturation", ctypes.c_int32),
                    ("OptErr", ctypes.c_int32),
                    ("OptPos", ctypes.c_int32)];

    # Data Information Type
    class DataInfosType(ctypes.Structure):
        _fields_ = [("IRQskipped", ctypes.c_int32),
                    ("NbRows", ctypes.c_int32),
                    ("NbCols", ctypes.c_int32),
                    ("TechniqueIndex", ctypes.c_int32),
                    ("TechniqueID", ctypes.c_int32),
                    ("ProcessIndex", ctypes.c_int32),
                    ("loop", ctypes.c_int32),
                    ("StartTime", ctypes.c_double),
                    ("MuxPad", ctypes.c_int32)];

    # Data buffer Type
    DataBufferType = ctypes.c_uint32 * 1000;

    # ECC parameter structure
    class EccParamType(ctypes.Structure):
        _fields_ = [("ParamStr", 64 * ctypes.c_byte),
                    ("ParamType", ctypes.c_int32),
                    ("ParamVal", ctypes.c_uint32),
                    ("ParamIndex", ctypes.c_int32)];

    # ECC parameters structure
    class EccParamsType(ctypes.Structure):
        _fields_ = [("len", ctypes.c_int32),
                    ("pParams", ctypes.c_void_p)];

    # Array of units
    UnitsType = ctypes.c_byte * UNITS_NB;

    # Array of results
    ResultsType = ctypes.c_int32 * UNITS_NB;

    # Error Enumeration
    class ErrorCodeEnum(object):
        ERR_NOERROR = 0;

    # Technique Parameter Type Enumeration
    class ParamTypeEnum(object):
        PARAM_INT = 0;
        PARAM_BOOLEAN = 1;
        PARAM_SINGLE = 2;


    # ErrorCode BL_ConvertNumericIntoSingle(int num, ref float psgl)
    BL_ConvertNumericIntoSingle = __dll["BL_ConvertNumericIntoSingle"];
    BL_ConvertNumericIntoSingle.restype = ctypes.c_int;

    # ErrorCode BL_Connect(string server, byte timeout, ref int connection_id, ref DeviceInfo pInfos)
    BL_Connect = __dll["BL_Connect"];
    BL_Connect.restype = ctypes.c_int;
    
    # ErrorCode BL_TestConnection(int ID)
    BL_TestConnection = __dll["BL_TestConnection"];
    BL_TestConnection.restype = ctypes.c_int;

    # ErrorCode BL_LoadFirmware(int ID, byte[] pChannels, int[] pResults, byte Length, bool ShowGauge, bool ForceReload, string BinFile, string XlxFile)
    BL_LoadFirmware = __dll["BL_LoadFirmware"];
    BL_LoadFirmware.restype = ctypes.c_int;

    # bool BL_IsChannelPlugged(int ID, byte ch)
    BL_IsChannelPlugged = __dll["BL_IsChannelPlugged"];
    BL_IsChannelPlugged.restype = ctypes.c_bool;

    # ErrorCode BL_GetChannelsPlugged(int ID, byte[] pChPlugged, byte Size)
    BL_GetChannelsPlugged = __dll["BL_GetChannelsPlugged"];
    BL_GetChannelsPlugged.restype = ctypes.c_int;

    # ErrorCode BL_GetMessage(int ID, byte ch, [MarshalAs(UnmanagedType.LPArray)] byte[] msg, ref int size)
    BL_GetMessage = __dll["BL_GetMessage"];
    BL_GetMessage.restype = ctypes.c_int;

    # ErrorCode BL_LoadTechnique(int ID, byte channel, string pFName, EccParams pparams, bool FirstTechnique, bool LastTechnique, bool DisplayParams)
    BL_LoadTechnique = __dll["BL_LoadTechnique"];
    BL_LoadTechnique.restype = ctypes.c_int;

    # ErrorCode BL_DefineSglParameter(string lbl, float value, int index, IntPtr pParam)
    BL_DefineSglParameter = __dll["BL_DefineSglParameter"];
    BL_DefineSglParameter.restype = ctypes.c_int;

    # ErrorCode BL_TestCommSpeed(int ID, byte channel, ref int spd_rcvt, ref int spd_kernel)
    BL_TestCommSpeed = __dll["BL_TestCommSpeed"];
    BL_TestCommSpeed.restype = ctypes.c_int;

    # ErrorCode BL_StartChannel(int ID, byte channel)
    BL_StartChannel = __dll["BL_StartChannel"];
    BL_StartChannel.restype = ctypes.c_int;

    # ErrorCode BL_GetData(int ID, byte channel, [MarshalAs(UnmanagedType.LPArray, SizeConst=1000)] int[] buf, ref DataInfos pInfos, ref CurrentValues pValues)
    BL_GetData = __dll["BL_GetData"];
    BL_GetData.restype = ctypes.c_int;
