"""
ECLIB User Example.
"""

# imports
from eclib import eclib;
import ctypes;
import sys;
import time;
import threading;


# Configuration
cfg_conn_ip = "10.64.2.254";
cfg_conn_timeout = 10;
cfg_channel = 7;
cfg_debug_enabled = False;

# exit codes
EXIT_OK = 0;
EXIT_GetMessage = -1;
EXIT_Connect = -2;
EXIT_GetChannelsPlugged = -3;
EXIT_LoadFirmware = -4;
EXIT_DefineSglParameter = -5;
EXIT_LoadTechnique = -6;
EXIT_GetData = -7;
EXIT_ConvertNumericIntoSingle = -8;
EXIT_Unknown = -127;


# Global variables
glob_firmware_loaded = False;
glob_stop = False;
glob_conn_id = ctypes.c_int(-1);
# We use eclib mutex because we cannot call two BL fucntions simultaneously 
glob_eclib_mutex = threading.Lock();
# We use printf mutex because two treads can print simultaneously
glob_printf_mutex = threading.Lock();


# Exit function
def exit(code, error):
    global glob_stop;
    log("Exit code "+ str(code) + ". Error " + str(error));
    glob_stop = True;
    time.sleep(10);
    sys.exit(code);


# log to replace printf
def log(info):
    global glob_printf_mutex;
    glob_printf_mutex.acquire();
    glob_printf_mutex.release();


# Show device info function
def device_info_show(device_info):
    log("========== Device Information ==========");
    log("RAM size: " + str(device_info.RAMSize));
    log("Device code: " + str(device_info.DeviceCode));
    log("Number of channels: " + str(device_info.NumberOfChannels));
    log("========================================");
    log("");


# Parse OCV data
def parseOcvData(DataInfos, CurrentValues, DataBuffer):
    offset = 0;

    for i in range(0, DataInfos.NbRows):
        raw_t_high = DataBuffer[offset + 0];
        raw_t_low = DataBuffer[offset + 1];
        raw_ewe = DataBuffer[offset + 2];

        ewe = ctypes.c_float(0.0);
        #t = DataInfos.StartTime + CurrentValues.TimeBase * (raw_t_high * 2^32 + raw_t_low);
        t = CurrentValues.TimeBase * (raw_t_high * 2^32 + raw_t_low);

        glob_eclib_mutex.acquire();       
        error = eclib.BL_ConvertNumericIntoSingle(raw_ewe, ctypes.byref(ewe));
        if error != eclib.ErrorCodeEnum.ERR_NOERROR:
            exit(EXIT_ConvertNumericIntoSingle, error);
        glob_eclib_mutex.release();

        # log 
        log(str(t) + " s / " + str(ewe.value) + " V");

        offset = offset + 3;


# thread to get firmware debug messages
def thread_debug(reserved):
    global glob_firmware_loaded;
    global glob_stop;
    global glob_conn_id;

    # loop 
    while not glob_stop:
        if glob_firmware_loaded and cfg_debug_enabled:
            msg_size = 2048;
            msg = ctypes.create_string_buffer(msg_size);
            size = ctypes.c_int32(msg_size); 
            glob_eclib_mutex.acquire();
            error = eclib.BL_GetMessage(glob_conn_id, ctypes.c_ubyte(cfg_channel), ctypes.byref(msg), ctypes.byref(size));
            glob_eclib_mutex.release();
            if error != eclib.ErrorCodeEnum.ERR_NOERROR:
                exit(EXIT_GetMessage, error);
            elif size.value > 0:
                log("Firmware debug: " + msg.value);
            else:
                pass;
        time.sleep(0.25);


# Experiment tread routine
def thread_experiment(reserved):
    global glob_firmware_loaded;
    global glob_stop;
    global glob_conn_id;

    # Connection
    device_info = eclib.DeviceInfoType();
    glob_eclib_mutex.acquire();
    error = eclib.BL_Connect(ctypes.c_char_p(cfg_conn_ip), ctypes.c_byte(cfg_conn_timeout), ctypes.byref(glob_conn_id), ctypes.byref(device_info));
    log("BL_Connect: " + str(error));
    glob_eclib_mutex.release();
    if error != eclib.ErrorCodeEnum.ERR_NOERROR:
        exit(EXIT_Connect, error);
    device_info_show(device_info);

    # Connected channels
    units = eclib.UnitsType();
    glob_eclib_mutex.acquire();
    error = eclib.BL_GetChannelsPlugged(glob_conn_id, ctypes.byref(units), ctypes.c_ubyte(eclib.UNITS_NB));
    log("BL_GetChannelsPlugged: " + str(error));
    glob_eclib_mutex.release();
    if error != eclib.ErrorCodeEnum.ERR_NOERROR:
        exit(EXIT_GetChannelsPlugged, error);

    # Firmware loading. Attention! ShowGauge must be False.
    force = True;
    results = eclib.ResultsType();
    glob_eclib_mutex.acquire();
    error = eclib.BL_LoadFirmware(glob_conn_id, units, ctypes.byref(results), ctypes.c_ubyte(eclib.UNITS_NB), False, force, None, None);
    log("BL_LoadFirmware: " + str(error));
    glob_eclib_mutex.release();
    if error != eclib.ErrorCodeEnum.ERR_NOERROR:
        exit(EXIT_LoadFirmware, error);
    glob_firmware_loaded = True;

    # Technique parameters   
    params_nb = 2;
    EccParamArrayType = eclib.EccParamType * params_nb;
    EccParamArray = EccParamArrayType();
    EccParams = eclib.EccParamsType();
    EccParams.len = ctypes.c_int32(params_nb);
    EccParams.pParams = ctypes.cast(EccParamArray, ctypes.c_void_p);

    # Technique Parameter #0
    label = "Rest_time_T";
    value = 10.0;
    index = 0;
    glob_eclib_mutex.acquire();
    error = eclib.BL_DefineSglParameter(ctypes.c_char_p(label), ctypes.c_float(value), ctypes.c_int32(index), ctypes.byref(EccParamArray[0]));
    glob_eclib_mutex.release();
    if error != eclib.ErrorCodeEnum.ERR_NOERROR:
        exit(EXIT_DefineSglParameter, error);

    # Technique Parameter #1
    label = "Record_every_dT";
    value = 0.1;
    index = 0;
    glob_eclib_mutex.acquire();
    error = eclib.BL_DefineSglParameter(ctypes.c_char_p(label), ctypes.c_float(value), ctypes.c_int32(index), ctypes.byref(EccParamArray[1]));
    glob_eclib_mutex.release();
    if error != eclib.ErrorCodeEnum.ERR_NOERROR:
        exit(EXIT_DefineSglParameter, error);

    # Load technique
    glob_eclib_mutex.acquire();
    error = eclib.BL_LoadTechnique(glob_conn_id, ctypes.c_ubyte(cfg_channel), "ocv.ecc", EccParams, True, True, True);
    log("BL_LoadTechnique: " + str(error));
    glob_eclib_mutex.release();
    if error != eclib.ErrorCodeEnum.ERR_NOERROR:
        exit(EXIT_LoadTechnique, error);

    # Start technique
    glob_eclib_mutex.acquire();
    error = eclib.BL_StartChannel(glob_conn_id, ctypes.c_ubyte(cfg_channel));
    glob_eclib_mutex.release();

    # loop to get data
    while not glob_stop:
        buffer = eclib.DataBufferType();
        infos = eclib.DataInfosType();
        values = eclib.CurrentValuesType();
        glob_eclib_mutex.acquire();
        error = eclib.BL_GetData(glob_conn_id, ctypes.c_ubyte(cfg_channel), ctypes.byref(buffer), ctypes.byref(infos), ctypes.byref(values));
        log("BL_GetData: " + str(error));
        glob_eclib_mutex.release();
        if error != eclib.ErrorCodeEnum.ERR_NOERROR:
            exit(EXIT_GetData, error);

        parseOcvData(infos, values, buffer);

        time.sleep(1);


# Main function
def main():
    # treads 
    threads = {}
    threads["experiment"] = threading.Thread(target=thread_experiment, args=(0,));
    threads["debug"] = threading.Thread(target=thread_debug, args=(0,));
    # starting threads
    threads["experiment"].start();
    threads["debug"].start();
    # loop
    while not glob_stop:
        time.sleep(1);


# Main call 
if __name__ == "__main__":
    main();
