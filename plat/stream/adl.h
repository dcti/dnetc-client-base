/*
 * Copyright 2010 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/


#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64) 
#include <windows.h>
#endif

#include <adl_sdk.h>
#include "amdstream_setup.h"

typedef int ( *ADL_MAIN_CONTROL_CREATE )(ADL_MAIN_MALLOC_CALLBACK, int );
typedef int ( *ADL_MAIN_CONTROL_DESTROY )();
typedef int ( *ADL_ADAPTER_NUMBEROFADAPTERS_GET ) ( int* );
typedef int ( *ADL_ADAPTER_ADAPTERINFO_GET ) ( LPAdapterInfo, int );
typedef int ( *ADL_OVERDRIVE5_THERMALDEVICES_ENUM ) ( int, int, ADLThermalControllerInfo* );
typedef int ( *ADL_OVERDRIVE5_TEMPERATURE_GET ) ( int, int, ADLTemperature* );

typedef struct
{
  int bus;
  int device;
  int iAdapterIndex;
  int active;
}ThermalDevices_t;

extern ThermalDevices_t ThermalDevices[AMD_STREAM_MAX_GPUS];

void ADLinit();
void ADLdeinit();
int stream_cputemp();
