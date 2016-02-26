/*
 * Copyright 2010 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
#include "baseincs.h"
#include "logstuff.h"

#include <adl_sdk.h>
#include "amdstream_setup.h"
#include "adl.h"

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
} ThermalDevices_t;

static ThermalDevices_t ThermalDevices[AMD_STREAM_MAX_GPUS];

static ADL_MAIN_CONTROL_CREATE            ADL_Main_Control_Create;
static ADL_MAIN_CONTROL_DESTROY           ADL_Main_Control_Destroy;
static ADL_ADAPTER_NUMBEROFADAPTERS_GET   ADL_Adapter_NumberOfAdapters_Get;
static ADL_ADAPTER_ADAPTERINFO_GET        ADL_Adapter_AdapterInfo_Get;
static ADL_OVERDRIVE5_THERMALDEVICES_ENUM ADL_Overdrive5_ThermalDevices_Enum;
static ADL_OVERDRIVE5_TEMPERATURE_GET     ADL_Overdrive5_Temperature_Get;
#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
static HINSTANCE hDLL;		// Handle to ADL DLL
#endif

// Memory allocation function
static void* __stdcall ADL_Main_Memory_Alloc ( int iSize )
{
    void* lpBuffer = malloc ( iSize );
    return lpBuffer;
}

// Optional Memory de-allocation function
static void __stdcall ADL_Main_Memory_Free ( void** lpBuffer )
{
    if ( NULL != *lpBuffer )
    {
        free ( *lpBuffer );
        *lpBuffer = NULL;
    }
}

void ADLinit()
{
  int i;

//Log("Initializing ADL...\n");
  for(i=0;i<AMD_STREAM_MAX_GPUS;i++)
    ThermalDevices[i].active=0;

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64) 

  hDLL = LoadLibrary("atiadlxx.dll");
  if (hDLL == NULL)
    hDLL = LoadLibrary("atiadlxy.dll");

  if (NULL == hDLL)
    return;

//Log("Retrieving ADL handlers\n");
  ADL_Main_Control_Create = (ADL_MAIN_CONTROL_CREATE) GetProcAddress(hDLL,"ADL_Main_Control_Create");
  ADL_Main_Control_Destroy = (ADL_MAIN_CONTROL_DESTROY) GetProcAddress(hDLL,"ADL_Main_Control_Destroy");
  ADL_Adapter_NumberOfAdapters_Get = (ADL_ADAPTER_NUMBEROFADAPTERS_GET) GetProcAddress(hDLL,"ADL_Adapter_NumberOfAdapters_Get");
  ADL_Adapter_AdapterInfo_Get = (ADL_ADAPTER_ADAPTERINFO_GET) GetProcAddress(hDLL,"ADL_Adapter_AdapterInfo_Get");
  ADL_Overdrive5_ThermalDevices_Enum = (ADL_OVERDRIVE5_THERMALDEVICES_ENUM) GetProcAddress(hDLL,"ADL_Overdrive5_ThermalDevices_Enum");
  ADL_Overdrive5_Temperature_Get = (ADL_OVERDRIVE5_TEMPERATURE_GET) GetProcAddress(hDLL,"ADL_Overdrive5_Temperature_Get");

#endif
  if ( NULL == ADL_Main_Control_Create || NULL == ADL_Main_Control_Destroy || NULL == ADL_Adapter_AdapterInfo_Get
    || NULL == ADL_Adapter_AdapterInfo_Get || NULL == ADL_Overdrive5_ThermalDevices_Enum 
    || NULL == ADL_Overdrive5_Temperature_Get)
      return;

  // Initialize ADL. The second parameter is 1, which means:
  // retrieve adapter information only for adapters that are physically present and enabled in the system
  if (ADL_Main_Control_Create (ADL_Main_Memory_Alloc, 1)!=ADL_OK)
    return;

  int iNumberAdapters;
  LPAdapterInfo     lpAdapterInfo = NULL;
  
  if ( ADL_Adapter_NumberOfAdapters_Get ( &iNumberAdapters )!=ADL_OK )
    return;

  if(iNumberAdapters>0)
  {
    lpAdapterInfo = (LPAdapterInfo) malloc ( sizeof (AdapterInfo) * iNumberAdapters );
    memset ( lpAdapterInfo,'\0', sizeof (AdapterInfo) * iNumberAdapters );

    // Get the AdapterInfo structure for all adapters in the system
    ADL_Adapter_AdapterInfo_Get (lpAdapterInfo, sizeof (AdapterInfo) * iNumberAdapters);
  }

    // Repeat for all available adapters in the system
    ADLThermalControllerInfo lpThermalControllerInfo;
    for ( int i = 0; (i < iNumberAdapters) & (i < AMD_STREAM_MAX_GPUS); i++ )
    {
      //Retrieve Termal Controller Info
      if (ADL_Overdrive5_ThermalDevices_Enum (i, 0, &lpThermalControllerInfo) == ADL_OK) 
        if (lpThermalControllerInfo.iThermalDomain == ADL_DL_THERMAL_DOMAIN_GPU)
        {
          for(int j=0; j<AMD_STREAM_MAX_GPUS; j++)
          {
            if(!ThermalDevices[j].active)
            {
              ThermalDevices[j].active=1;
              ThermalDevices[j].bus=lpAdapterInfo[j].iBusNumber;
              ThermalDevices[j].device=lpAdapterInfo[j].iDeviceNumber;
              ThermalDevices[j].iAdapterIndex=i;
              break;
            }
            if((ThermalDevices[j].bus==lpAdapterInfo[j].iBusNumber)&&
              (ThermalDevices[j].device==lpAdapterInfo[j].iDeviceNumber))
                continue;
          }
        }
    }
    ADL_Main_Memory_Free ( (void**)&lpAdapterInfo );
}

void ADLdeinit()
{
  if(ADL_Main_Control_Destroy)
    ADL_Main_Control_Destroy ();

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64) 
  if (hDLL)
    FreeLibrary(hDLL);
#endif
}

int stream_cputemp()
{
  unsigned K=27315;	//273.15K

  ADLTemperature lpTemperature;
  if(ADL_Overdrive5_Temperature_Get)
  {
    unsigned maxtemp=0;

    for(int i=0; i<AMD_STREAM_MAX_GPUS; i++)
    {
      if(ADL_Overdrive5_Temperature_Get(ThermalDevices[i].iAdapterIndex, 0, &lpTemperature) == ADL_OK)
        if(maxtemp<lpTemperature.iTemperature)
          maxtemp=lpTemperature.iTemperature;
    }
    return maxtemp/10+K;
  }else
    return K;
  
}
