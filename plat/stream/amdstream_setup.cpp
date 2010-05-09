/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: amdstream_setup.cpp,v 1.16 2010/05/09 10:47:37 stream Exp $
*/

#include "amdstream_setup.h"
#include "amdstream_context.h"
#include "amdstream_info.h"
#include "logstuff.h"
#include "adl.h"

#include <calcl.h>
#include <stdlib.h>

stream_context_t CContext[AMD_STREAM_MAX_GPUS];
int atistream_numDevices = -1;

PFNCALCTXWAITFOREVENTS calCtxWaitForEvents;
u32 isCalCtxWaitForEventsSupported;

void AMDInitMutex();

static void ati_diagnose(CALresult result, const char *where, u32 DeviceIndex, const char *errmsg)
{
  Log("Error %s on GPU %u\n", where, DeviceIndex);
  Log("Error code %u, message: %s\n", result, errmsg);
}

CALresult ati_verbose(CALresult result, const char *where, u32 DeviceIndex)
{
  if (result != CAL_RESULT_OK)
  {
    ati_diagnose(result, where, DeviceIndex, calGetErrorString());
    if (atistream_numDevices > 1)
      Log("(message could be wrong in multi-threaded environments)\n");
  }
  return result;
}

CALresult ati_verbose_cl(CALresult result, const char *where, u32 DeviceIndex)
{
  if (result != CAL_RESULT_OK)
  {
    ati_diagnose(result, where, DeviceIndex, calclGetErrorString());
    // compiler is mutex-protected so no warnings about multiple threads
  }
  return result;
}

int AMDStreamInitialize()
{
  CALuint numDevices;
  unsigned i;

  if (atistream_numDevices >= 0)
    return atistream_numDevices;

  AMDInitMutex();
  atistream_numDevices = 0;

  // Initialize CAL (global)
  if (CAL_RESULT_OK != ati_verbose( calInit(), "initializing CAL", 0 ))
    return 0;

  // Query number of devices
  if (CAL_RESULT_OK != ati_verbose( calDeviceGetCount(&numDevices), "querying number of devices", 0 ))
    return 0;
  if (numDevices > AMD_STREAM_MAX_GPUS)
    numDevices = AMD_STREAM_MAX_GPUS;

  // Open each device and get it's capabilities
  for (i = 0; i < numDevices; i++)
  {
    stream_context_t *dev = &CContext[i];
    CALresult result;

    // Open device
    if (CAL_RESULT_OK != ati_verbose( calDeviceOpen(&dev->device, i), "opening device", i ))
      continue;

    // Query device attribs
    dev->attribs.struct_size = sizeof(CALdeviceattribs);
    result = calDeviceGetAttribs(&dev->attribs, i);
    if (result != CAL_RESULT_OK)
    {
      /* SDK 2.1 incompatibility?
       * Structure size in .h is 0x58 but Windows 7 Radeon HD2600 driver
       * rejects values above 0x50. Try again with smaller structure size.
       *
       * In this case, numberOfShaderEngines and targetRevision are undefined
       * (they're not used in our code anyway).
       */
      if (dev->attribs.struct_size > 0x50)
      {
        memset(&dev->attribs, 0xEE, sizeof(CALdeviceattribs)); // 0xEEEEEEEE in undefined fields
        dev->attribs.struct_size = 0x50;
        result = calDeviceGetAttribs(&CContext[i].attribs, i);
      }
    }
    if (CAL_RESULT_OK != ati_verbose( result, "querying device attributes", i ))
      continue;

    dev->active      = true;
    dev->domainSizeX = 256;
    dev->domainSizeY = 256;
    dev->maxIters    = 16;
  }

  isCalCtxWaitForEventsSupported = 0;
  if (calExtSupported((CALextid)0x8009) == CAL_RESULT_OK)
    if (calExtGetProc((CALextproc*)&calCtxWaitForEvents, (CALextid)0x8009, "calCtxWaitForEvents") == CAL_RESULT_OK)
      isCalCtxWaitForEventsSupported = 1;

  ADLinit();

  // Postpone initialization of atistream_numDevices to shut up ati_verbose() complains
  // about multi-threaded environment during initalization.
  atistream_numDevices = numDevices;

  return numDevices;
}

static void ati_zap_resource(CALresource *pRes, const char *where, int Device)
{
  if (*pRes)
  {
    ati_verbose( calResFree(*pRes), where, Device );
    *pRes = 0;
  }
}

static void ati_zap_mem(CALmem *pMem, CALcontext ctx, const char *where, int Device)
{
  if (*pMem)
  {
    ati_verbose( calCtxReleaseMem(ctx, *pMem), where, Device );
    *pMem = 0;
  }
}

static void ati_zap_module(CALmodule *pModule, CALcontext ctx, const char *where, int Device)
{
  if (*pModule)
  {
    ati_verbose( calModuleUnload(ctx, *pModule), where, Device );
    *pModule = 0;
  }
}

void AMDStreamReinitializeDevice(int Device)
{
  stream_context_t *dev = &CContext[Device];

  // Creation: CALdevice -> CALcontext, CALresource;
  //           CALcontext, CALresource -> CALmem;
  //           CALcontext, CALimage -> CALmodule;
  // Deinitialize in reverse order

  // Unloading the module
  ati_zap_module(&dev->module0, dev->ctx, "unloading module0", Device);
  ati_zap_module(&dev->module1, dev->ctx, "unloading module1", Device);

  // Freeing compiled program binary
  if (dev->image)
  {
    ati_verbose( calclFreeImage(dev->image), "releasing program image", Device );
    dev->image = 0;
  }

  // Releasing resource from context
  ati_zap_mem(&dev->constMem0,  dev->ctx, "releasing constMem0",  Device);
  ati_zap_mem(&dev->constMem1,  dev->ctx, "releasing constMem1",  Device);
  ati_zap_mem(&dev->outputMem0, dev->ctx, "releasing outputMem0", Device);
  ati_zap_mem(&dev->outputMem1, dev->ctx, "releasing outputMem1", Device);
  ati_zap_mem(&dev->globalMem0, dev->ctx, "releasing globalMem0", Device);
  ati_zap_mem(&dev->globalMem1, dev->ctx, "releasing globalMem1", Device);

  // Deallocating resources
  ati_zap_resource(&dev->constRes0,  "releasing constRes0",  Device);
  ati_zap_resource(&dev->constRes1,  "releasing constRes1",  Device);
  ati_zap_resource(&dev->outputRes0, "releasing outputRes0", Device);
  ati_zap_resource(&dev->outputRes1, "releasing outputRes1", Device);
  ati_zap_resource(&dev->globalRes0, "releasing globalRes0", Device);
  ati_zap_resource(&dev->globalRes1, "releasing globalRes1", Device);

  // Destroying context
  if (dev->ctx)
  {
    ati_verbose( calCtxDestroy(dev->ctx), "destroying context", Device );
    dev->ctx = 0;
  }

  dev->coreID = CORE_NONE;
}
