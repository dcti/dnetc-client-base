/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: amdstream_setup.cpp,v 1.12 2010/01/03 10:02:33 sla Exp $
*/

#include "amdstream_setup.h"
#include "amdstream_context.h"
#include "amdstream_info.h"
#include "logstuff.h"

#include <calcl.h>
#include <stdlib.h>

stream_context_t CContext[AMD_STREAM_MAX_GPUS];
int atistream_numDevices = -1;

extern int ati_RC_error;

void InitMutex();

void AMDStreamInitialize()
{
  if (atistream_numDevices >= 0)
    return;

  atistream_numDevices=0;
  calInit();
  {
    CALuint numDevices;
    // Finding number of devices
    if(calDeviceGetCount(&numDevices)!=CAL_RESULT_OK)
      return;
    if(numDevices==0) {
//      LogScreen("No supported devices found!");
      return;
    }
    atistream_numDevices = numDevices;
  }

  InitMutex();
  if(atistream_numDevices>AMD_STREAM_MAX_GPUS)
    atistream_numDevices=AMD_STREAM_MAX_GPUS;

  ati_RC_error=0;

  CALdevice dev;
  for(int i=0; i<AMD_STREAM_MAX_GPUS; i++) {
    CContext[i].active=false;
    CContext[i].coreID=CORE_NONE;

    CContext[i].constMem0=0;
    CContext[i].outputMem0=0;
    CContext[i].globalMem0=0;

    CContext[i].constName0=0;
    CContext[i].outName0=0;
    CContext[i].globalName0=0;

    CContext[i].constRes0=0;
    CContext[i].outputRes0=0;
    CContext[i].globalRes0=0;

    CContext[i].ctx=0;
    CContext[i].device=0;
    CContext[i].module0=0;
    CContext[i].image=0L;
    CContext[i].idleCounter=0;

    if(i<atistream_numDevices) {
      // Opening device
      calDeviceOpen(&dev, i);
      CContext[i].device=dev;

      // Querying device attribs
      CContext[i].attribs.struct_size = sizeof(CALdeviceattribs);
      if(calDeviceGetAttribs(&CContext[i].attribs, i)!=CAL_RESULT_OK)
        continue;
      CContext[i].active=true;

      CContext[i].domainSizeX=32;
      CContext[i].domainSizeY=32;
      CContext[i].maxIters=256;
    }
  }
}

void AMDStreamReinitializeDevice(int Device)
{
  // Unloading the module
  if(CContext[Device].module0)
  {
    calModuleUnload(CContext[Device].ctx, CContext[Device].module0);
    CContext[Device].module0=0;
  }

  // Freeing compiled program binary
  if(CContext[Device].image) {
    calclFreeImage(CContext[Device].image);
    CContext[Device].image=0L;
  }

  // Releasing resource from context
  if(CContext[Device].constMem0) {
    calCtxReleaseMem(CContext[Device].ctx, CContext[Device].constMem0);
    CContext[Device].constMem0=0;
  }
  if(CContext[Device].outputMem0) {
    calCtxReleaseMem(CContext[Device].ctx, CContext[Device].outputMem0);
    CContext[Device].outputMem0=0;
  }
  if(CContext[Device].globalMem0) {
    calCtxReleaseMem(CContext[Device].ctx, CContext[Device].globalMem0);
    CContext[Device].globalMem0=0;
  }

  // Deallocating resources
  if(CContext[Device].constRes0) {
    calResFree(CContext[Device].constRes0);
    CContext[Device].constRes0=0;
  }

  // Deallocating resources
  if(CContext[Device].outputRes0) {
    calResFree(CContext[Device].outputRes0);
    CContext[Device].outputRes0=0;
  }

  // Deallocating resources
  if(CContext[Device].globalRes0) {
    calResFree(CContext[Device].globalRes0);
    CContext[Device].globalRes0=0;
  }

  // Destroying context
  calCtxDestroy(CContext[Device].ctx);
  CContext[Device].coreID=CORE_NONE;

  if(ATIstream_GPUname)
  {
    free(ATIstream_GPUname);
    ATIstream_GPUname=0L;
  }
}
