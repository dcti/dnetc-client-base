/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev
 * PanAm
 * Alexei Chupyatov
 *
 * $Id: amdstream_setup.cpp,v 1.8 2009/01/02 03:25:28 andreasb Exp $
*/

#include "amdstream_setup.h"
#include "amdstream_context.h"
#include "logstuff.h"

#include <calcl.h>

stream_context_t CContext[AMD_STREAM_MAX_GPUS];
int amdstream_numDevices = -1;

void AMDStreamInitialize()
{
  if (amdstream_numDevices >= 0)
    return;

  amdstream_numDevices=0;
  calInit();
  {
    CALuint numDevices;
    // Finding number of devices
    if(calDeviceGetCount(&numDevices)!=CAL_RESULT_OK)
      return;
    if(numDevices==0) {
      LogScreen("No supported devices found!");
      //exit(-1);             //TODO: // thou shalt not call exit()!
      return;
    }
    numDevices=1;           //TODO: add multigpu support
    amdstream_numDevices = numDevices;
  }
  if(amdstream_numDevices>AMD_STREAM_MAX_GPUS)
    amdstream_numDevices=AMD_STREAM_MAX_GPUS;
  for(int i=0; i<AMD_STREAM_MAX_GPUS; i++) {
    CContext[i].active=false;
    CContext[i].coreID=CORE_NONE;
    CContext[i].constMem=0;
    CContext[i].constName=0;
    CContext[i].constRes=0;
    CContext[i].ctx=0;
    CContext[i].device=0;
    CContext[i].module=0;
    CContext[i].outName0=0;
    CContext[i].outputMem0=0;
    CContext[i].outputRes0=0;
    CContext[i].obj=NULL;
    CContext[i].image=NULL;

    if(i<amdstream_numDevices) {
      // Opening device
      calDeviceOpen(&CContext[i].device, i);

      // Querying device attribs
      CContext[i].attribs.struct_size = sizeof(CALdeviceattribs);
      if(calDeviceGetAttribs(&CContext[i].attribs, i)!=CAL_RESULT_OK)
        continue;
      CContext[i].active=true;

      CContext[i].domainSizeX=32;
      CContext[i].domainSizeY=32;
      CContext[i].maxIters=512;
    }
  }
}

void AMDStreamReinitializeDevice(int Device)
{
  // Unloading the module
  if(CContext[Device].module)
  {
    calModuleUnload(CContext[Device].ctx, CContext[Device].module);
    CContext[Device].module=0;
  }

  // Freeing compiled program binary
  if(CContext[Device].image) {
    calclFreeImage(CContext[Device].image);
    CContext[Device].image=NULL;
  }
  if(CContext[Device].obj) {
    calclFreeObject(CContext[Device].obj);
    CContext[Device].obj=NULL;
  }

  // Releasing resource from context
  if(CContext[Device].constMem) {
    calCtxReleaseMem(CContext[Device].ctx, CContext[Device].constMem);
    CContext[Device].constMem=0;
  }
  if(CContext[Device].outputMem0) {
    calCtxReleaseMem(CContext[Device].ctx, CContext[Device].outputMem0);
    CContext[Device].outputMem0=0;
  }

  // Deallocating resources
  if(CContext[Device].constRes) {
    calResFree(CContext[Device].constRes);
    CContext[Device].constRes=0;
  }

  // Deallocating resources
  if(CContext[Device].outputRes0) {
    calResFree(CContext[Device].outputRes0);
    CContext[Device].outputRes0=0;
  }

  // Destroying context
  calCtxDestroy(CContext[Device].ctx);
  CContext[Device].coreID=CORE_NONE;
}
