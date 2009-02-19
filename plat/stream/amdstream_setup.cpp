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
 * $Id: amdstream_setup.cpp,v 1.9 2009/02/19 23:19:28 andreasb Exp $
*/

#include "amdstream_setup.h"
#include "amdstream_context.h"
#include "logstuff.h"

#include <calcl.h>
#include <cal_ext.h>
#include <cal_ext_counter.h>

stream_context_t CContext[AMD_STREAM_MAX_GPUS];
int amdstream_numDevices = -1;

PFNCALCTXCREATECOUNTER calCtxCreateCounterExt;
PFNCALCTXDESTROYCOUNTER calCtxDestroyCounterExt;
PFNCALCTXBEGINCOUNTER calCtxBeginCounterExt;
PFNCALCTXENDCOUNTER calCtxEndCounterExt;
PFNCALCTXGETCOUNTER calCtxGetCounterExt;
bool amdstream_usePerfCounters = false;

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

  calCtxCreateCounterExt=NULL;
  calCtxDestroyCounterExt=NULL;
  calCtxBeginCounterExt=NULL;
  calCtxEndCounterExt=NULL;
  calCtxGetCounterExt=NULL;
  amdstream_usePerfCounters=false;
  if (calExtSupported((CALextid)CAL_EXT_COUNTERS) == CAL_RESULT_OK)
  {
    if (calExtGetProc((CALextproc*)&calCtxCreateCounterExt, (CALextid)CAL_EXT_COUNTERS, "calCtxCreateCounter")==CAL_RESULT_OK)
    {
      if (calExtGetProc((CALextproc*)&calCtxDestroyCounterExt, (CALextid)CAL_EXT_COUNTERS, "calCtxDestroyCounter")==CAL_RESULT_OK)
      {
        if (calExtGetProc((CALextproc*)&calCtxBeginCounterExt, (CALextid)CAL_EXT_COUNTERS, "calCtxBeginCounter")==CAL_RESULT_OK)
        {
          if (calExtGetProc((CALextproc*)&calCtxEndCounterExt, (CALextid)CAL_EXT_COUNTERS, "calCtxEndCounter")==CAL_RESULT_OK)
          {
            if (calExtGetProc((CALextproc*)&calCtxGetCounterExt, (CALextid)CAL_EXT_COUNTERS, "calCtxGetCounter")==CAL_RESULT_OK)
            {
              amdstream_usePerfCounters=true;
            }
          }
        }
      }
    }
  }

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
    CContext[i].idleCounter=0;

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
      CContext[i].maxIters=256;
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
