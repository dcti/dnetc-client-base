/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: amdstream_context.h,v 1.7 2010/03/18 18:56:49 sla Exp $
*/

#ifndef AMD_STREAM_CONTEXT_H
#define AMD_STREAM_CONTEXT_H

#include "amdstream_setup.h"
#include "cputypes.h"

#include <cal.h>
#include <cal_ext_counter.h>


#define AMD_STREAM_MAX_GPUS 16

#define CORE_NONE   0xffffffff

typedef struct {
  CALdevice device;
  CALdeviceattribs attribs;
  CALcontext ctx;

  CALresource outputRes0, outputRes1;
  CALresource constRes0, constRes1;
  CALresource globalRes0, globalRes1;

  CALmodule module0, module1;

  CALmem outputMem0, outputMem1;
  CALmem constMem0, constMem1;
  CALmem globalMem0, globalMem1;

  CALfunc func0, func1;
  CALname outName0, constName0, outName1, constName1;
  CALname globalName0, globalName1;

  CALimage image;

  CALint domainSizeY;
  CALint domainSizeX;
  unsigned maxIters;

  CALcounter idleCounter;

  u32 coreID;
  bool active;
} stream_context_t;

extern stream_context_t CContext[AMD_STREAM_MAX_GPUS];
extern int atistream_numDevices;

typedef CALresult (CALAPIENTRYP PFNCALCTXWAITFOREVENTS)(CALcontext ctx, CALevent *event, CALuint num, CALuint flags);
extern PFNCALCTXWAITFOREVENTS calCtxWaitForEvents;
extern u32 isCalCtxWaitForEventsSupported;

#endif // AMD_STREAM_CONTEXT_H
