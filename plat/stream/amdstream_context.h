/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: amdstream_context.h,v 1.12 2012/01/14 13:45:09 sla Exp $
*/

#ifndef AMD_STREAM_CONTEXT_H
#define AMD_STREAM_CONTEXT_H

#include "cputypes.h"

#include <CAL/cal.h>
#include <CAL/cal_ext_counter.h>


enum
{
  CORE_NONE, // must be zero
#ifdef HAVE_RC5_72_CORES
  CORE_IL4N,
  CORE_IL4NA,
  CORE_IL42T,
  CORE_IL4_1I,
#endif
#ifdef HAVE_OGR_CORES
  CORE_IL_OGRNG_BASIC,
#endif
  CORE_IL_TOTAL
};

typedef struct {
  int clientDeviceNo;  // client GPU index (for logs)

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

  u32 USEcount;		//# of Unexpected Stop Errors
} stream_context_t;

stream_context_t *stream_get_context(int device);

void AMDStreamReinitializeDevice(stream_context_t *cont);

typedef CALresult (CALAPIENTRYP PFNCALCTXWAITFOREVENTS)(CALcontext ctx, CALevent *event, CALuint num, CALuint flags);
extern PFNCALCTXWAITFOREVENTS calCtxWaitForEvents;
extern u32 isCalCtxWaitForEventsSupported;

#endif // AMD_STREAM_CONTEXT_H
