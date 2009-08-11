/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: amdstream_context.h,v 1.5 2009/08/11 17:27:34 sla Exp $
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

  CALresource outputRes0;
  CALresource constRes0;
  CALresource globalRes0;

  CALmodule module0;

  CALmem outputMem0;
  CALmem constMem0;
  CALmem globalMem0;

  CALfunc func0;
  CALname outName0, constName0;
  CALname globalName0;

  CALimage image;

  CALint domainSizeY;
  CALint domainSizeX;
  unsigned maxIters;

  CALcounter idleCounter;

  u32 coreID;
  bool active;
} stream_context_t;

extern stream_context_t CContext[AMD_STREAM_MAX_GPUS];
extern int amdstream_numDevices;

#endif // AMD_STREAM_CONTEXT_H
