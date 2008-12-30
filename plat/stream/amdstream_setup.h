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
 * $Id: amdstream_setup.h,v 1.2 2008/12/30 17:01:33 andreasb Exp $
*/

#ifndef AMD_STREAM_SETUP_H
#define AMD_STREAM_SETUP_H

#include <cal.h>
#include <calcl.h>

#include "ccoreio.h"
#include <malloc.h>
#include <stdio.h>
#include <string.h>

#include "logstuff.h"  // LogScreen()

#define CORE_NONE   0xffffffff

typedef struct {

  CALdevice device;
  CALdeviceattribs attribs;
  CALcontext ctx;

  CALresource outputRes0;
  CALresource constRes;

  CALmodule module;

  CALmem outputMem0;
  CALmem constMem;

  CALfunc func;
  CALname outName0, constName;

  CALobject obj;
  CALimage image;

  CALint domainSizeY;
  CALint domainSizeX;
  unsigned maxIters;

  u32 coreID;
  bool active;
}stream_context_t;

extern stream_context_t CContext[16];   //MAXCPUS?
extern bool cInit;


u32 getAMDStreamDeviceCount();
u32 getAMDStreamDeviceFreq();
long __GetRawProcessorID(const char **cpuname);
void Deinitialize_rc5_72_il4(u32 Device);

#endif // AMD_STREAM_SETUP_H
