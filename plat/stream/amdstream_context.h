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
 * $Id: amdstream_context.h,v 1.1 2008/12/30 17:48:38 andreasb Exp $
*/

#ifndef AMD_STREAM_CONTEXT_H
#define AMD_STREAM_CONTEXT_H

#include "amdstream_setup.h"

#include <cal.h>


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
} stream_context_t;

extern stream_context_t CContext[16];   //MAXCPUS?
extern bool cInit;

#endif // AMD_STREAM_CONTEXT_H
