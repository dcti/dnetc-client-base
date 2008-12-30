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
 * $Id: r72stream-common.h,v 1.1 2008/12/30 16:26:45 andreasb Exp $
*/

#ifndef IL_COMMON_H
#define IL_COMMON_H

#include <cal.h>
#include <calcl.h>
 
#include "ccoreio.h"
#include <malloc.h>
#include <stdio.h>
#include <string.h>

#include "logstuff.h"  // LogScreen()

#define NULL 0

#define CORE_IL4	0
#define CORE_IL4C	1
#define CORE_IL4N	2
#define CORE_NONE	0xffffffff

typedef struct{

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

	u32		coreID;
	bool	active;
}stream_context_t;

extern stream_context_t CContext[16];	//MAXCPUS?
extern bool cInit;


u32 getAMDStreamDeviceCount();
u32 getAMDStreamDeviceFreq();
long __GetRawProcessorID(const char **cpuname);
u32 sub72(u32 t_hi, u32 t_mid, u32 s_hi, u32 s_mid);
u32 cmp72(u32 o1h, u32 o1m, u32 o1l, u32 o2h, u32 o2m, u32 o2l);
void key_incr(unsigned *hi, unsigned *mid, unsigned *lo, unsigned incr);
void Deinitialize_rc5_72_il4(u32 Device);

#endif
