/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: r72stream-common.h,v 1.11 2009/08/11 17:19:31 sla Exp $
*/

#ifndef IL_COMMON_H
#define IL_COMMON_H

#include "amdstream_setup.h"
#include "amdstream_context.h"

#include <cal.h>
#include <calcl.h>

#include "ccoreio.h"
#include <malloc.h>
#include <stdio.h>
#include <string.h>

#include "logstuff.h"  // LogScreen()
#include "pollsys.h"   // NonPolledUsleep()
#include "clisync.h"   // fastlock
#include "triggers.h"  // RaiseExitRequestTrigger()

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define CORE_IL4N   1
#define CORE_IL4NA  2


void key_incr(u32 *hi, u32 *mid, u32 *lo, u32 incr);
CALresult compileProgram(CALcontext *ctx, CALimage *image, CALmodule *module, CALchar *src, CALtarget target);


#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
// CliTimer does not have fine resolution as required by the core
typedef ui64 hirestimer_type;

inline void HiresTimerGet(hirestimer_type * t)
{
  QueryPerformanceCounter((LARGE_INTEGER *)t);
}

inline double HiresTimerGetResolution()
{
  LARGE_INTEGER f;
  double fr_d;
  ui64 fr;

  QueryPerformanceFrequency(&f);
  fr=f.HighPart; fr<<=32; fr+=f.LowPart;
  fr_d=(double)fr/1000.f;

   return fr_d;
}

//returns ???
inline double HiresTimerDiff(const hirestimer_type t1, const hirestimer_type t2)
{
  return (double)(t1 < t2 ? t2 - t1 : t1 - t2);
}
#else
typedef struct timeval hirestimer_type;

inline void HiresTimerGet(hirestimer_type * t)
{
  CliTimer(t);
}

// returns milliseconds
inline double HiresTimerDiff(const hirestimer_type t1, const hirestimer_type t2)
{
  hirestimer_type diff;
  CliTimerDiff(&diff, &t1, &t2);
  return (double)diff.tv_sec * 1000.0 + (double)diff.tv_usec / 1000.0;
}

inline double HiresTimerGetResolution()
{
  return 1.f;
}

#endif

#endif
