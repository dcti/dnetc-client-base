/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: r72stream-common.h,v 1.8 2009/03/29 22:43:29 andreasb Exp $
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

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif


#define CORE_IL4    1
#define CORE_IL4N   2

u32 cmp72(u32 o1h, u32 o1m, u32 o1l, u32 o2h, u32 o2m, u32 o2l);
void key_incr(u32 *hi, u32 *mid, u32 *lo, u32 incr);


#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
// CliTimer does not have fine resolution as required by the core
typedef ui64 hirestimer_type;

inline void HiresTimerGet(hirestimer_type * t)
{
  QueryPerformanceCounter((LARGE_INTEGER *)t);
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
#endif

#endif
