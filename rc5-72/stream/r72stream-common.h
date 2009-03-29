/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: r72stream-common.h,v 1.7 2009/03/29 20:02:27 andreasb Exp $
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


#define CORE_IL4    1
#define CORE_IL4N   2

u32 cmp72(u32 o1h, u32 o1m, u32 o1l, u32 o2h, u32 o2m, u32 o2l);
void key_incr(u32 *hi, u32 *mid, u32 *lo, u32 incr);

#endif
