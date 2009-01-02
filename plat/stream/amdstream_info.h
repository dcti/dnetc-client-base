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
 * $Id: amdstream_info.h,v 1.2 2009/01/02 03:25:28 andreasb Exp $
*/

#ifndef AMD_STREAM_INFO_H
#define AMD_STREAM_INFO_H

#include "cputypes.h"

u32 getAMDStreamDeviceCount();
u32 getAMDStreamDeviceFreq();
long getAMDStreamRawProcessorID(const char **cpuname);

#endif // AMD_STREAM_INFO_H
