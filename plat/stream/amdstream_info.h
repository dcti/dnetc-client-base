/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
*/

#ifndef AMD_STREAM_INFO_H
#define AMD_STREAM_INFO_H

#include "cputypes.h"

int  getAMDStreamDeviceCount(void);
u32  getAMDStreamDeviceFreq(int device);
long getAMDStreamRawProcessorID(int device, const char **cpuname);
void AMDStreamPrintExtendedGpuInfo(int device);

#endif // AMD_STREAM_INFO_H
