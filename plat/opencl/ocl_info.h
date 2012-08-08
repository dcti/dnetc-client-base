 /*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id:
*/

#ifndef OCL_INFO_H
#define OCL_INFO_H

#include "cputypes.h"
#include <CL/cl.h>

int     getOpenCLDeviceCount();
u32     getOpenCLDeviceFreq(unsigned device=0);
long    getOpenCLRawProcessorID(const char **cpuname, unsigned device=0);
void    OpenCLPrintExtendedGpuInfo(void);

extern cl_uint numPlatforms;
extern cl_platform_id *platforms;
extern cl_int numDevices;
extern cl_device_id *devices;


#endif // OCL_INFO_H
