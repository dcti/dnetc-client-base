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
 * $Id: amdstream_info.h,v 1.1 2008/12/30 17:39:36 andreasb Exp $
*/

#ifndef AMD_STREAM_INFO_H
#define AMD_STREAM_INFO_H

u32 getAMDStreamDeviceCount();
u32 getAMDStreamDeviceFreq();
long getAMDStreamRawProcessorID(const char **cpuname);

#endif // AMD_STREAM_INFO_H
