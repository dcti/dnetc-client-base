/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: amdstream_setup.h,v 1.8 2010/04/22 19:09:21 sla Exp $
*/

#ifndef AMD_STREAM_SETUP_H
#define AMD_STREAM_SETUP_H

#define AMD_STREAM_MAX_GPUS 16

void AMDStreamInitialize();
void AMDStreamReinitializeDevice(int Device);

#endif // AMD_STREAM_SETUP_H
