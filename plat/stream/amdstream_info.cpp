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
 * $Id: amdstream_info.cpp,v 1.2 2008/12/30 18:08:50 andreasb Exp $
*/

#include "amdstream_info.h"
#include "amdstream_setup.h"

u32 getAMDStreamDeviceCount()
{
  return amdstream_numDevices;
}

unsigned getAMDStreamDeviceFreq()
{
  return CContext[0].attribs.engineClock;       //TODO:device id
}

long getAMDStreamRawProcessorID(const char **cpuname)
{
  const char *name = "unknown";
  switch(CContext[0].attribs.target) {          //TODO:device id
  case CAL_TARGET_600:
    name = "R600";
    break;
  case CAL_TARGET_610:
    name = "RV610";
    break;
  case CAL_TARGET_630:
    name = "RV630";
    break;
  case CAL_TARGET_670:
    name = "RV670";
    break;
  case CAL_TARGET_7XX:
    name = "R700 class";
    break;
  case CAL_TARGET_770:
    name = "RV770";
    break;
  case CAL_TARGET_710:
    name = "RV710";
    break;
  case CAL_TARGET_730:
    name = "RV730";
    break;
  default:
    break;
  }
  *cpuname=name;
  return CContext[0].attribs.target;            //TODO:device id
}
