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
 * $Id: amdstream_info.cpp,v 1.1 2008/12/30 17:39:36 andreasb Exp $
*/

#include "amdstream_info.h"
#include "amdstream_setup.h"

u32 getAMDStreamDeviceCount()
{
  return numDevices;
}

unsigned getAMDStreamDeviceFreq()
{
  return CContext[0].attribs.engineClock;       //TODO:device id
}

long getAMDStreamRawProcessorID(const char **cpuname)
{
  static char buf[30];
  buf[0]='\0';
  switch(CContext[0].attribs.target) {          //TODO:device id
  case CAL_TARGET_600:
    strcpy(buf,"R600");
    break;
  case CAL_TARGET_610:
    strcpy(buf,"RV610");
    break;
  case CAL_TARGET_630:
    strcpy(buf,"RV630");
    break;
  case CAL_TARGET_670:
    strcpy(buf,"RV670");
    break;
  case CAL_TARGET_7XX:
    strcpy(buf,"R700 class");
    break;
  case CAL_TARGET_770:
    strcpy(buf,"RV770");
    break;
  case CAL_TARGET_710:
    strcpy(buf,"RV710");
    break;
  case CAL_TARGET_730:
    strcpy(buf,"RV730");
    break;
  default:
    break;
  }
  *cpuname=buf;
  return CContext[0].attribs.target;            //TODO:device id
}
