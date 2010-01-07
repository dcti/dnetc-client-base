/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: amdstream_info.cpp,v 1.8 2010/01/07 09:01:08 sla Exp $
*/

#include "amdstream_info.h"
#include "amdstream_context.h"
#include <stdlib.h>
#include <string.h>

char *ATIstream_GPUname=0L;

u32 getAMDStreamDeviceCount()
{
  return atistream_numDevices;
}

unsigned getAMDStreamDeviceFreq()
{
  if(getAMDStreamDeviceCount()>0)
    return CContext[0].attribs.engineClock;
  else
    return 0;
}

static const char* GetNameById(u32 id)
{
  const char *name = "unknown";
  switch(id) {
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
  case 8: //RV870
    name = "RV870";
    break;
  case 9: //RV840
    name = "RV840";
    break;
  default:
    break;
  }
  return name;
}

long getAMDStreamRawProcessorID(const char **cpuname)
{
  u32 nCPUs;
  const char *curr_name;
  u32 strl;

  nCPUs=getAMDStreamDeviceCount();
  if(nCPUs==0)
  {
    *cpuname=GetNameById(0);
    return 0;
  }
  //ѕодсчитываем количество требуемой пам€ти
  strl=0;
  u32 i;
  for(i=0; i<nCPUs; i++)
  {
    curr_name=GetNameById(CContext[i].attribs.target);
    strl+=strlen(curr_name)+3;          //+", "
  }
  if(ATIstream_GPUname)
  {
    free(ATIstream_GPUname);
    ATIstream_GPUname=0L;
  }
  ATIstream_GPUname=(char*)malloc(strl);
  if(!ATIstream_GPUname)
  {
    *cpuname=GetNameById(0);
    return 0;
  }
  ATIstream_GPUname[0]=0;
  for(i=0; i<nCPUs; i++)
  {
    if(strlen(ATIstream_GPUname)>0)
      strcat(ATIstream_GPUname,", ");
    curr_name=GetNameById(CContext[i].attribs.target);
    strcat(ATIstream_GPUname,curr_name);
  }
  *cpuname=ATIstream_GPUname;
  return CContext[0].attribs.target;
}
