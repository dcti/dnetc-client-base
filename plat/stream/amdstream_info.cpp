/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: amdstream_info.cpp,v 1.11 2010/12/02 19:42:30 sla Exp $
*/

#include "amdstream_info.h"
#include "amdstream_context.h"
#include <stdlib.h>
#include <string.h>

#include "logstuff.h"

u32 getAMDStreamDeviceCount(void)
{
  return atistream_numDevices;
}

unsigned getAMDStreamDeviceFreq(void)
{
  if (getAMDStreamDeviceCount() > 0)
    return CContext[0].attribs.engineClock;

  return 0;
}

static const char* GetNameById(u32 id, u32 nSIMDs=0)
{
  switch (id)
  {
  case CAL_TARGET_600: return "HD2900";
  case CAL_TARGET_610: return "HD2400";
  case CAL_TARGET_630: return "HD2600";
  case CAL_TARGET_670: return "HD3870/HD3850/HD3690";
  case CAL_TARGET_7XX: return "R700 class";
  case CAL_TARGET_770: 
    if(nSIMDs==8)  return "HD4830/HD4860";
    if(nSIMDs==10) return "HD4850/HD4870";
    return "HD48xx";
  case CAL_TARGET_710: return "HD43xx/HD45xx";
  case CAL_TARGET_730: return "HD4650/HD4670";
//  case CAL_TARGET_740: return "HD4770/HD4750";
  case CAL_TARGET_CYPRESS: 
    if(nSIMDs==20) return "HD5870/HD5970";
    if(nSIMDs==18) return "HD5850";
    if(nSIMDs==14) return "HD5830";
    return "HD58xx";
  case CAL_TARGET_JUNIPER:
    if(nSIMDs==10) return "HD5770";
    if(nSIMDs==9)  return "HD5750";
    return "HD57xx";
  case CAL_TARGET_REDWOOD:
    if(nSIMDs==5)  return "HD5670/HD5570";
    if(nSIMDs==4)  return "HD5550";
    return "HD56xx/HD55xx";
  case CAL_TARGET_CEDAR: return "HD54xx";
  case 17:         //Hack for Barts until stream SDK 2.3
    if(nSIMDs==12) return "HD6850";
    if(nSIMDs==14) return "HD6870";
    return "HD68xx";
  default:         return "unknown";
  }
}

long getAMDStreamRawProcessorID(const char **cpuname)
{
  u32 nCPUs;
  u32 i, amount;
  static char *ATIstream_GPUname;

  if (ATIstream_GPUname)
  {
    *cpuname = ATIstream_GPUname;
    return CContext[0].attribs.target;
  }

  nCPUs = getAMDStreamDeviceCount();
  if (nCPUs == 0)
  {
    *cpuname = GetNameById(0);
    return 0;
  }

  // Calculate amount of required memory
  for (i=amount=0; i < nCPUs; i++)
    amount += strlen(GetNameById(CContext[i].attribs.target, CContext[i].attribs.numberOfSIMD)) + 3; /* include ", " */

  ATIstream_GPUname = (char*)malloc(amount);
  if (!ATIstream_GPUname)
  {
    *cpuname = GetNameById(0);
    return 0;
  }

  ATIstream_GPUname[0]=0;
  for (i = 0; i < nCPUs; i++)
  {
    if (i != 0)
      strcat(ATIstream_GPUname, ", ");
    strcat(ATIstream_GPUname, GetNameById(CContext[i].attribs.target, CContext[i].attribs.numberOfSIMD));
  }

  *cpuname = ATIstream_GPUname;
  return CContext[0].attribs.target;
}

void AMDStreamPrintExtendedGpuInfo(void)
{
  int i;

  if (atistream_numDevices <= 0)
  {
    LogRaw("No supported devices found\n");
    return;
  }
  for (i = 0; i < atistream_numDevices; i++)
  {
    stream_context_t *dev = &CContext[i];

    LogRaw("\n");
    if (!dev->active)
    {
      LogRaw("Warning: device %d not activated\n", i);
      continue;
    }

    LogRaw("GPU %d attributes (EEEEEEEE == undefined):\n", i);
#define sh(name) LogRaw("%24s: %08X (%d)\n", #name, dev->attribs.##name, dev->attribs.##name)
    sh(target);
    sh(targetRevision);
    sh(localRAM);
    sh(uncachedRemoteRAM);
    sh(cachedRemoteRAM);
    sh(engineClock);
    sh(memoryClock);
    sh(wavefrontSize);
    sh(numberOfSIMD);
    sh(doublePrecision);
    sh(localDataShare);
    sh(globalDataShare);
    sh(globalGPR);
    sh(computeShader);
    sh(memExport);
    sh(pitch_alignment);
    sh(surface_alignment);
    sh(numberOfUAVs);
    sh(bUAVMemExport);
    sh(b3dProgramGrid);
    sh(numberOfShaderEngines);
#undef sh
  }
}
