/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: amdstream_info.cpp,v 1.9 2010/05/09 10:42:17 stream Exp $
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

static const char* GetNameById(u32 id)
{
  switch (id)
  {
  case CAL_TARGET_600: return "R600";
  case CAL_TARGET_610: return "RV610";
  case CAL_TARGET_630: return "RV630";
  case CAL_TARGET_670: return "RV670";
  case CAL_TARGET_7XX: return "R700 class";
  case CAL_TARGET_770: return "RV770";
  case CAL_TARGET_710: return "RV710";
  case CAL_TARGET_730: return "RV730";
  case 8: /* RV870 */  return "RV870";
  case 9: /* RV840 */  return "RV840";
  default:             return "unknown";
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
    amount += strlen(GetNameById(CContext[i].attribs.target)) + 3; /* include ", " */

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
    strcat(ATIstream_GPUname, GetNameById(CContext[i].attribs.target));
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
    sh(targetRevision);
#undef sh
  }
}
