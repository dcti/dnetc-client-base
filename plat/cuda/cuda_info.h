/*
 * Copyright distributed.net 2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_info.h,v 1.4 2011/11/10 02:09:09 snikkel Exp $
*/

#ifndef CUDA_INFO_H
#define CUDA_INFO_H

// returns -1 if not supported
// returns 0 if no supported GPU was found
int GetNumberOfDetectedCUDAGPUs();

long GetRawCUDAGPUID(const char **cpuname);

// returns the frequency in MHz, or 0.
unsigned int GetCUDAGPUFrequency();

// Number of cores per MP (varies with SM version)
// from NVIDIA CUDA SDK sample code 'deviceQuery'
// http://developer.nvidia.com/cuda-cc-sdk-code-samples
typedef struct
{
  int SM; // 0xMm (hexidecimal notation)
          // M = SM Major version, m = SM minor version
  int Cores;
} sSMtoCores;
sSMtoCores CUDACoresPerSM[] =
{
  { 0x10,  8 },
  { 0x11,  8 },
  { 0x12,  8 },
  { 0x13,  8 },
  { 0x20, 32 },
  { 0x21, 48 },
  {   -1, -1 }
};

inline int getCUDACoresPerSM (int major, int minor)
{
  int index = 0;
  int SM = (major << 4) + minor;

  while (CUDACoresPerSM[index].SM != -1) {
    if (CUDACoresPerSM[index].SM == SM ) {
      return CUDACoresPerSM[index].Cores;
    }
    index++;
  }
  return -1;
}

