/*
 * Copyright distributed.net 2014 - All rights reserved.
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: cuda_core_count.h,v 1.0 2014/02/23 21:04:04 zebe Exp $
 */
 
#ifndef CUDA_CORE_COUNT_H
#define CUDA_CORE_COUNT_H

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

#endif // CUDA_CORE_COUNT_H